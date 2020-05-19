#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/mpi_consensus_algorithms.templates.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/rtree.h>

using namespace dealii;

template <int dim, int spacedim>
class RemoteQuadraturePointEvaluator
{
public:
  /**
   * Constructor.
   *
   * @param quadrature_points quadrature points at which function should be
   *   evaluated
   * @param tria_fluid (partitioned) triangulation on which results should be
   *   evaluated
   */
  RemoteQuadraturePointEvaluator(
    const std::vector<Point<spacedim>> quadrature_points,
    const parallel::distributed::Triangulation<dim, spacedim> &tria)
    : comm(tria.get_communicator())
  {
    // create bounding boxed of local active cells
    std::vector<BoundingBox<spacedim>> local_boxes;
    for (const auto cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        local_boxes.push_back(cell->bounding_box());

    // create r-tree of bounding boxes
    const auto local_tree = pack_rtree(local_boxes);

    // compress r-tree to a minimal set of bounding boxes
    const auto local_reduced_box = extract_rtree_level(local_tree, 0);

    // gather bounding boxes of other processes
    const auto global_bounding_boxes =
      Utilities::MPI::all_gather(comm, local_reduced_box);

    // determine ranks which might posses quadrature point
    std::vector<std::vector<Point<spacedim>>> points_per_process(
      global_bounding_boxes.size());

    std::vector<std::vector<unsigned int>> points_per_process_offset(
      global_bounding_boxes.size());

    for (unsigned int i = 0; i < quadrature_points.size(); ++i)
      {
        const auto &point = quadrature_points[i];
        for (unsigned rank = 0; rank < global_bounding_boxes.size(); ++rank)
          for (const auto &box : global_bounding_boxes[rank])
            if (box.point_inside(point))
              {
                points_per_process[rank].emplace_back(point);
                points_per_process_offset[rank].emplace_back(i);
              }
      }

    // only communicate with processes that might have a quadrature point
    std::vector<unsigned int> targets;

    for (unsigned int i = 0; i < points_per_process.size(); ++i)
      if (points_per_process[i].size() > 0 &&
          i != Utilities::MPI::this_mpi_process(comm))
        targets.emplace_back(i);

    // for local quadrature points no communication is needed...
    local_quadrature_points =
      points_per_process[Utilities::MPI::this_mpi_process(comm)];


    std::map<unsigned int, std::vector<Point<spacedim>>>
      relevant_points_per_process;
    std::map<unsigned int, std::vector<unsigned int>>
      relevant_points_per_process_offset;
    std::map<unsigned int, std::vector<unsigned int>>
      relevant_points_per_process_count;

    // send to remote ranks the requested quadrature points and eliminate
    // not needed ones (note: currently, we cannot communicate points ->
    // switch to doubles here)
    Utilities::MPI::ConsensusAlgorithms::AnonymousProcess<double, unsigned int>
      process(
        [&]() { return targets; },
        [&](const unsigned int other_rank, std::vector<double> &send_buffer) {
          // send requested points
          for (auto point : points_per_process[other_rank])
            for (unsigned int i = 0; i < spacedim; ++i)
              send_buffer.emplace_back(point[i]);
        },
        [&](const unsigned int &       other_rank,
            const std::vector<double> &recv_buffer,
            std::vector<unsigned int> &request_buffer) {
          // received points, determine if point is actually possessed, and
          // send the result back

          std::vector<Point<spacedim>> relevant_remote_points;
          std::vector<unsigned int>    relevant_remote_points_count;

          request_buffer.clear();
          request_buffer.resize(recv_buffer.size() / spacedim);

          for (unsigned int i = 0, j = 0; i < recv_buffer.size();
               i += spacedim, ++j)
            {
              Point<spacedim> point;
              for (unsigned int j = 0; j < spacedim; ++j)
                point[j] = recv_buffer[i + j];

              unsigned int counter = j % 4; // TODO

              request_buffer[j] = counter;

              if (counter > 0)
                {
                  relevant_remote_points.push_back(point);
                  relevant_remote_points_count.push_back(counter);
                }
            }

          if (relevant_remote_points.size() > 0)
            {
              relevant_remote_points_per_process[other_rank] =
                relevant_remote_points;
              relevant_remote_points_count_per_process[other_rank] =
                relevant_remote_points_count;
            }
        },
        [&](const unsigned int         other_rank,
            std::vector<unsigned int> &recv_buffer) {
          // prepare buffer
          recv_buffer.resize(points_per_process[other_rank].size());
        },
        [&](const unsigned int               other_rank,
            const std::vector<unsigned int> &recv_buffer) {
          // store recv_buffer -> make the algorithm deterministic

          const auto &potentially_relevant_points =
            points_per_process[other_rank];
          const auto &potentially_relevant_points_offset =
            points_per_process_offset[other_rank];

          std::vector<Point<spacedim>> points;
          std::vector<unsigned int>    points_offset;
          std::vector<unsigned int>    count;

          AssertDimension(potentially_relevant_points.size(),
                          recv_buffer.size());
          AssertDimension(potentially_relevant_points_offset.size(),
                          recv_buffer.size());

          for (unsigned int i = 0; i < recv_buffer.size(); ++i)
            if (recv_buffer[i] > 0)
              {
                points.push_back(potentially_relevant_points[i]);
                points_offset.push_back(potentially_relevant_points_offset[i]);
                count.push_back(recv_buffer[i]);
              }

          if (points.size() > 0)
            {
              relevant_points_per_process[other_rank]        = points;
              relevant_points_per_process_offset[other_rank] = points_offset;
              relevant_points_per_process_count[other_rank]  = count;
              map_recv[other_rank]                           = points;
            }
        });

    Utilities::MPI::ConsensusAlgorithms::Selector(process, comm).run();

    quadrature_points_count.resize(quadrature_points.size(), 0);

    for (const auto &i : relevant_points_per_process)
      {
        const unsigned int rank = i.first;

        std::vector<std::pair<unsigned int, unsigned int>> indices;

        const auto &relevant_points = relevant_points_per_process[rank];
        const auto &relevant_points_offset =
          relevant_points_per_process_offset[rank];
        const auto &relevant_points_count =
          relevant_points_per_process_count[rank];

        for (unsigned int j = 0; j < relevant_points.size(); ++j)
          {
            for (unsigned int k = 0; k < relevant_points_count[j]; ++k)
              {
                AssertIndexRange(relevant_points_offset[j],
                                 quadrature_points_count.size());
                auto &qp_counter =
                  quadrature_points_count[relevant_points_offset[j]];
                indices.emplace_back(relevant_points_offset[j], qp_counter);

                ++qp_counter;
              }
          }

        this->indices_per_process[rank] = indices;
      }
  }

  template <typename T>
  void
  init_surface_values(std::vector<std::vector<T>> &output) const
  {
    output.resize(quadrature_points_count.size());

    for (unsigned int i = 0; i < quadrature_points_count.size(); ++i)
      output[i].resize(quadrature_points_count[i]);
  }

  template <typename T>
  void
  init_intermediate_values(
    std::map<unsigned int, std::vector<std::vector<T>>> &input) const
  {
    for (const auto &i : this->relevant_remote_points_count_per_process)
      {
        const unsigned int rank = i.first;

        std::vector<std::vector<T>> temp(i.second.size());

        for (unsigned int j = 0; j < i.second.size(); ++j)
          temp.resize(i.second[j]);

        input[rank] = temp;
      }
  }

  /**
   * Evaluate function @p fu in the requested quadrature points. The result
   * is sorted according to rank.
   */
  template <typename T>
  void
  process(const std::map<unsigned int, std::vector<std::vector<T>>> &input,
          std::vector<std::vector<T>> &output) const
  {
    // process remote quadrature points and send them away
    std::map<unsigned int, std::vector<char>> temp_map;

    std::vector<MPI_Request> requests;
    requests.reserve(input.size());

    const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

    for (const auto &vec : input)
      {
        if (vec.first == my_rank)
          continue;

        temp_map[vec.first] = Utilities::pack(vec.second);

        auto &buffer = temp_map[vec.first];

        requests.resize(requests.size() + 1);

        MPI_Isend(buffer.data(),
                  buffer.size(),
                  MPI_CHAR,
                  vec.first,
                  11,
                  comm,
                  &requests.back());
      }

    // receive result

    std::map<unsigned int, std::vector<std::vector<T>>> temp_recv_map;
    // temp_recv_map[my_rank] = input[my_rank];  //TODO

    for (unsigned int counter = 0; counter < map_recv.size(); ++counter)
      {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, 11, comm, &status);

        int message_length;
        MPI_Get_count(&status, MPI_CHAR, &message_length);

        std::vector<char> buffer(message_length);

        MPI_Recv(buffer.data(),
                 buffer.size(),
                 MPI_CHAR,
                 status.MPI_SOURCE,
                 11,
                 comm,
                 MPI_STATUS_IGNORE);

        temp_recv_map[status.MPI_SOURCE] =
          Utilities::unpack<std::vector<std::vector<T>>>(buffer);
      }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    for (const auto &i : temp_recv_map)
      {
        const unsigned int rank = i.first;

        auto it = indices_per_process.at(rank).begin();

        for (const auto &i : temp_recv_map[rank])
          for (const auto &j : i)
            {
              output[it->first][it->second] = j;
              ++it;
            }
      }
  }

  /**
   * Return quadrature points (sorted according to rank).
   */
  std::map<unsigned int, std::vector<Point<spacedim>>>
  get_quadrature_points() const
  {
    auto result = map_recv;

    result[Utilities::MPI::this_mpi_process(comm)] =
      this->local_quadrature_points;

    return result;
  }

  /**
   * Print internal data structures.
   */
  void
  print() const
  {
    std::cout << "Locally owned quadrature points:" << std::endl;

    for (const auto &point : local_quadrature_points)
      std::cout << point << std::endl;
    std::cout << std::endl;

    for (const auto &i : map_recv)
      {
        std::cout << "Receive from " << i.first << ":" << std::endl;

        for (const auto &point : i.second)
          std::cout << point << std::endl;

        std::cout << std::endl;
      }

    for (const auto &i : relevant_remote_points_per_process)
      {
        std::cout << "Send to " << i.first << ":" << std::endl;

        for (const auto &point : i.second)
          std::cout << point << std::endl;

        std::cout << std::endl;
      }
  }

private:
  const MPI_Comm &comm;

  // receiver side
  std::vector<unsigned int> quadrature_points_count;
  std::map<unsigned int, std::vector<std::pair<unsigned int, unsigned int>>>
                                                       indices_per_process;
  std::map<unsigned int, std::vector<Point<spacedim>>> map_recv;

  // sender side (TODO: merge)
  std::vector<Point<spacedim>> local_quadrature_points;
  std::map<unsigned int, std::vector<Point<spacedim>>>
    relevant_remote_points_per_process;
  std::map<unsigned int, std::vector<unsigned int>>
    relevant_remote_points_count_per_process;
};

template <int dim, int spacedim>
void
test(const MPI_Comm &comm)
{
  // Ia) create mesh
  parallel::distributed::Triangulation<dim, spacedim> tria_solid(comm);
  GridGenerator::hyper_ball(tria_solid);
  tria_solid.refine_global(3);

  parallel::distributed::Triangulation<dim, spacedim> tria_fluid(comm);
  GridGenerator::hyper_cube(tria_fluid, -2, 3);
  tria_fluid.refine_global(4);

  GridOut grid_out;

  grid_out.write_mesh_per_processor_as_vtu(tria_solid, "tria_solid");
  grid_out.write_mesh_per_processor_as_vtu(tria_fluid, "tria_fluid");

  // 2) collect quadrature points on surface of solid
  std::vector<Point<spacedim>> surface_quadrature_points;

  MappingQ<dim, spacedim> mapping(2);
  FE_Q<dim, spacedim>     fe(1);
  QGauss<dim - 1>         quad(3);

  FEFaceValues<dim, spacedim> fe_face_values(mapping,
                                             fe,
                                             quad,
                                             update_quadrature_points);

  for (auto cell : tria_solid.active_cell_iterators())
    {
      if (cell->is_locally_owned() == false)
        continue;

      for (auto face : GeometryInfo<dim>::face_indices())
        {
          if (cell->at_boundary(face) == false)
            continue;

          fe_face_values.reinit(cell, face);

          surface_quadrature_points.insert(
            surface_quadrature_points.end(),
            fe_face_values.get_quadrature_points().begin(),
            fe_face_values.get_quadrature_points().end());
        }
    }

  // Ib) setup communication pattern
  const RemoteQuadraturePointEvaluator<dim, spacedim> eval(
    surface_quadrature_points, tria_fluid);

  // Ic) allocate memory
  std::map<unsigned int, std::vector<std::vector<Tensor<1, spacedim>>>>
    intermediate_values;
  eval.init_intermediate_values(intermediate_values);

  std::vector<std::vector<Tensor<1, spacedim>>> surface_values;
  eval.init_surface_values(surface_values);

  // IIa) fill intermediate values (TODO)

  // IIb) communicate values
  eval.process(/*src=*/intermediate_values, /*dst=*/surface_values);

  // IIc) use surface values
  for (unsigned int i = 0; i < surface_values.size(); ++i)
    {
      std::cout << surface_quadrature_points[i] << " ";

      for (const auto &value : surface_values[i])
        std::cout << value << " ";

      std::cout << std::endl;
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<2, 2>(MPI_COMM_WORLD);
}