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
    const auto local_tree        = pack_rtree(local_boxes);
    
    // compress r-tree to a minimal set of bounding boxes
    const auto local_reduced_box = extract_rtree_level(local_tree, 0);

    // gather bounding boxes of other processes
    const auto global_bounding_boxes =
      Utilities::MPI::all_gather(comm, local_reduced_box);

    // determine ranks which might posses quadrature point 
    std::vector<std::vector<Point<spacedim>>> points_per_process(
      global_bounding_boxes.size());

    for (const auto &point : quadrature_points)
      for (unsigned i = 0; i < global_bounding_boxes.size(); ++i)
        for (const auto &box : global_bounding_boxes[i])
          if (box.point_inside(point))
            points_per_process[i].emplace_back(point);

    // only communicate with processes that might have a quadrature point
    std::vector<unsigned int> targets;

    for (unsigned int i = 0; i < points_per_process.size(); ++i)
      if (points_per_process[i].size() > 0 &&
          i != Utilities::MPI::this_mpi_process(comm))
        targets.emplace_back(i);

    // for local quadrature points no communication is needed...
    local_quadrature_points =
      points_per_process[Utilities::MPI::this_mpi_process(comm)];

    // send to remote ranks the requested quadrature points and eliminate
    // not needed ones (note: currently, we cannot communicate points ->
    // switch to doubles here)
    Utilities::MPI::ConsensusAlgorithms::AnonymousProcess<double, double>
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
            std::vector<double> &      request_buffer) {
          // received points, determine if point is actually possessed, and
          // send the result back
            
          std::vector<Point<spacedim>> points;

          request_buffer.resize(recv_buffer.size());

          for (unsigned int i = 0; i < recv_buffer.size(); i += spacedim)
            {
              Point<spacedim> point;
              for (unsigned int j = 0; j < spacedim; ++j)
                point[j] = recv_buffer[i + j];

              bool found = false;

              for (const auto &box : local_boxes)
                if (box.point_inside(point))
                  {
                    points.push_back(point);

                    for (unsigned int j = 0; j < spacedim; ++j)
                      request_buffer[i + j] = recv_buffer[i + j];


                    found = true;

                    break;
                  }

              if (found)
                continue;

              for (unsigned int j = 0; j < spacedim; ++j)
                request_buffer[i + j] = std::numeric_limits<double>::max();
            }

          if (points.size() > 0)
            map_send[other_rank] = points;
        },
        [&](const unsigned int other_rank, std::vector<double> &recv_buffer) {
            // prepare buffer
          recv_buffer.resize(spacedim * points_per_process[other_rank].size());
        },
        [&](const unsigned int         other_rank,
            const std::vector<double> &recv_buffer) {
          // receive points, which are actually owned by other process
          std::vector<Point<spacedim>> points;

          for (unsigned int i = 0; i < recv_buffer.size(); i += spacedim)
            {
              if (recv_buffer[i] == std::numeric_limits<double>::max())
                continue;

              Point<spacedim> point;
              for (unsigned int j = 0; j < spacedim; ++j)
                point[j] = recv_buffer[i + j];

              points.push_back(point);
            }

          if (points.size() > 0)
            map_recv[other_rank] = points;
        });

    Utilities::MPI::ConsensusAlgorithms::Selector<double, double>(
      process, tria.comm)
      .run();
  }

  /**
   * Evaluate function @p fu in the requested quadrature points. The result
   * is sorted according to rank.
   */
  template <typename T>
  std::map<unsigned int, std::vector<T>>
  process(const std::function<T(Point<spacedim>)> &fu) const
  {
    std::map<unsigned int, std::vector<T>> result;

    // process local quadrature points
    {
      std::vector<T> temp;
      temp.reserve(local_quadrature_points.size());
      for (auto p : local_quadrature_points)
        temp.push_back(fu(p));
      result[Utilities::MPI::this_mpi_process(comm)] = temp;
    }

    // process remote quadrature points and send them away
    std::map<unsigned int, std::vector<char>> temp_map;

    std::vector<MPI_Request> requests;
    requests.reserve(map_send.size());

    for (const auto &vec : map_send)
      {
        std::vector<T> temp;
        temp.reserve(vec.second.size());

        for (auto p : vec.second)
          temp.push_back(fu(p));
        temp_map[vec.first] = Utilities::pack(temp);

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

        result[status.MPI_SOURCE] = Utilities::unpack<std::vector<T>>(buffer);
      }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    return result;
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

    for (const auto &i : map_send)
      {
        std::cout << "Send to " << i.first << ":" << std::endl;

        for (const auto &point : i.second)
          std::cout << point << std::endl;

        std::cout << std::endl;
      }
  }

private:
  const MPI_Comm &comm;

  std::vector<Point<spacedim>>                         local_quadrature_points;
  std::map<unsigned int, std::vector<Point<spacedim>>> map_recv;
  std::map<unsigned int, std::vector<Point<spacedim>>> map_send;
};

template <int dim, int spacedim>
void
test(const MPI_Comm &comm)
{
  // 1) create mesh
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

  // 3) setup communication pattern
  RemoteQuadraturePointEvaluator<dim, spacedim> eval(surface_quadrature_points,
                                                     tria_fluid);

  // 4) get quadrature points (sorted according rank)
  std::map<unsigned int, std::vector<Point<spacedim>>> results_points =
    eval.get_quadrature_points();

  // 5) compute derived quantities at quadrature points and get results (sorted
  // according rank)
  std::map<unsigned int, std::vector<Tensor<1, spacedim>>> results_values =
    eval.template process<Tensor<1, spacedim>>(
      [](const auto &point) { return point; });

  // 6) print results
  for (const auto &ranks_and_points : results_points)
    {
      const auto &values = results_values[ranks_and_points.first];

      for (unsigned int i = 0; i < values.size(); ++i)
        std::cout << ranks_and_points.second[i] << " " << values[i]
                  << std::endl;
      std::cout << std::endl;
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<2, 2>(MPI_COMM_WORLD);
}