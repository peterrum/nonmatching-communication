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
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/rtree.h>

using namespace dealii;

double const X_0    = 0.0;  // origin (x-coordinate)
double const Y_0    = 0.0;  // origin (y-coordinate)
double const L      = 2.5;  // x-coordinate of outflow boundary
double const H      = 0.41; // height of channel
double const X_C    = 0.2;  // center of cylinder (x-coordinate)
double const Y_C    = 0.2;  // center of cylinder (y-coordinate)
double const X_2    = 2.0 * X_C;
double const D      = 0.1;                    // cylinder diameter
double const R      = D / 2.0;                // cylinder radius
double const T      = 0.02;                   // thickness of flag
double const L_FLAG = 0.35;                   // length of flag
double const X_3    = X_C + R + L_FLAG * 1.6; // only relevant for mesh
double const Y_3    = H / 3.0;                // only relevant for mesh

bool STRUCTURE_COVERS_FLAG_ONLY = true;


void create_triangulation_structure(Triangulation<2> &tria)
{
  if (STRUCTURE_COVERS_FLAG_ONLY)
    {
      GridGenerator::subdivided_hyper_rectangle(
        tria,
        {8, 1} /* subdivisions x,y */,
        Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
        Point<2>(X_C + R + L_FLAG, Y_C + T / 2.0));
    }
  else
    {
      std::vector<Triangulation<2>> tria_vec;

      tria_vec.resize(12);

      GridGenerator::general_cell(
        tria_vec[0],
        {Point<2>(X_C + R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
         Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
         Point<2>(X_C + T / 2.0, Y_C - T),
         Point<2>(X_C + T / 2.0 +
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C - T / 4.0)});

      GridGenerator::general_cell(
        tria_vec[1],
        {Point<2>(X_C + T / 2.0, Y_C - T),
         Point<2>(X_C + T / 2.0 +
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C - T / 4.0),
         Point<2>(X_C + T / 2.0, Y_C + T),
         Point<2>(X_C + T / 2.0 +
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C + T / 4.0)});

      GridGenerator::general_cell(
        tria_vec[2],
        {Point<2>(X_C + T / 2.0 +
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C - T / 4.0),
         Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
         Point<2>(X_C + T / 2.0 +
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C + T / 4.0),
         Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))),
                  Y_C + T / 2.0)});

      GridGenerator::general_cell(
        tria_vec[3],
        {Point<2>(X_C + T / 2.0 +
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C + T / 4.0),
         Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C + T / 2.0),
         Point<2>(X_C + T / 2.0, Y_C + T),
         Point<2>(X_C + R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0))});

      GridGenerator::general_cell(tria_vec[4],
                                  {Point<2>(X_C - T / 2.0, Y_C - T),
                                   Point<2>(X_C + T / 2.0, Y_C - T),
                                   Point<2>(X_C - T / 2.0, Y_C + T),
                                   Point<2>(X_C + T / 2.0, Y_C + T)});

      GridGenerator::general_cell(
        tria_vec[5],
        {Point<2>(X_C - R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
         Point<2>(X_C + R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
         Point<2>(X_C - T / 2.0, Y_C - T),
         Point<2>(X_C + T / 2.0, Y_C - T)});

      GridGenerator::general_cell(
        tria_vec[6],
        {Point<2>(X_C - T / 2.0, Y_C + T),
         Point<2>(X_C + T / 2.0, Y_C + T),
         Point<2>(X_C - R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
         Point<2>(X_C + R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0))});

      GridGenerator::general_cell(
        tria_vec[7],
        {Point<2>(X_C - R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
         Point<2>(X_C - T / 2.0, Y_C - T),
         Point<2>(X_C - R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
         Point<2>(X_C - T / 2.0 -
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C - T / 4.0)});

      GridGenerator::general_cell(
        tria_vec[8],
        {Point<2>(X_C - T / 2.0 -
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C - T / 4.0),
         Point<2>(X_C - T / 2.0, Y_C - T),
         Point<2>(X_C - T / 2.0 -
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C + T / 4.0),
         Point<2>(X_C - T / 2.0, Y_C + T)});

      GridGenerator::general_cell(
        tria_vec[9],
        {Point<2>(X_C - R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
         Point<2>(X_C - T / 2.0 -
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C - T / 4.0),
         Point<2>(X_C - R * std::cos(std::asin(T / (2.0 * R))), Y_C + T / 2.0),
         Point<2>(X_C - T / 2.0 -
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C + T / 4.0)});

      GridGenerator::general_cell(
        tria_vec[10],
        {Point<2>(X_C - R * std::cos(std::asin(T / (2.0 * R))), Y_C + T / 2.0),
         Point<2>(X_C - T / 2.0 -
                    (X_C + R * std::cos(std::asin(T / (2.0 * R))) -
                     (X_C + T / 2.0)) /
                      2.0,
                  Y_C + T / 4.0),
         Point<2>(X_C - R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
         Point<2>(X_C - T / 2.0, Y_C + T)});

      GridGenerator::subdivided_hyper_rectangle(
        tria_vec[11],
        {8, 1} /* subdivisions x,y */,
        Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
        Point<2>(X_C + R + L_FLAG, Y_C + T / 2.0));

      std::vector<Triangulation<2> const *> tria_vec_ptr(tria_vec.size());
      for (unsigned int i = 0; i < tria_vec.size(); ++i)
        tria_vec_ptr[i] = &tria_vec[i];

      GridGenerator::merge_triangulations(tria_vec_ptr, tria);
    }
}

void create_triangulation_fluid(Triangulation<2> &tria)
{
  std::vector<Triangulation<2>> tria_vec;
  tria_vec.resize(11);

  GridGenerator::general_cell(
    tria_vec[0],
    {Point<2>(X_0, 0.0),
     Point<2>(X_C - R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
     Point<2>(X_0, H),
     Point<2>(X_C - R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0))});

  GridGenerator::general_cell(
    tria_vec[1],
    {Point<2>(X_0, 0.0),
     Point<2>(X_2, 0.0),
     Point<2>(X_C - R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
     Point<2>(X_C + R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0))});

  GridGenerator::general_cell(
    tria_vec[2],
    {Point<2>(X_C - R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
     Point<2>(X_C + R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
     Point<2>(X_0, H),
     Point<2>(X_2, H)});

  GridGenerator::general_cell(
    tria_vec[3],
    {Point<2>(X_C + R / std::sqrt(2.0), Y_C - R / std::sqrt(2.0)),
     Point<2>(X_2, 0.0),
     Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C - T / 2.0),
     Point<2>(X_2, Y_C - T / 2.0)});

  GridGenerator::general_cell(
    tria_vec[4],
    {Point<2>(X_C + R * std::cos(std::asin(T / (2.0 * R))), Y_C + T / 2.0),
     Point<2>(X_2, Y_C + T / 2.0),
     Point<2>(X_C + R / std::sqrt(2.0), Y_C + R / std::sqrt(2.0)),
     Point<2>(X_2, H)});

  GridGenerator::subdivided_hyper_rectangle(tria_vec[5],
                                            {1, 1} /* subdivisions x,y */,
                                            Point<2>(X_2, 0.0),
                                            Point<2>(X_C + R + L_FLAG,
                                                     Y_C - T / 2.0));

  GridGenerator::subdivided_hyper_rectangle(tria_vec[6],
                                            {1, 1} /* subdivisions x,y */,
                                            Point<2>(X_2, Y_C + T / 2.0),
                                            Point<2>(X_C + R + L_FLAG, H));

  GridGenerator::general_cell(tria_vec[7],
                              {Point<2>(X_C + R + L_FLAG, 0.0),
                               Point<2>(X_3, 0.0),
                               Point<2>(X_C + R + L_FLAG, Y_C - T / 2.0),
                               Point<2>(X_3, Y_3)});

  GridGenerator::general_cell(tria_vec[8],
                              {Point<2>(X_C + R + L_FLAG, Y_C + T / 2.0),
                               Point<2>(X_3, 2.0 * Y_3),
                               Point<2>(X_C + R + L_FLAG, H),
                               Point<2>(X_3, H)});

  GridGenerator::general_cell(tria_vec[9],
                              {Point<2>(X_C + R + L_FLAG, Y_C - T / 2.0),
                               Point<2>(X_3, Y_3),
                               Point<2>(X_C + R + L_FLAG, Y_C + T / 2.0),
                               Point<2>(X_3, 2.0 * Y_3)});

  GridGenerator::subdivided_hyper_rectangle(tria_vec[10],
                                            {8, 3} /* subdivisions x,y */,
                                            Point<2>(X_3, 0.0),
                                            Point<2>(L, H));

  std::vector<Triangulation<2> const *> tria_vec_ptr(tria_vec.size());
  for (unsigned int i = 0; i < tria_vec.size(); ++i)
    tria_vec_ptr[i] = &tria_vec[i];

  GridGenerator::merge_triangulations(tria_vec_ptr, tria);
}

template <int dim>
unsigned int
n_locally_owned_active_cells_around_point(const Triangulation<dim> &tria,
                                          const Mapping<dim> &      mapping,
                                          const Point<dim> &        point,
                                          const double              tolerance)
{
  using Pair =
    std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>;

  std::vector<Pair> adjacent_cells =
    GridTools::find_all_active_cells_around_point(mapping,
                                                  tria,
                                                  point,
                                                  tolerance);

  // count locally owned active cells
  unsigned int counter = 0;
  for (auto cell : adjacent_cells)
    {
      if (cell.first->is_locally_owned())
        {
          Assert(GeometryInfo<dim>::distance_to_unit_cell(cell.second) < 1e-10,
                 ExcInternalError());

          ++counter;
        }
    }


  return counter;
}


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
    const parallel::distributed::Triangulation<dim, spacedim> &tria,
    const Mapping<dim, spacedim> &                             mapping,
    const double                                               tolerance)
    : comm(tria.get_communicator())
  {
    // create bounding boxed of local active cells
    std::vector<BoundingBox<spacedim>> local_boxes;
    for (const auto cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        local_boxes.push_back(mapping.get_bounding_box(cell));

    // create r-tree of bounding boxes
    const auto local_tree = pack_rtree(local_boxes);

    // compress r-tree to a minimal set of bounding boxes
    const auto local_reduced_box = extract_rtree_level(local_tree, 0);

    // gather bounding boxes of other processes
    const auto global_bounding_boxes =
      Utilities::MPI::all_gather(comm, local_reduced_box);

    // determine ranks which might poses quadrature point
    auto points_per_process =
      std::vector<std::vector<Point<spacedim>>>(global_bounding_boxes.size());

    auto points_per_process_offset =
      std::vector<std::vector<unsigned int>>(global_bounding_boxes.size());

    for (unsigned int i = 0; i < quadrature_points.size(); ++i)
      {
        const auto &point = quadrature_points[i];
        for (unsigned rank = 0; rank < global_bounding_boxes.size(); ++rank)
          for (const auto &box : global_bounding_boxes[rank])
            if (box.point_inside(point))
              {
                points_per_process[rank].emplace_back(point);
                points_per_process_offset[rank].emplace_back(i);
                break;
              }
      }

    // only communicate with processes that might have a quadrature point
    std::vector<unsigned int> targets;

    for (unsigned int i = 0; i < points_per_process.size(); ++i)
      if (points_per_process[i].size() > 0 &&
          i != Utilities::MPI::this_mpi_process(comm))
        targets.emplace_back(i);


    std::map<unsigned int, std::vector<Point<spacedim>>>
      relevant_points_per_process;
    std::map<unsigned int, std::vector<unsigned int>>
      relevant_points_per_process_offset;
    std::map<unsigned int, std::vector<unsigned int>>
      relevant_points_per_process_count;


    // for local quadrature points no communication is needed...
    {
      const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

      const auto &potentially_local_points = points_per_process[my_rank];


      {
        std::vector<Point<spacedim>> points;
        std::vector<unsigned int>    points_offset;
        std::vector<unsigned int>    count;

        const auto &potentially_relevant_points = points_per_process[my_rank];
        const auto &potentially_relevant_points_offset =
          points_per_process_offset[my_rank];

        for (unsigned int j = 0; j < potentially_local_points.size(); ++j)
          {
            const unsigned int counter =
              n_locally_owned_active_cells_around_point(
                tria, mapping, potentially_relevant_points[j], tolerance);

            if (counter > 0)
              {
                points.push_back(potentially_relevant_points[j]);
                points_offset.push_back(potentially_relevant_points_offset[j]);
                count.push_back(counter);
              }
          }


        if (points.size() > 0)
          {
            relevant_remote_points_per_process[my_rank]       = points;
            relevant_remote_points_count_per_process[my_rank] = count;

            relevant_points_per_process[my_rank]        = points;
            relevant_points_per_process_offset[my_rank] = points_offset;
            relevant_points_per_process_count[my_rank]  = count;
            map_recv[my_rank]                           = points;
          }
      }
    }

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

              const unsigned int counter =
                n_locally_owned_active_cells_around_point(tria,
                                                          mapping,
                                                          point,
                                                          tolerance);

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

    Utilities::MPI::ConsensusAlgorithms::Selector<double, unsigned int>(process,
                                                                        comm)
      .run();

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
      output[i].assign(quadrature_points_count[i], T());
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
          temp[j].resize(i.second[j]);

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

    // process locally-owned values
    if (input.find(my_rank) != input.end())
      temp_recv_map[my_rank] = input.at(my_rank);

    for (const auto &vec : map_recv)
      {
        if (vec.first == my_rank)
          continue;

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
  const std::map<unsigned int, std::vector<Point<spacedim>>> &
  get_remote_quadrature_points() const
  {
    return relevant_remote_points_per_process;
  }

  /**
   * Print internal data structures.
   */
  void
  print() const
  {
    std::cout << "Locally owned quadrature points:" << std::endl;

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
  create_triangulation_structure(tria_solid);
  // tria_solid.refine_global(3);

  parallel::distributed::Triangulation<dim, spacedim> tria_fluid(comm);
  create_triangulation_fluid(tria_fluid);
  // tria_fluid.refine_global(4);

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
    surface_quadrature_points, tria_fluid, mapping, 1.e-10);

  // Ic) allocate memory
  std::vector<std::vector<Tensor<1, spacedim>>> surface_values;
  eval.init_surface_values(surface_values);

  std::map<unsigned int, std::vector<std::vector<Tensor<1, spacedim>>>>
    intermediate_values;
  eval.init_intermediate_values(intermediate_values);

  // IIa) fill intermediate values
  const auto &remote_quadrature_point_per_process =
    eval.get_remote_quadrature_points();

  for (auto &p : intermediate_values)
    {
      const unsigned int rank   = p.first;
      auto &             values = p.second;
      const auto &       quadrature_points =
        remote_quadrature_point_per_process.at(rank);

      for (unsigned int i = 0; i < values.size(); ++i)
        for (unsigned int j = 0; j < values[i].size(); ++j)
          values[i][j] = quadrature_points[i]; // TODO: do something useful
    }

  // IIb) communicate values
  eval.process(/*src=*/intermediate_values, /*dst=*/surface_values);

  // IIc) use surface values
  for (unsigned int i = 0; i < surface_values.size(); ++i)
    {
      if (surface_values[i].size() == 0)
        continue;

      std::cout << surface_values[i].size() << " : "
                << surface_quadrature_points[i] << " ";

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