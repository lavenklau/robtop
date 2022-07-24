#pragma once

#ifndef __CGAL_DEFINITION_H
#define __CGAL_DEFINITION_H

#include "CGAL/Surface_mesh/Surface_mesh.h"
#include "CGAL/Simple_cartesian.h"
#include "CGAL/Side_of_triangle_mesh.h"
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include "CGAL/AABB_segment_primitive.h"

#include "CGAL/point_generators_3.h"
#include "CGAL/Orthogonal_k_neighbor_search.h"
#include "CGAL/Search_traits_3.h"
#include "CGAL/Search_traits_adapter.h"
#include "boost/iterator/zip_iterator.hpp"
#include "CGAL/Plane_3.h"

#include "CGAL/Polygon_mesh_processing/corefinement.h"

//#include "CGAL/bary"

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector3;
typedef Kernel::Segment_3 Segment;
typedef CGAL::Surface_mesh<Point> CGMesh;
typedef CGAL::Triangle_3<Kernel> Triangle;
typedef CGAL::Segment_3<Kernel> Segment;
typedef CGAL::AABB_tree< CGAL::AABB_traits<
	Kernel, CGAL::AABB_triangle_primitive<
	Kernel, std::vector<Triangle>::iterator
	>
>
> aabb_tree_t;

typedef boost::tuple<Point, int> PointInt;
typedef CGAL::Search_traits_3<Kernel> KdTreeTraits;
typedef CGAL::Search_traits_adapter<PointInt, CGAL::Nth_of_tuple_property_map<0, PointInt>, KdTreeTraits> Traits;
typedef CGAL::Orthogonal_k_neighbor_search<Traits> KdTreeSearch;
typedef KdTreeSearch::Tree KdTree;

typedef boost::graph_traits<CGMesh>::vertex_descriptor Vertex_descriptor;
typedef boost::graph_traits<CGMesh>::face_descriptor Face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;

extern CGMesh cgmesh_container, cgmesh_object;

extern std::vector<Triangle> aabb_tris;
extern aabb_tree_t aabb_tree;

extern std::vector<Point> kdpoints;
extern std::vector<int> kdpointids;
extern KdTree kdtree;

//CGMesh gmesh;
extern CGMesh::Property_map<Face_descriptor, Vector3> cgmesh_fnormals;
extern CGMesh::Property_map<Vertex_descriptor, Vector3> cgmesh_vnormals;

//void mesh2cgalmesh(Mesh& mesh, CGMesh& cgmesh);
#endif


