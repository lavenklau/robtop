#pragma once
#include <OpenMesh/Core/Geometry/VectorT.hh>
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh"
#include "OpenMesh/Core/IO/MeshIO.hh"
#include <string>
#include "glm/glm.hpp"

namespace mesh_utils {

	struct MeshTraits : public OpenMesh::DefaultTraits
	{
		typedef OpenMesh::Vec3d Point;
		typedef OpenMesh::Vec3d Normal;
		typedef OpenMesh::Vec2d TexCoord2D;

		VertexAttributes(OpenMesh::Attributes::Status);
		FaceAttributes(OpenMesh::Attributes::Status);
		EdgeAttributes(OpenMesh::Attributes::Status);
		HalfedgeAttributes(OpenMesh::Attributes::Status);
	};

	struct PolyMeshTraits : public OpenMesh::DefaultTraits
	{
		typedef OpenMesh::Vec3d Point;
		typedef OpenMesh::Vec3d Normal;
	};


	typedef OpenMesh::TriMesh_ArrayKernelT<> Mesh;

	//typedef OpenMesh::PolyMesh_ArrayKernelT<PolyMeshTraits> PolyMesh;


	Mesh ReadMesh(std::string meshfile);

	std::vector<glm::vec4> FlattenVertex(Mesh& m);

	//PolyMesh Mesh pf2Mesh(std::vector<float> p[3], std::vector<int> trif[3], std::vector<int> quadf[4]);
}

