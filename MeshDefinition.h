#ifndef MESHDEFINITION_H
#define MESHDEFINITION_H

#include <OpenMesh/Core/Geometry/VectorT.hh>
#include<OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
//#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include<vector>
struct MeshTraits : public OpenMesh::DefaultTraits
{
	typedef OpenMesh::Vec3d Point;
	typedef OpenMesh::Vec3d Normal;
	typedef OpenMesh::Vec2d TexCoord2D;

	VertexAttributes(OpenMesh::Attributes::Status);
	FaceAttributes(OpenMesh::Attributes::Status);
	EdgeAttributes(OpenMesh::Attributes::Status);
	HalfedgeAttributes(OpenMesh::Attributes::Status);

	//FaceTraits
	//{
	//};
	//EdgeTraits
	//{
	//};
	//HalfedgeTraits
	//{
	//	HalfedgeT() :face_he_var(-1)
	//	{
	//	};
	//private:
	//	int face_he_var;
	//public:
	//	int get_face_he_var() { return face_he_var; };
	//	void set_face_he_var(int fhe) { face_he_var = fhe; };
	//};
	//VertexTraits
	//{
	//	VertexT() : new_pos_fixed(false)
	//	{
	//	};
	//private:
	//	OpenMesh::Vec3d new_pos;//can be used for deformation and parameterization
	//	bool new_pos_fixed;
	//public:
	//	void set_New_Pos(const OpenMesh::Vec3d& n_p){ new_pos = n_p; };
	//	OpenMesh::Vec3d& get_New_Pos(){ return new_pos; };
	//	void set_new_pos_fixed(bool f){ new_pos_fixed = f; };
	//	bool get_new_pos_fixed(){ return new_pos_fixed; };
	//};

};


typedef OpenMesh::TriMesh_ArrayKernelT<MeshTraits> Mesh;

//typedef OpenMesh::PolyMesh_ArrayKernelT<MeshTraits> Mesh;
//typedef OpenMesh::PolyMesh_ArrayKernelT<MyMeshTraits> MyMesh;


bool is_flip_ok_openmesh(Mesh::EdgeHandle& eh, Mesh& mesh_);//just copy the code from openmesh
bool flip_openmesh(Mesh::EdgeHandle& eh, Mesh& mesh_);

bool check_in_triangle_face(const std::vector<OpenMesh::Vec3f>& tri, const OpenMesh::Vec3f& p);


#endif