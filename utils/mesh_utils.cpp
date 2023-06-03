#include "mesh_utils.h"

using namespace mesh_utils;

Mesh mesh_utils::ReadMesh(std::string meshfile)
{
	Mesh m;
	bool suc = OpenMesh::IO::read_mesh(m, meshfile);
	if (!suc) {
		throw std::runtime_error("read mesh failed");
	}
	return m;
}

std::vector<glm::vec4> mesh_utils::FlattenVertex(Mesh& m)
{
	std::vector<glm::vec4> v4list;
	for (auto iter = m.vertices_begin(); iter != m.vertices_end(); iter++) {
		auto p = m.point(*iter);
		v4list.emplace_back(p[0], p[1], p[2], 0);
	}
	return v4list;
}

