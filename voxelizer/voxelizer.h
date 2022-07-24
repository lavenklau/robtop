// voxelizer.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <iostream>
#include <vector>
#include "util.h"
#include "util_io.h"

voxinfo voxelize_mesh(
	const std::vector<float>& vertex_coords, const std::vector<int>& triface_ids,
	int prefered_resolution, std::vector<unsigned int>& solid_bits, int out_resolution[3], float out_box[2][3]
);

void hierarchy_voxelize_mesh(
	const std::vector<float>& vertex_coords, const std::vector<int>& triface_ids, int nlayer,
	int prefered_resolution, std::vector<std::vector<unsigned int>>& solid_bits, std::vector<std::array<int, 3>> out_resolutions, std::vector<std::pair<std::array<float, 3>, std::array<float, 3> >> out_boxs
);


