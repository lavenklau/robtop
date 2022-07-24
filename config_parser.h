#pragma once

#ifndef __CONFIG_PARSER_H
#define __CONFIG_PARSER_H

#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "BoundaryCondition.h"

#include "stdint.h"

#include "gflags/gflags.h"



DECLARE_string(jsonfile);

DECLARE_string(meshfile);

DECLARE_string(outdir);

DECLARE_double(power_penalty);

DECLARE_double(volume_ratio);

DECLARE_double(youngs_modulus);

DECLARE_double(damp_ratio);

DECLARE_double(design_step);

DECLARE_double(vol_reduction);

DECLARE_double(filter_radius);

DECLARE_int32(gridreso);

DECLARE_bool(enable_log);

DECLARE_int32(max_itn);

DECLARE_double(min_density);

DECLARE_double(poisson_ratio);

DECLARE_double(shell_width);

DECLARE_string(workmode);

DECLARE_string(testname);

DECLARE_bool(logdensity);

DECLARE_bool(logcompliance);

DECLARE_string(inputdensity);

DECLARE_string(testmesh);

struct config_parser_t {

	static_range::rangeUnion loadArea;

	static_range::rangeUnion fixArea;

	dynamic_range::Field_t force_field;

	std::function<bool(double[3])> inLoadArea;
	std::function<bool(double[3])> inFixArea;
	std::function<Eigen::Matrix<double, 3, 1>(double[3])> loadField;

	std::vector<float> mesh_vertices;
	std::vector<int> mesh_faces;

	int logflag = 0;

	void parse(int argc, char** argv);

	void parse_benchmark_configuration(const std::string& filename);

	void readMesh(const std::string& meshfile, std::vector<float>& coords, std::vector<int>& trfaces);
};



#endif

