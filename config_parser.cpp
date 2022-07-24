#include "config_parser.h"
#include "MeshDefinition.h"

DEFINE_string(jsonfile, "", "The boundary config file in json format");

DEFINE_string(meshfile, "", "The optimization model");

DEFINE_string(outdir, "", "The output directory of result");

DEFINE_double(power_penalty, 3, "The power exponet of density");

DEFINE_double(volume_ratio, 0.3, "The target volume ratio ");

DEFINE_double(youngs_modulus, 1e6, "The Young's Modulus of used material");

DEFINE_double(damp_ratio, 0.5, "Optimization parameter damp ratio of Optimal Criteria");

DEFINE_double(design_step, 0.03, "Optimization parameter design step of Optimal Criteria");

DEFINE_double(vol_reduction, 0.02, "Optimization parameter volume reduction of Optimal Criteria");

DEFINE_double(filter_radius, 2, "The filter radius of sensitivity filter");

DEFINE_int32(gridreso, 200, "The grid resolution along alongest axis");

DEFINE_bool(enable_log, false, "Enable log ");

DEFINE_int32(max_itn, 100, "Maximal iteration number");

DEFINE_double(min_density, 1e-3, "Minimal density restriction");

DEFINE_double(poisson_ratio, 0.4, "Material Poisson Ratio");

DEFINE_double(shell_width, 3, "Shell width in the unit of element");

DEFINE_string(workmode, "", "Working mode");

DEFINE_string(testname, "None", "Specifying a test name ");

DEFINE_bool(logdensity, false, "Whether to output current density field");

DEFINE_bool(logcompliance, false, "Whether to log compliance distribution");

DEFINE_string(inputdensity, "", "input density field in VDB format, must much the structure of grids");

DEFINE_string(testmesh, "", "test mesh to show stress or compliance");

dynamic_range::range_t* _parse_region_object(const rapidjson::GenericObject<false, rapidjson::Value>& rg) {
	if (!rg.HasMember("type")) { printf("\033[31mError occurred in json parsing :A region should has member \"type\": \"box\"/\"sphere\"/\"halfspace\"...\033[0m"); exit(0); }
	if (!rg["type"].IsString()) { printf("\033[31mError occurred in json parsing :type should be a string ...\033[0m"); exit(0); }
	std::string typeStr = rg["type"].GetString();
	bool closed = false;
	bool complement = false;
	if (rg.HasMember("closed") && rg["closed"].IsBool()) {
		if (rg["closed"].GetBool()) {
			closed = true;
		}
	}
	if (rg.HasMember("complement") && rg["complement"].IsBool()) {
		if (rg["complement"].GetBool()) { complement = true; }
	}
	do {
		if (typeStr == "box") {
			if (!rg.HasMember("p") || !rg["p"].IsArray()) { printf("\033[31mError occurred in json parsing :box should contains corner \"p\" represented by array [px,py,pz] ...\033[0m"); exit(0); }
			if (!rg.HasMember("q") || !rg["q"].IsArray()) { printf("\033[31mError occurred in json parsing :box should contains corner represented by array \"q\" [qx,qy,qz] ...\033[0m"); exit(0); }
			auto pcoord = rg["p"].GetArray();
			double p[3]{ 0,0,0 }, q[3]{ 0,0,0 };
			for (int i = 0; i < pcoord.Size(); i++) {
				if (i >= 3) { printf("\033[31mError occurred in json parsing :box corner p should be represented by array \"p\" [px,py,pz], no enough member of array were given\033[0m"); exit(0); }
				if (!pcoord[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of box corner  p should be number\033[0m"); exit(0); }
				p[i] = pcoord[i].GetDouble();
			}
			auto qcoord = rg["q"].GetArray();
			for (int i = 0; i < qcoord.Size(); i++) {
				if (i >= 3) { printf("\033[31mError occurred in json parsing :box corner q should be represented by array \"q\" [qx,qy,qz], no enough member of array were given\033[0m"); exit(0); }
				if (!qcoord[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of box corner  q should be number\033[0m"); exit(0); }
				q[i] = qcoord[i].GetDouble();
			}
			return new dynamic_range::boxRange_t(p, q, closed, complement);
		}
		if (typeStr == "sphere") {
			double centerCoord[3]{ 0,0,0 };
			double r;
			if (rg.HasMember("center") && rg["center"].IsArray()) {
				auto center = rg["center"].GetArray();
				for (int i = 0; i < center.Size(); i++) {
					if (i >= 3) { printf("\033[31mError occurred in json parsing :center of sphere should be represented by array \"center\" [cx,cy,cz], no enough member of array were given\033[0m"); exit(0); }
					if (!center[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of sphere center should be number\033[0m"); exit(0); }
					centerCoord[i] = center[i].GetDouble();
				}
			}
			if (rg.HasMember("radius") && rg["radius"].IsNumber()) {
				r = rg["radius"].GetDouble();
			}
			return new dynamic_range::sphereRange_t(centerCoord, r, closed, complement);
		}
		if (typeStr == "halfspace") {
			double p[3][3];
			if (rg.HasMember("p1") && rg["p1"].IsArray()) {
				auto p1 = rg["p1"].GetArray();
				for (int i = 0; i < p1.Size(); i++) {
					if (i >= 3) { printf("\033[31mError occurred in json parsing :anchor point p1 of plane should be represented by array \"p1\" [p1x,p1y,p1z], no enough member of array were given\033[0m"); exit(0); }
					if (!p1[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of anchor point should be number\033[0m"); exit(0); }
					p[0][i] = p1[i].GetDouble();
				}
			}
			if (rg.HasMember("p2") && rg["p2"].IsArray()) {
				auto p2 = rg["p2"].GetArray();
				for (int i = 0; i < p2.Size(); i++) {
					if (i >= 3) { printf("\033[31mError occurred in json parsing :anchor point p2 of plane should be represented by array \"p2\" [p2x,p2y,p2z], no enough member of array were given\033[0m"); exit(0); }
					if (!p2[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of anchor point should be number\033[0m"); exit(0); }
					p[1][i] = p2[i].GetDouble();
				}
			}
			if (rg.HasMember("p3") && rg["p3"].IsArray()) {
				auto p3 = rg["p3"].GetArray();
				for (int i = 0; i < p3.Size(); i++) {
					if (i >= 3) { printf("\033[31mError occurred in json parsing :anchor point p2 of plane should be represented by array \"p2\" [p2x,p2y,p2z], no enough member of array were given\033[0m"); exit(0); }
					if (!p3[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of anchor point should be number\033[0m"); exit(0); }
					p[2][i] = p3[i].GetDouble();
				}
			}
			return new dynamic_range::halfSpaceRange_t(p[0], p[1], p[2], closed, complement);
			break;
		}
		if (typeStr == "cylinder") {
			Eigen::Matrix<double, 3, 1> cbot;
			Eigen::Matrix<double, 3, 1> vh;
			double radi;
			radi = rg["radius"].GetDouble();
			for (int i = 0; i < 3; i++) {
				cbot[i] = rg["bottomCenter"].GetArray()[i].GetDouble();
				vh[i] = rg["heightVector"].GetArray()[i].GetDouble();
			}
			return new dynamic_range::cylinderRange_t(cbot, radi, vh);
			break;
		}
	} while (0);
}

void _parse_one_region_to_union(rapidjson::GenericObject<false, rapidjson::Value>& rg, static_range::rangeUnion& rgAccum) {
	// parse type 
	if (!rg.HasMember("type")) { printf("\033[31mError occurred in json parsing :A region should has member \"type\": \"box\"/\"sphere\"/\"halfspace\"...\033[0m"); exit(0); }
	if (!rg["type"].IsString()) { printf("\033[31mError occurred in json parsing :type should be a string ...\033[0m"); exit(0); }
	std::string typeStr = rg["type"].GetString();
	bool closed = false;
	bool complement = false;
	if (rg.HasMember("closed") && rg["closed"].IsBool()) {
		if (rg["closed"].GetBool()) {
			closed = true;
		}
	}
	if (rg.HasMember("complement") && rg["complement"].IsBool()) {
		if (rg["complement"].GetBool()) { complement = true; }
	}
	do {
		if (typeStr == "box") {
			if (!rg.HasMember("p") || !rg["p"].IsArray()) { printf("\033[31mError occurred in json parsing :box should contains corner \"p\" represented by array [px,py,pz] ...\033[0m"); exit(0); }
			if (!rg.HasMember("q") || !rg["q"].IsArray()) { printf("\033[31mError occurred in json parsing :box should contains corner represented by array \"q\" [qx,qy,qz] ...\033[0m"); exit(0); }
			auto pcoord = rg["p"].GetArray();
			double p[3]{ 0,0,0 }, q[3]{ 0,0,0 };
			for (int i = 0; i < pcoord.Size(); i++) {
				if (i >= 3) { printf("\033[31mError occurred in json parsing :box corner p should be represented by array \"p\" [px,py,pz], no enough member of array were given\033[0m"); exit(0); }
				if (!pcoord[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of box corner  p should be number\033[0m"); exit(0); }
				p[i] = pcoord[i].GetDouble();
			}
			auto qcoord = rg["q"].GetArray();
			for (int i = 0; i < qcoord.Size(); i++) {
				if (i >= 3) { printf("\033[31mError occurred in json parsing :box corner q should be represented by array \"q\" [qx,qy,qz], no enough member of array were given\033[0m"); exit(0); }
				if (!qcoord[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of box corner  q should be number\033[0m"); exit(0); }
				q[i] = qcoord[i].GetDouble();
			}
			if (closed) {
				if (complement) { rgAccum.add_range(static_range::complementRange_t<static_range::boxRange_t<true>>(p, q)); }
				else { rgAccum.add_range(static_range::boxRange_t<true>(p, q)); }
			}
			else {
				if (complement) { rgAccum.add_range(static_range::complementRange_t<static_range::boxRange_t<false>>(p, q)); }
				else { rgAccum.add_range(static_range::boxRange_t<false>(p, q)); }
			}
		}
		if (typeStr == "sphere") {
			double centerCoord[3]{ 0,0,0 };
			double r;
			if (rg.HasMember("center") && rg["center"].IsArray()) {
				auto center = rg["center"].GetArray();
				for (int i = 0; i < center.Size(); i++) {
					if (i >= 3) { printf("\033[31mError occurred in json parsing :center of sphere should be represented by array \"center\" [cx,cy,cz], no enough member of array were given\033[0m"); exit(0); }
					if (!center[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of sphere center should be number\033[0m"); exit(0); }
					centerCoord[i] = center[i].GetDouble();
				}
			}
			if (rg.HasMember("radius") && rg["radius"].IsNumber()) {
				r = rg["radius"].GetDouble();
			}
			if (closed) {
				if (complement) { rgAccum.add_range(static_range::complementRange_t<static_range::sphereRange_t<true>>(centerCoord, r)); }
				else { rgAccum.add_range(static_range::sphereRange_t<true>(centerCoord, r)); }
			}
			else {
				if (complement) { rgAccum.add_range(static_range::complementRange_t<static_range::sphereRange_t<false>>(centerCoord, r)); }
				else { rgAccum.add_range(static_range::sphereRange_t<false>(centerCoord, r)); }
			}
		}
		if (typeStr == "halfspace") {
			double p[3][3];
			if (rg.HasMember("p1") && rg["p1"].IsArray()) {
				auto p1 = rg["p1"].GetArray();
				for (int i = 0; i < p1.Size(); i++) {
					if (i >= 3) { printf("\033[31mError occurred in json parsing :anchor point p1 of plane should be represented by array \"p1\" [p1x,p1y,p1z], no enough member of array were given\033[0m"); exit(0); }
					if (!p1[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of anchor point should be number\033[0m"); exit(0); }
					p[0][i] = p1[i].GetDouble();
				}
			}
			if (rg.HasMember("p2") && rg["p2"].IsArray()) {
				auto p2 = rg["p2"].GetArray();
				for (int i = 0; i < p2.Size(); i++) {
					if (i >= 3) { printf("\033[31mError occurred in json parsing :anchor point p2 of plane should be represented by array \"p2\" [p2x,p2y,p2z], no enough member of array were given\033[0m"); exit(0); }
					if (!p2[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of anchor point should be number\033[0m"); exit(0); }
					p[1][i] = p2[i].GetDouble();
				}
			}
			if (rg.HasMember("p3") && rg["p3"].IsArray()) {
				auto p3 = rg["p3"].GetArray();
				for (int i = 0; i < p3.Size(); i++) {
					if (i >= 3) { printf("\033[31mError occurred in json parsing :anchor point p2 of plane should be represented by array \"p2\" [p2x,p2y,p2z], no enough member of array were given\033[0m"); exit(0); }
					if (!p3[i].IsNumber()) { printf("\033[31mError occurred in json parsing : coordinate of anchor point should be number\033[0m"); exit(0); }
					p[2][i] = p3[i].GetDouble();
				}
			}
			if (closed) {
				if (complement) { rgAccum.add_range(static_range::complementRange_t<static_range::halfSpaceRange_t<true>>(p[0], p[1], p[2])); }
				else { rgAccum.add_range(static_range::halfSpaceRange_t<true>(p[0], p[1], p[2])); }
			}
			else {
				if (complement) { rgAccum.add_range(static_range::complementRange_t<static_range::halfSpaceRange_t<false>>(p[0], p[1], p[2])); }
				else { rgAccum.add_range(static_range::halfSpaceRange_t<false>(p[0], p[1], p[2])); }
			}
			break;
		}
		if (typeStr == "cylinder") {
			double radi;
			radi = rg["radius"].GetDouble();
			Eigen::Matrix<double, 3, 1> botCenter, heighVector;
			for (int i = 0; i < 3; i++) {
				botCenter[i] = rg["bottomCenter"].GetArray()[i].GetDouble();
				heighVector[i] = rg["heightVector"].GetArray()[i].GetDouble();
			}
			rgAccum.add_range(static_range::cylinderRange<false>(botCenter, radi, heighVector));
			break;
		}
	} while (0);
}

void _parse_one_force_field(rapidjson::GenericObject<false, rapidjson::Value>& rg, dynamic_range::Field_t& fields) {
	// parse region
	if (!rg.HasMember("region") || !rg["region"].IsObject()) { printf("\033[31mforce field should have a restricted act region\033[0m"); exit(0); }
	dynamic_range::range_t* force_region = _parse_region_object(rg["region"].GetObject());


	// parse direction if it has this term
	Eigen::Matrix<double, 3, 1> termDirection;
	if (rg.HasMember("direction") && rg["direction"].IsArray()) {
		auto fArray = rg["direction"].GetArray();
		for (int i = 0; i < fArray.Size(); i++) {
			if (i >= 3) { printf("\033[31mtwo many component of direction is given! direction should be a 3-elements array!\033[0m\n"); exit(0); }
			if (!fArray[i].IsNumber()) { printf("\033[31mdirection should be a 3 element number array\033[0m"); exit(0); }
			termDirection[i] = fArray[i].GetDouble();
		}
	}

	// parse direction if it has this term
	Eigen::Matrix<double, 3, 1> termCenter;
	if (rg.HasMember("center") && rg["center"].IsArray()) {
		auto fArray = rg["center"].GetArray();
		for (int i = 0; i < fArray.Size(); i++) {
			if (i >= 3) { printf("\033[31mtwo many component of direction is given! direction should be a 3-elements array!\033[0m\n"); exit(0); }
			if (!fArray[i].IsNumber()) { printf("\033[31mdirection should be a 3 element number array\033[0m"); exit(0); }
			termCenter[i] = fArray[i].GetDouble();
		}
	}

	// parse field
	if (!rg.HasMember("type") || !rg["type"].IsString()) { printf("\033[31mError occurred in json parsing :A force field should has member \"type\": \"const\"/\"radial\"/\"gradient\"...\033[0m"); exit(0); }
	std::string typeStr = rg["type"].GetString();
	if (typeStr == "const") {
		if (!rg.HasMember("norm") || !rg["norm"].IsNumber()) { printf("\033[31mthe norm of const force field should be given \033[0m"); }
		double nf = rg["norm"].GetDouble();
		if (!rg.HasMember("direction") || !rg["direction"].IsArray()) { printf("\033[31mthe direction of const force field should be given as a number array [fx,fy,fz]\033[0m"); exit(0); }
		dynamic_range::vectorFieldBase_t* F = new dynamic_range::constant_vectorField_t(dynamic_range::constant_normGen_t(nf), dynamic_range::constant_dirGen_t(termDirection));
		fields.add_field(force_region, F);
	}
	else if (typeStr == "RadialNormConstDir") {
		if (!rg.HasMember("center") || !rg["center"].IsArray()) { printf("\033[31mradial force field should be given a center\033[0m"); exit(0); }
		double damp_ratio = 1;
		double amplifier = 1;
		if (rg.HasMember("dampRatio") && rg["dampRatio"].IsNumber()) { damp_ratio = rg["dampRatio"].GetDouble(); }
		if (rg.HasMember("amplifier") && rg["amplifier"].IsNumber()) { amplifier = rg["amplifier"].GetDouble(); }
		dynamic_range::vectorFieldBase_t* F = new dynamic_range::radialNormConstDirField_t(
			dynamic_range::radial_normGen_t(termCenter, amplifier, damp_ratio),
			dynamic_range::constant_dirGen_t(termDirection)
		);
		fields.add_field(force_region, F);
	}
	else if (typeStr == "ConstNormRadialDir") {
		if (!rg.HasMember("center") || !rg["center"].IsArray()) { printf("\033[31mradial force field should be given a center\033[0m"); exit(0); }
		if (!rg.HasMember("norm") || !rg["norm"].IsNumber()) { printf("\033[31mConstNormRadialDir force field should be given a norm\033[0m"); exit(0); }
		double nr = rg["norm"].GetDouble();
		dynamic_range::vectorFieldBase_t* F = new dynamic_range::constNormRadialDirField_t(
			dynamic_range::constant_normGen_t(nr),
			dynamic_range::radial_dirGen_t(termCenter)
		);
		fields.add_field(force_region, F);
	}
	else if (typeStr == "RadialNormRadialDir") {
		if (!rg.HasMember("NormCenter") || !rg["NormCenter"].IsArray()) { printf("\033[31mRadialNormRadialDir force field should be given a center for norm\033[0m"); exit(0); }
		if (!rg.HasMember("DirCenter") || !rg["DirCenter"].IsArray()) { printf("\033[31mRadialNormRadialDir force field should be given a center for direction\033[0m"); exit(0); }
		Eigen::Matrix<double, 3, 1> nc, dc;
		auto ncArray = rg["NormCenter"].GetArray();
		for (int i = 0; i < ncArray.Size(); i++) {
			if (i >= 3) { printf("\033[31mtoo many component for NormCenter!\033[0m"); exit(0); }
			if (!ncArray[i].IsNumber()) { printf("\033[31mcomponent of norm center should be number !\033[0m"); exit(0); }
			nc[i] = ncArray[i].GetDouble();
		}
		auto dcArray = rg["DirCenter"].GetArray();
		for (int i = 0; i < dcArray.Size(); i++) {
			if (i >= 3) { printf("\033[31mtoo many component for DirCenter center!\033[0m"); exit(0); }
			if (!dcArray[i].IsNumber()) { printf("\033[31mcomponent of direction center should be number !\033[0m"); exit(0); }
			dc[i] = dcArray[i].GetDouble();
		}
		double damp_ratio = 1;
		double amplifier = 1;
		if (rg.HasMember("dampRatio") && rg["dampRatio"].IsNumber()) { damp_ratio = rg["dampRatio"].GetDouble(); }
		if (rg.HasMember("amplifier") && rg["amplifier"].IsNumber()) { amplifier = rg["amplifier"].GetDouble(); }
		dynamic_range::vectorFieldBase_t* F = new dynamic_range::rnrdField_t(
			dynamic_range::radial_normGen_t(nc, amplifier, damp_ratio),
			dynamic_range::radial_dirGen_t(dc)
		);
		fields.add_field(force_region, F);
	}
	else if (typeStr == "Twist") {
		try {
			Eigen::Matrix<double, 3, 1> axis[2];
			auto aaArray = rg["AxisStart"].GetArray();
			axis[0][0] = aaArray[0].GetDouble();
			axis[0][1] = aaArray[1].GetDouble();
			axis[0][2] = aaArray[2].GetDouble();
			auto abArray = rg["AxisEnd"].GetArray();
			axis[1][0] = abArray[0].GetDouble();
			axis[1][1] = abArray[1].GetDouble();
			axis[1][2] = abArray[2].GetDouble();
			std::cout << "Axis " << "(" << axis[0][0] << ", " << axis[0][1] << ", " << axis[0][2] << " ; " << "(" << axis[1][0] << ", " << axis[1][1] << ", " << axis[1][2] << ")" << std::endl;
			auto anorm = rg["amplifier"].GetDouble();
			dynamic_range::vectorFieldBase_t* F = new dynamic_range::tntdField_t(
				dynamic_range::twist_normGen_t(axis[0], axis[1], anorm),
				dynamic_range::twist_dirGen_t(axis[0], axis[1])
			);
			fields.add_field(force_region, F);
		}
		catch (...){
			printf("\033[31mParse Json failed, Line %d, File %s\033[0m\n", __LINE__, __FILE__);
		}
	}
	else if (typeStr == "gradient") {

	}
}


void config_parser_t::parse_benchmark_configuration(const std::string& filename) {
	std::cout << "\033[32m" << "parsing json " << filename << "..." << "\033[0m" << std::endl;
	//rapidjson::Document d;
	//d.par
	std::ifstream jsonConfig(filename);
	if (!jsonConfig) {
		printf("\033[31mCannot open file %s\033[0m", filename.c_str());
		exit(0);
	}
	std::string fileContent((std::istreambuf_iterator<char>(jsonConfig)), std::istreambuf_iterator<char>());;
	rapidjson::Document doc;
	doc.Parse(fileContent.c_str());

	auto err = doc.GetParseError();
	if (err) {
		std::cout << "parse error occurred ! error code " << err << std::endl;
	}

	const char* term;

	// deal with boundary condition
	term = "loadarea";
	if (doc.HasMember(term)) {
		if (!doc[term].IsArray()) { printf("\033[31mError occurred in json parsing : loadarea should be a set of region desc [{xxx},{xxx}...] !\033[0m"); exit(0); }
		auto loadRegionDescList = doc[term].GetArray();
		std::cout << "[Json] adding load region..." << std::endl;
		for (auto itr = loadRegionDescList.Begin(); itr != loadRegionDescList.End(); itr++) {
			if (!itr->IsObject()) { printf("\033[31mError occurred in json parsing : an area should be a object {xxx} !\033[0m"); exit(0); }
			auto region = itr->GetObject();
			_parse_one_region_to_union(region, loadArea);
		}
	}

	// deal with force field
	term = "force";
	if (doc.HasMember(term)) {
		if (!doc[term].IsArray()) { printf("\033[31mError occurred in json parsing : externel force field should be a set of single field desc [{xxx},{xxx}...] !\033[0m"); exit(0); }
		auto fArray = doc[term].GetArray();
		std::cout << "[Json] adding force field...  ";
		int counter = 0;
		for (auto itr = fArray.Begin(); itr != fArray.End(); itr++) {
			if (!itr->IsObject()) { printf("\033[31mError occurred in json parsing : an area should be a object {xxx} !\033[0m"); exit(0); }
			auto field = itr->GetObject();
			_parse_one_force_field(field, force_field);
			counter++;
		}
		std::cout << counter << " field" << std::endl;
	}

	// deal with dirichlet boundary
	term = "fixedarea";
	if (doc.HasMember(term)) {
		if (!doc[term].IsArray()) { printf("\033[31mError occurred in json parsing : fixed area should be a set of single region [{xxx},{xxx}...] !\033[0m"); exit(0); }
		auto fixArray = doc[term].GetArray();
		std::cout << "[Json] adding fixed area...  ";
		int counter = 0;
		for (auto itr = fixArray.Begin(); itr != fixArray.End(); itr++) {
			if (!itr->IsObject()) { printf("\033[31mError occurred in json parsing : region format wrong, it should be a object !\033[0m"); exit(0); }
			auto fixarea = itr->GetObject();
			_parse_one_region_to_union(fixarea, fixArea);
			counter++;
		}
		std::cout << counter << " area" << std::endl;
	}
}

void config_parser_t::readMesh(const std::string& meshfile, std::vector<float>& coords, std::vector<int>& trfaces)
{
	Mesh m;
	OpenMesh::IO::read_mesh(m, meshfile);

	coords.resize(m.n_vertices() * 3);
	trfaces.resize(m.n_faces() * 3);

	int counter = 0;
	for (auto iter = m.vertices_begin(); iter != m.vertices_end(); iter++) {
		auto p = m.point(*iter);
		coords[counter++] = p[0];
		coords[counter++] = p[1];
		coords[counter++] = p[2];
	}

	counter = 0;
	for (auto iter = m.faces_begin(); iter != m.faces_end(); iter++) {
		for (auto fv = m.fv_begin(*iter); fv != m.fv_end(*iter); fv++) {
			trfaces[counter++] = fv->idx();
		}
	}
	return;
}

void output_option(void) {
	std::cout << "[Config]:" << std::endl;
	std::cout << " =jsonfile         - - - - - - - - - - - - - - - -   " << FLAGS_jsonfile << std::endl;
	std::cout << " =meshfile         - - - - - - - - - - - - - - - -   " << FLAGS_meshfile << std::endl;
	std::cout << " =outdir           - - - - - - - - - - - - - - - -   " << FLAGS_outdir << std::endl;
	std::cout << " =power_penalty    - - - - - - - - - - - - - - - -   " << FLAGS_power_penalty << std::endl;
	std::cout << " =volume_ratio     - - - - - - - - - - - - - - - -   " << FLAGS_volume_ratio << std::endl;
	std::cout << " =youngs_modulus   - - - - - - - - - - - - - - - -   " << FLAGS_youngs_modulus << std::endl;
	std::cout << " =damp_ratio       - - - - - - - - - - - - - - - -   " << FLAGS_damp_ratio << std::endl;
	std::cout << " =design_step      - - - - - - - - - - - - - - - -   " << FLAGS_design_step << std::endl;
	std::cout << " =vol_reduction    - - - - - - - - - - - - - - - -   " << FLAGS_vol_reduction << std::endl;
	std::cout << " =filter_radius    - - - - - - - - - - - - - - - -   " << FLAGS_filter_radius << std::endl;
	std::cout << " =gridreso         - - - - - - - - - - - - - - - -   " << FLAGS_gridreso << std::endl;;
	std::cout << " =enable_log       - - - - - - - - - - - - - - - -   " << (FLAGS_enable_log ? "Yes" : "No") << std::endl;
	std::cout << " =max_itn          - - - - - - - - - - - - - - - -   " << FLAGS_max_itn << std::endl;
	std::cout << " =min_density      - - - - - - - - - - - - - - - -   " << FLAGS_min_density << std::endl;;
	std::cout << " =youngs_module    - - - - - - - - - - - - - - - -   " << FLAGS_youngs_modulus << std::endl;;
	std::cout << " =poisson_ratio    - - - - - - - - - - - - - - - -   " << FLAGS_poisson_ratio << std::endl;;
	std::cout << " =shell_width      - - - - - - - - - - - - - - - -   " << FLAGS_shell_width << std::endl;
	std::cout << " =work_mode        - - - - - - - - - - - - - - - -   " << FLAGS_workmode << std::endl;
	std::cout << " =logdensity       - - - - - - - - - - - - - - - -   " << (FLAGS_logdensity ? "Yes" : "No") << std::endl;
	std::cout << " =logcompliance    - - - - - - - - - - - - - - - -   " << (FLAGS_logcompliance ? "Yes" : "No") << std::endl;
	std::cout << " =testname         - - - - - - - - - - - - - - - -   " << FLAGS_testname << std::endl;
	std::cout << " =testmesh         - - - - - - - - - - - - - - - -   " << FLAGS_testmesh << std::endl;
}


void config_parser_t::parse(int argc, char** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, false);
	output_option();
	parse_benchmark_configuration(FLAGS_jsonfile);
	inLoadArea = loadArea.generate();
	inFixArea = fixArea.generate();
	loadField = force_field.generate();
	readMesh(FLAGS_meshfile, mesh_vertices, mesh_faces);
}

