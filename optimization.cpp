#include "optimization.h"
#include "projection.h"
#include "matlab_utils.h"
#include "binaryIO.h"
#include "tictoc.h"


gpu_manager_t gpu_manager;

grid::HierarchyGrid grids(gpu_manager);

Parameter params;

void buildGrids(const std::vector<float>& coords, const std::vector<int>& trifaces) 
{
	grids.set_prefer_reso(params.gridreso);
	grids.set_skip_layer(true);
	grids.genFromMesh(coords, trifaces);
}

void logParams(std::string file, std::string version_str, int argc, char** argv)
{
	std::ofstream ofs(grids.getPath(file));
	ofs << "[version] " << version_str << std::endl;
	for (int i = 0; i < argc; i++) {
		ofs << argv[i] << " ";
	}
	ofs << std::endl;
	ofs.close();
}

void setParameters(
	float volRatio, float volDecrease, float designStep, float filterRadi, float dampRatio, float powerPenal,
	float min_density, int gridreso, float youngs_modulu, float poisson_ratio, float shell_width,
	bool logdensity, bool logcompliance
) {
	params.volume_ratio = volRatio;
	params.volume_decrease = volDecrease;
	params.design_step = designStep;
	params.filter_radius = filterRadi;
	params.damp_ratio = dampRatio;
	params.power_penalty = powerPenal;
	params.min_rho = min_density;
	params.gridreso = gridreso;
	params.youngs_modulu = youngs_modulu;
	params.poisson_ratio = poisson_ratio;
	grids.set_shell_width(shell_width);
	grids.enable_logdensity(logdensity);
	grids.enable_logcompliance(logcompliance);
}

void setOutpurDir(const std::string& dirname)
{
	std::string outdir = dirname;

	char lastChar = *outdir.rbegin();
	if (lastChar != '\\' && lastChar != '/') {
		outdir.push_back('\\');
	}
	std::cout << "\033[32m-- Output path \033[0m \n  " << outdir << "\033[0m" << std::endl;

	grids.setOutPath(outdir);
}

void setWorkMode(const std::string& modestr)
{
	if (modestr == "nscf") {
		grids.setMode(no_support_constrain_force_direction);
	}
	else if (modestr == "nsff") {
		grids.setMode(no_support_free_force);
	}
	else if (modestr == "wscf") {
		grids.setMode(with_support_constrain_force_direction);
	}
	else if (modestr == "wsff") {
		grids.setMode(with_support_free_force);
	}
	else {
		printf("-- unsupported mode\n");
		exit(-1);
	}
}

void solveFEM(void)
{
	double rel_res = 1;
	while (rel_res > 1e-4) {
		rel_res = grids.v_cycle();
	}
}


void matlab_utils_test(void) {
#ifdef ENABLE_MATLAB
	std::vector<Eigen::Triplet<double>>  triplist;
	for (int i = 0; i < 300; i++) {
		triplist.emplace_back(i * 2, i * 10, i);
	}

	Eigen::SparseMatrix<double> sp;

	sp.resize(4000, 4000);

	sp.setFromTriplets(triplist.begin(), triplist.end());

	eigen2ConnectedMatlab("sp", sp);

	getMatEngine().eval("a=nnz(sp);");

	Eigen::Matrix<double, -1, -1> spnnz;
	matlab2eigen("a", spnnz);

	if (spnnz(0, 0) > 90) {
		printf("-- matlab utils test passed\n");
	}
	else {
		printf("-- matlab utils test failed\n");
	}
#endif
}

double modifiedPM(void)
{
	// DEBUG
	//test_rigid_displacement();
	//exit(-1);

#if 1
	// generate random force
	grids[0]->randForce();
#else
	// pertubate force on last force
	grids[0]->pertubForce(0.8);
#endif

	// project force to balanced load on load region
	forceProject(grids[0]->getForce());
	
	// normalize force
	grids[0]->unitizeForce();

	// reset displacement
	grids[0]->reset_displacement();

	// DEBUG
	grids[0]->force2matlab("finit");

	getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

	double fch = 1;

	int max_itn = 500;

	int itn = 0;

	double rel_res = 1;

	printf("\033[32m[ModiPM]\n\033[0m");

	//grids.test_vcycle();

	snippet::SerialChar<double> fchserial;

	bool failed = false;

	// 1e-5
	while (itn++ < max_itn && (fch > 1e-4 || rel_res > 1e-2)) {
#if 1
		// do one v_cycle
		rel_res = grids.v_cycle(1, 1);
#else
		if (fchserial.arising() && itn > 30) {
			rel_res = grids.v_halfcycle(1, 1, 1);
		}
		else {
			rel_res = grids.v_cycle(1, 1);
		}
#endif

		// if fch is arising, init itn
		//if (itn > 30 && fchserial.arising()) {
		//	itn -= 30;
		//	// clear arising serial
		//	fchserial.add(-std::numeric_limits<double>::infinity());
		//	printf("-- fch arising, itn ->%d\n", itn);
		//}

		// remove rigid displacement
		//grids[0]->displacement2matlab("u1");
#if 0
		if (!grids.hasSupport() && rel_res > 2) displacementProject(grids[0]->getDisplacement());
#else
		if (!grids.hasSupport()) displacementProject(grids[0]->getDisplacement());
#endif
		//grids[0]->displacement2matlab("u2");

		failed = rel_res > 1e4;

		if (failed) break;

		// project to balanced load on load region
		grids[0]->v3_copy(grids[0]->getDisplacement(), grids[0]->getForce());
		forceProject(grids[0]->getForce());

		// normalize projected force
		grids[0]->unitizeForce();

		// compute change of force
		fch = grids[0]->supportForceCh() / grids[0]->supportForceNorm();

		//grids[0]->force2matlab("f1");

		// check fch serial
		fchserial.add(fch);

		// update support force
		getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

		// output residual information
		//printf("-- r_rel %6.2lf%%, fch %2.2lf%%  %s\n", rel_res * 100, fch * 100, fchserial.arising() ? "( + )" : "");
		printf("--[%d] r_rel %6.2lf%%, fch %2.2lf%%  \n", itn, rel_res * 100, fch * 100);

		// DEBUG
		//if (itn % 20 == 0) {
		//	grids.writeSupportForce(grids.getPath("fs"));
		//}
	}

	if (failed) {
		printf("-- modified PM failed\n");
		grids.writeSupportForce(grids.getPath("fserr"));
		grids.writeDisplacement(grids.getPath("uerr"));
		exit(-1);
	}

	// reduce relative residual 
	itn = 0;
#if 0
	while (rel_res > 1e-5 && itn++ < 50) { rel_res = grids.v_cycle(); printf("-- r_rel %6.2lf%%\n", rel_res * 100); }
#endif

	// DEBUG
	grids[0]->force2matlab("fworst");
	grids[0]->displacement2matlab("uworst");
	//grids.writeSupportForce(grids.getPath("fs"));

	//grids[0]->v3_copy(grids[0]->getForce(), grids[0]->getWorstForce());

	//grids[0]->v3_copy(grids[0]->getDisplacement(), grids[0]->getWorstDisplacement());

	double worstCompliance = grids[0]->compliance();

	if (isnan(worstCompliance)) { printf("\033[31m-- NaN occurred !\033[0m\n"); exit(-1); }

	grids[0]->_keyvalues["mu"] = worstCompliance;

	printf("-- Worst Compliance %6.3e\n", worstCompliance);

	return worstCompliance;
}

#if 0
double eigenCG(void)
{
	// f ( u ) = u ^T P u / u ^ T K u;
	double * p[3];
	double * g[3];
	double * g1[3];
	double * xa[3];
	double * ga[3];
	grids[0]->v3_create(p);
	grids[0]->v3_create(g);
	grids[0]->v3_create(g1);
	grids[0]->v3_create(xa);
	grids[0]->v3_create(ga);

	double** x = grids[0]->getDisplacement();

	// generate random x0
	grids[0]->v3_rand(x, -1, 1);

	auto pknorm = RayleighGradient(x, g);
	double pnorm = pknorm.first;
	double knorm = pknorm.second;
	
	grids[0]->v3_copy(g, p);

	int itn = 0;
	double res = 1;
	double gk2 = grids[0]->v3_dot(g, g);
	double newgk2 = gk2;

	while (res > 1e-6 & itn++ < 200) {

		double al = 0, au = 10;
		double a = 1;
		// line search
		auto h = [&](double t) {
			grids[0]->v3_copy(x, xa);
			grids[0]->v3_add(xa, t, p);
			double pn = Pnorm(xa);
			double kn = Knorm(xa);
			return pn / kn;
		};
		auto dh = [&](double t, double* gnew[3], double& pn, double& kn) {
			grids[0]->v3_copy(x, xa);
			grids[0]->v3_add(xa, t, p);
			auto pk = RayleighGradient(xa, gnew);
			pn = pk.first; kn = pk.second;
			return grids[0]->v3_dot(gnew, p);
		};

		double amax = 100;
		double c1 = 1e-1, c2 = 1 / 3;
		snippet::circle_array_t<double, 20> alist, hlist;
		alist[0] = 0; alist[1] = 1;
		double pn0, kn0;
		double dh0 = dh(alist[0], ga, pn0, kn0);
		double h0 = pn0 / kn0;
		hlist[0] = h0;
		for (int itn = 1; itn < 100; itn++) {
			double pn, kn;
			double dha = dh(alist[itn], ga, pn, kn);
			double ha = pn / kn;
			hlist[itn] = ha;

			auto zoom = [&](double aL, double aU) {
				double a;
				for (int itnn = 0; itnn < 100; itnn++) {
					// interpolate  a
					a = (aL + aU) / 2;

					// narrow interval
					double pn, kn;
					double dha = dh(a, ga, pn, kn);
					double ha = pn / kn;
					double hal = h(aL);
					if (ha > h0 + c1 * a * dh0 || ha >= hal) {
						aU = a;
					}
					else {
						if (abs(dha) <= -c2 * dh0) {
							return a;
						}
						if (dha*(aU - aL) > 0) {
							aU = aL;
						}
						aL = a;
					}
				}
				return a;
			};

			if (ha > h0 + c1 * alist[itn] * dh0 || (hlist[itn] >= hlist[itn - 1] && itn > 1)) {
				a = zoom(alist[itn - 1], alist[itn]);
				break;
			}
			if (abs(dha) <= -c2 * dh0) {
				a = alist[itn];
				break;
			}
			if (dha > 0) {
				a = zoom(alist[itn], alist[itn - 1]);
			}
			alist[itn + 1] = (a + amax) / 2;
		}
		
		
		// update x
		grids[0]->v3_add(x, a, p);

		// update direction
		auto pknorm = RayleighGradient(x, g1);
		newgk2 = grids[0]->v3_dot(g1, g1);
		grids[0]->v3_minus(g1, 1, g);
		double gk12 = grids[0]->v3_dot(g1, g);
		//newgk2 = grids[0]->v3_dot(g, g);
		double beta = gk12 / gk2;
		gk2 = newgk2;
	}
}
#endif

std::pair<double, double> RayleighGradient(double* u[3], double* g[3])
{
	grids[0]->v3_copy(u, g);
	forceProject(g);
	double pnorm = grids[0]->v3_dot(g, g);
	grids[0]->applyK(u, grids[0]->getForce());
	double knorm = grids[0]->v3_dot(u, grids[0]->getForce());
#if 0
	grids[0]->v3_add(2.0 / knorm, g, -2.0 * pnorm / (knorm*knorm), grids[0]->getForce());
#else
	grids[0]->v3_add(-2.0 / knorm, g, +2.0 * pnorm / (knorm*knorm), grids[0]->getForce());
#endif
	return { pnorm,knorm };
}

double Pnorm(double* u[3])
{
	getForceSupport(u, grids[0]->getSupportForce());
	std::vector<double> fs[3];
	for (int i = 0; i < 3; i++) {
		gpu_manager_t::download_buf(fs[i].data(), grids[0]->getSupportForce()[i], sizeof(double)*n_loadnodes());
	}

	double* fsp[3] = { fs[0].data(),fs[1].data(),fs[2].data() };
	forceProject(fsp);

	double s = 0;
#pragma omp parallel for reduction(+:s)
	for (int i = 0; i < n_loadnodes(); i++) {
		s += pow(fsp[0][i], 2) + pow(fsp[1][i], 2) + pow(fsp[2][i], 2);
	}

	return s;
}

double Knorm(double* u[3])
{
	grids[0]->applyK(u, grids[0]->getForce());
	double s = grids[0]->v3_dot(u, grids[0]->getForce());
	return s;
}

double inCompleteModiPM(double fch_thres /*= 1e-2*/)
{
	// generate random force
	grids[0]->randForce();
	// project force to balanced load on load region
	forceProject(grids[0]->getForce());
	// normalize force
	grids[0]->unitizeForce();
	// reset displacement
	grids[0]->reset_displacement();
	getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

	double fch = 1;
	int max_itn = 50;
	int itn = 0;
	double rel_res = 1;
	printf("\033[32m[ModiPM]\n\033[0m");

	// 1e-5
	while (itn++<max_itn && fch>fch_thres) {
		// do one v_cycle
		rel_res = grids.v_cycle(1, 1);

		// project to balanced load on load region
		grids[0]->v3_copy(grids[0]->getDisplacement(), grids[0]->getForce());
		forceProject(grids[0]->getForce());

		// normalize projected force
		grids[0]->unitizeForce();

		// compute change of force
		fch = grids[0]->supportForceCh() / grids[0]->supportForceNorm();

		// update support force
		getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

		// output residual information
		printf("-- r_rel %6.2lf%%, fch %2.2lf%%\n", rel_res * 100, fch * 100);
	}

	double worstCompliance = grids[0]->compliance();
	if (isnan(worstCompliance)) { printf("\033[31m-- NaN occurred !\033[0m\n"); exit(-1); }
	grids[0]->_keyvalues["mu"] = worstCompliance;
	printf("-- Worst Compliance %6.3e\n", worstCompliance);
	return worstCompliance;
}

double MGPSOR(void)
{
	// do some modipm cycle
	double c = inCompleteModiPM();
	
	double lambda = c;
	double fch = 1;
	int itn = 0;
	double fsnorm = 1;
	double* xk[3];
	grids[0]->v3_create(xk);

	while (itn++ < 200 && fch > 1e-4 || itn < 3) {
		// backup old xk
		grids[0]->v3_copy(grids[0]->getDisplacement(), xk);

		// f = k * x
		grids[0]->applyK(grids[0]->getDisplacement(), grids[0]->getForce());

		// c = x * K * x
		c = grids[0]->v3_dot(grids[0]->getDisplacement(), grids[0]->getForce());

		// compute Px
		forceProject(grids[0]->getDisplacement());

		// compute support force norm
		fsnorm = grids[0]->v3_norm(grids[0]->getDisplacement());
		lambda = pow(fsnorm, 2) / c;

		// compute Eigen residual
		grids[0]->v3_add(-lambda, grids[0]->getForce(), 1, grids[0]->getDisplacement());
		grids[0]->force2matlab("feig");

		// compute force change
		grids[0]->v3_normalize(grids[0]->getDisplacement());
		fch = grids[0]->supportForceCh(grids[0]->getDisplacement());
		getForceSupport(grids[0]->getDisplacement(), grids[0]->getSupportForce());

		// reset displacement
		grids[0]->reset_displacement();

		// Preconditioned SOR
		double rel_res = 1;
		int vit = 0;
		while (rel_res > 1e-1 && vit++ < 3) {
			rel_res = grids.v_cycle();
		}
		grids[0]->displacement2matlab("u");
		grids[0]->residual2matlab("r");
		
		grids[0]->v3_add(grids[0]->getDisplacement(), 1, xk);
		printf("-- lam = %6.4e, fch = %6.2lf%%, r_rel = %6.2lf%%\n", lambda, fch * 100, rel_res * 100);
	}

	grids[0]->v3_destroy(xk);

	grids[0]->v3_scale(grids[0]->getDisplacement(), 1.0 / fsnorm);
	grids[0]->applyK(grids[0]->getDisplacement(), grids[0]->getForce());

	double c_worst = grids[0]->v3_dot(grids[0]->getDisplacement(), grids[0]->getForce());
	return c_worst;
}

double project_v_cycle(HierarchyGrid& grds) {
	//forceProject(grds[0]->getForce());

	for (int i = 0; i < grds.n_grid(); i++) {
		if (grds[i]->is_dummy()) continue;
		if (i > 0) {
			grds[i]->fineGrid->update_residual();
			//if (grds[i]->fineGrid->_layer == 0) forceProjectComplementary(grds[i]->fineGrid->getResidual());
			grds[i]->restrict_residual();
			grds[i]->reset_displacement();
		}
		if (i < grds.n_grid() - 1) {
			grds[i]->gs_relax();
		}
		else {
			grds[i]->solve_fem_host();
		}
	}

	for (int i = grds.n_grid() - 2; i >= 0; i--) {
		if (grds[i]->is_dummy()) continue;
		grds[i]->prolongate_correction();
		grds[i]->gs_relax();
	}

	grds[0]->force2matlab("f");
	grds[0]->update_residual();
	grds[0]->residual2matlab("r");
	grds[0]->displacement2matlab("u");

	forceProjectComplementary(grds[0]->getDisplacement());

	grds[0]->displacement2matlab("up");
	grds[0]->update_residual();

	grds[0]->residual2matlab("rp");
	forceProjectComplementary(grds[0]->getResidual());
	grds[0]->residual2matlab("rpp");

	return grds[0]->relative_residual();
}

void findAdjointVariabls(void) {
	//grids[0]->v3_add(2, grids[0]->getDisplacement(), -2 * grids[0]->_keyvalues["mu"], grids[0]->getWorstForce());
}

void optimization(void) {
	// allocated total size
	printf("[GPU] Total Mem :  %4.2lfGB\n", double(gpu_manager.size()) / 1024 / 1024 / 1024);

	grids.testShell();

	initDensities(params.volume_ratio);

	// DEBUG
	setDEBUG(false);
	grids[0]->eidmap2matlab("eidmap");
	grids[0]->vidmap2matlab("vidmap");

	grids[0]->randForce();

	//float Vgoal = 1;
	float Vgoal = params.volume_ratio;

	int itn = 0;

	snippet::converge_criteria stop_check(1, 2, 5e-3);

	std::vector<double> cRecord, volRecord;

	std::vector<double> tRecord;

	double Vc = Vgoal - params.volume_ratio;

	while (itn++ < 100) {
		printf("\n* \033[32mITER %d \033[0m*\n", itn);

		Vgoal *= (1 - params.volume_decrease);

		Vc = Vgoal - params.volume_ratio;

		if (Vgoal < params.volume_ratio) Vgoal = params.volume_ratio;

		// update numeric stencil after density changed
		update_stencil();

		// solve worst displacement by modified power method
		auto t0 = tictoc::getTag();
#if 1
		double c_worst = modifiedPM();
#else
		double c_worst = MGPSOR();
#endif
		auto t1 = tictoc::getTag();
		tRecord.emplace_back(tictoc::Duration<tictoc::ms>(t0, t1));

		grids.writeSupportForce(grids.getPath(snippet::formated("iter%d_fs", itn)));

		cRecord.emplace_back(c_worst); volRecord.emplace_back(Vgoal);

		if (stop_check.update(c_worst, &Vc) && Vgoal <= params.volume_ratio + 1e-3) break;

		grids.log(itn);
		// compute adjoint variables
		//findAdjointVariabls();

		// compute sensitivity
		computeSensitivity();

		// update density
		updateDensities(Vgoal);

		// DEBUG
		if (itn % 5 == 0) {
			grids.writeDensity(grids.getPath("out.vdb"));
			grids.writeSensitivity(grids.getPath("sens.vdb"));
		}
	}

	printf("\n=   finished   =\n");

	// write result density field
	grids.writeDensity(grids.getPath("out.vdb"));

	// write worst compliance record during optimization
	bio::write_vector(grids.getPath("cworst"), cRecord);

	// write volume record during optimization
	bio::write_vector(grids.getPath("vrec"), volRecord);

	// write time cost record during optimization
	bio::write_vector(grids.getPath("trec"), tRecord);

	// write last worst f and u
	grids.writeSupportForce(grids.getPath("flast"));
	grids.writeDisplacement(grids.getPath("ulast"));
}

grid::HierarchyGrid& getGrids(void)
{
	return grids;
}

void setBoundaryCondition(std::function<bool(double[3])> fixarea, std::function<bool(double[3])> loadarea, std::function<Eigen::Matrix<double, 3, 1>(double[3])> loadforce)
{
	grids._inFixedArea = fixarea;
	grids._inLoadArea = loadarea;
	grids._loadField = loadforce;
}

void initDensities(double rho)
{
	grids[0]->init_rho(rho);
}

void update_stencil(void)
{
	grids.update_stencil();
}

void test_rigid_displacement(void) {
	grids[0]->reset_residual();
	for (int n = 0; n < 3; n++) {
		for (int i = 0; i < 6; i++) {
			uploadRigidDisplacement(grids[0]->getDisplacement(), i);
			//grids[0]->reset_displacement();
			grids[0]->reset_force();
			grids[0]->update_residual();
			grids[0]->residual2matlab("r");
			printf("-- u_rigid %d res = %lf\n", i, grids[0]->residual());
		}
	}
}
