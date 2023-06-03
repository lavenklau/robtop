#include <vector>
#include "test_utils.h"
#include "optimization.h"
#include "projection.h"
#include "binaryIO.h"
#define EIGEN_USE_MKL_ALL
#include "Eigen/Eigen"
#include "Eigen/Eigenvalues"
#include "Spectra/SymEigsSolver.h"

#include "config_parser.h"
#include "matlab_utils.h"

#include "mma_t.h"
#include "tictoc.h"
#include "mesh_utils.h"
#include "algorithm"
#include "openvdb_wrapper_t.h"
#include "dir_utils.h"

void aggregatedSystem(const std::vector<int>& loadnodes, Eigen::MatrixXd& C, double err_tol = 1e-2, Eigen::MatrixXd* pZ = nullptr);
void setForce(Eigen::Matrix<double, -1, 1>& f);
Eigen::Matrix<double, -1, 1> solveFem(void);
Eigen::Matrix<double, -1, 1> assembleForce(const std::vector<int>& nodeids, double* pforce);


void testSpectra(void);

double TestSuit::ModifPM(int max_vcycle, bool resetf /*= true*/, int presmooth /*= 1*/, int postsmooth /*= 1*/, bool silence /*= false*/, bool updateStencil /*= true*/)
{
	//initDensities(1);

	// update stencil
	if (updateStencil) {
		grids.update_stencil();
	}

	if (resetf) {
		// generate random force
		grids[0]->randForce();

		// project force to balanced load on load region
		forceProject(grids[0]->getForce());

		// normalize force
		grids[0]->unitizeForce();
	}

	// reset displacement
	grids[0]->reset_displacement();

	// DEBUG
	grids[0]->force2matlab("finit");

	getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

	double fch = 1;

	int max_itn = 500;

	int itn = 0;

	printf("\033[32m[ModiPM]\n\033[0m");

	std::vector<float>  cworstrecord;

	std::vector<float> fchrecord;

	cworstrecord.emplace_back(0);

	snippet::SerialChar<double> fchserial;

	double rel_res = 1;

	auto start_tag = tictoc::getTag();

	std::vector<float> timeRecord;

	// 1e-5
	while (itn++ < max_itn && (fch > 1e-4 || rel_res > 1e-2)) {
		rel_res = 1;

		// do v_cycle
		int vitn = 0;

		while (vitn++ < max_vcycle && rel_res > 1e-2) {
			rel_res = grids.v_cycle(presmooth, postsmooth);
		}

		// compute and record compliance
		cworstrecord.emplace_back(grids[0]->compliance());

		// remove rigid displacement
		//if (!grids.hasSupport()) displacementProject(grids[0]->getDisplacement());

		// project to balanced load on load region
		grids[0]->v3_copy(grids[0]->getDisplacement(), grids[0]->getForce());
		forceProject(grids[0]->getForce());

		// normalize projected force
		grids[0]->unitizeForce();

		// compute change of force
		fch = grids[0]->supportForceCh() / grids[0]->supportForceNorm();

		// record fch
		fchrecord.emplace_back(fch);

		// check fch serial
		fchserial.add(fch);

		// update support force
		getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

		timeRecord.emplace_back(tictoc::Duration<tictoc::s>(start_tag, tictoc::getTag()));

		// output residual information
		printf("-- r_rel %6.2lf%%, fch %2.2lf%%\n", rel_res * 100, fch * 100);

	}

	grids[0]->_keyvalues["tmodpm"] = tictoc::Duration<tictoc::s>(start_tag, tictoc::getTag());

	// reduce relative residual 
	itn = 0;
#if 0
	while (rel_res > 1e-3 && itn++ < 50) { rel_res = grids.v_cycle(); printf("-- r_rel %6.2lf%%\n", rel_res * 100); }
#endif

	//grids.writeSupportForce(grids.getPath("fs"));

	double worstCompliance = grids[0]->compliance();

	if (isnan(worstCompliance)) { printf("\033[31m-- NaN occurred !\033[0m\n"); exit(-1); }

	grids[0]->_keyvalues["mu"] = worstCompliance;

	printf("-- Worst Compliance %6.3e\n", worstCompliance);

	char modifpmname[1000];
	sprintf_s(modifpmname, "[modipm]%dVcycle", max_vcycle);

	if (!silence) {
		bio::write_vector(grids.getPath(std::string(modifpmname) + "_cworst"), cworstrecord);
		bio::write_vector(grids.getPath(std::string(modifpmname) + "_fch"), fchrecord);
		bio::write_vector(grids.getPath(std::string(modifpmname) + "_time"), timeRecord);
		grids.writeSupportForce(grids.getPath(std::string(modifpmname)) + "fsworst");
		grids.writeDisplacement(grids.getPath(std::string(modifpmname)) + "uworst");
	}

	return worstCompliance;
}

void testPM(void) {
	double eig;
	Eigen::Matrix<double, -1, -1> Cn(10000, 10000);
	Cn.setRandom();
	Cn = Cn * Cn.transpose();

	eigen2ConnectedMatlab("C", Cn);

	//_TIC("t_qr");
	//Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, -1, -1>> eigsolver(Cn);
	//eig = eigsolver.eigenvalues().reverse()[0];
	//_TOC;
	//printf("-- [QR] eig = %6.2e, %6.2e\n", eig, tictoc::get_record("t_qr"));

	std::pair<double, Eigen::Matrix<double, -1, 1>> dvpair;
	_TIC("t_pm");
	dvpair = TestSuit::powerMethod(Cn);
	_TOC;
	printf("-- [PM] eig = %6.2e, %6.2e\n", dvpair.first, tictoc::get_record("t_pm"));

}

void initforceTest(void) {
	initDensities(1);
	update_stencil();

	setForceSupport(getForceNormal(), grids[0]->getForce());
	
	forceProject(grids[0]->getForce());

	grids[0]->unitizeForce();

	getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

	grids[0]->reset_displacement();

	double rel_res = 1;

	double fch = 0;

	int itn = 0;

	while (rel_res > 1e-3 && itn++ < 200) {
		rel_res = grids.v_cycle();

		// project to balanced load on load region
		grids[0]->v3_copy(grids[0]->getDisplacement(), grids[0]->getForce());
		forceProject(grids[0]->getForce());

		// normalize projected force
		grids[0]->unitizeForce();

		// compute change of force
		fch = grids[0]->supportForceCh() / grids[0]->supportForceNorm();

		// update support force
		getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

		if (itn % 20 == 1) {
			grids.writeSupportForce(grids.getPath("fs"));
		}

		printf("-- rel_res = %6.2lf%%,  fch = %6.2lf%%\n", rel_res * 100, fch * 100);
	}

}

void userTest(void) { 
#if 0
	testPM();
#elif 0
	TestSuit::testKernels();
#elif 0
	initDensities(1);
	update_stencil();
	grids.test_vcycle();
#elif 0
	testSpectra();
#elif 1
	initforceTest();
#endif
}

// generate plane topology


void TestSuit::testMain(const std::string& testname)
{
	if (testname == "" || testname == "None") return;

	printf("\n\033[33m-- doing test on %s\n\033[0m", testname.c_str());
	if (testname == "surfelements") {
		writeSurfaceElements(grids.getPath("surfelements"));
	}
	else if (testname == "fworst") {
		writeWorstForce(grids.getPath("fworst"));
	}
	else if (testname == "uworst") {
		writeWorstDisplacement(grids.getPath("uworst"));
	}
	else if (testname == "inaccurate") {
		test1VcycleAndAccurateSolve();
	}
	else if (testname == "inaccurateInterRho") {
		testinAccurateSolveInterDensity();
	}
	else if (testname == "inaccurateTotalTime") {
		testinAccurateSolveTotalTime();
	}
	else if (testname == "accuratePM") {
		testAccuratePMTotalTime();
	}
	else if (testname == "testcv") {
		testStressComplianceDistribution();
	}
	else if (testname == "testfem") {
		testFEM();
	}
	else if (testname == "testunitload") {
		testUnitLoad();
	}
	else if (testname == "test2011") {
		test2011();
	}
	else if (testname == "testtest2011") {
		testtest2011();
	}
	else if (testname == "test2019") {
		test2019();
	}
	else if (testname == "usertest") {
		userTest();
	}
	else if (testname == "testmem") {
		testMemoryUsage();
	}
	else if (testname == "modipmerror") {
		testModifiedPMError();
	}
	else if (testname == "replaymodipm") {
		testReplayModiPM();
	}
	else if (testname == "gstimes") {
		testGStimes();
	}
	else if (testname == "testordtop") {
		testOrdinaryTopopt();
	}
	else if (testname == "testdistributeforce") {
		testDistributeForceOpt();
	}
	else if (testname == "testmodipm") {
		testModifiedPM();
	}
	else if (testname == "verifymodipm") {
		verifyModifiedPM();
	}
	else if (testname == "testeigensolvers") {
		testDifferentEigenSolvers();
	}
	else if (testname == "testinitforce") {
		testDifferentInitForce();
	}
	else if (testname == "extractmesh") {
		
	}
	else if (testname == "surfvoxel") {
		testSurfVoxels();
	}
	else {
		printf("\033[31mUnknown test\033[0m\n");
		exit(-1);
	}

	exit(0);
}

void TestSuit::writeSurfaceElements(const std::string& filename)
{
	grids.writeSurfaceElement(filename);
}

void TestSuit::writeWorstForce(const std::string& filename)
{
	initDensities(1);
	ModifPM(1);
	grids.writeSupportForce(filename);
}

void TestSuit::writeWorstDisplacement(const std::string& filename)
{
	initDensities(1);
	ModifPM(1);
	grids.writeDisplacement(filename);
}

//  
void TestSuit::test1VcycleAndAccurateSolve(void)
{
	initDensities(1);

	// generate init force
	grids[0]->randForce();
	forceProject(grids[0]->getForce());
	grids[0]->unitizeForce();
	double* fdev[3];
	grids[0]->v3_create(fdev);
	grids[0]->v3_copy(grids[0]->getForce(), fdev);

	// do 1 vcycle
	ModifPM(1, false);

	grids[0]->v3_copy(fdev, grids[0]->getForce());
	// accurate solve
	ModifPM(100000, false);
}

void TestSuit::testinAccurateSolveTotalTime(void)
{
	initDensities(1);

	// DEBUG
	setDEBUG(false);
	grids[0]->eidmap2matlab("eidmap");
	grids[0]->vidmap2matlab("vidmap");

	float Vgoal = 1;

	int itn = 0;

	snippet::converge_criteria stop_check(1, 5, 1e-3);

	std::vector<double> cRecord, volRecord;

	double Vc = Vgoal - params.volume_ratio;

	std::vector<float> tmodipm;

	while (itn++ < 100) {
		printf("\n* \033[32mITER %d \033[0m*\n", itn);

		Vgoal *= (1 - params.volume_decrease);

		Vc = Vgoal - params.volume_ratio;

		if (Vgoal < params.volume_ratio) Vgoal = params.volume_ratio;

		// update numeric stencil after density changed
		update_stencil();

		// solve worst displacement by modified power method
		double c_worst = ModifPM(10000);

		tmodipm.emplace_back(grids[0]->_keyvalues["tmodpm"]);

		cRecord.emplace_back(c_worst); volRecord.emplace_back(Vgoal);

		if (stop_check.update(c_worst, &Vc) && Vgoal <= params.volume_ratio) break;

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

	// write solve time in each modipm
	bio::write_vector(grids.getPath("tmodipm"), tmodipm);
}

void TestSuit::testinAccurateSolveInterDensity(void)
{
	grids.readDensity(FLAGS_inputdensity);
	// generate init force
	grids[0]->randForce();
	forceProject(grids[0]->getForce());
	grids[0]->unitizeForce();
	double* fdev[3];
	grids[0]->v3_create(fdev);
	grids[0]->v3_copy(grids[0]->getForce(), fdev);

	// do 1 vcycle
	ModifPM(1, false);

	// restore forec
	grids[0]->v3_copy(fdev, grids[0]->getForce());

	// accurate solve
	ModifPM(10000, false);
}

void TestSuit::testStressComplianceDistribution(void)
{
	grids.readDensity(FLAGS_inputdensity);
	bool fu_exist = true;
	try {
		grids[0]->reset_force();
		grids[0]->readSupportForce(grids.getPath("flast"));
		grids[0]->readDisplacement(grids.getPath("ulast"));
	}
	catch (...) {
		printf("\033[31m Read force and displacement Failed\033[0m\n");
		fu_exist = false;
	}

	update_stencil();

	if (!fu_exist) {
#if 1
		ModifPM(1, true, 1, 1, true, false);
#else
		setForceSupport(getForceNormal(), grids[0]->getForce());
		grids[0]->force2matlab("f");
		grids[0]->reset_displacement();
		solveFem();
#endif
	}

	auto m = mesh_utils::ReadMesh(FLAGS_testmesh);

	auto p4 = mesh_utils::FlattenVertex(m);

	std::vector<double> clist;
	std::vector<double> vonstress;
	stressAndComplianceOnVertex(p4, clist, vonstress);

	bio::write_vector(grids.getPath("clist"), clist);
	bio::write_vector(grids.getPath("vonlist"), vonstress);
	grids.writeSupportForce(grids.getPath("fs"));

	return;
}

void TestSuit::testAccuratePMTotalTime(void)
{
	grids.testShell();

#if 0
	initDensities(1);
	float Vgoal = 1;
#else
	initDensities(params.volume_ratio);
	float Vgoal = params.volume_ratio;
#endif

	// DEBUG
	setDEBUG(false);
	grids[0]->eidmap2matlab("eidmap");
	grids[0]->vidmap2matlab("vidmap");

	int itn = 0;

	snippet::converge_criteria stop_check(1, 3, 5e-2);

	std::vector<double> cRecord, volRecord;

	double Vc = Vgoal - params.volume_ratio;

	std::vector<double> tlist;

	while (itn++ < 100) {
		printf("\n* \033[32mITER %d \033[0m*\n", itn);

		Vgoal *= (1 - params.volume_decrease);

		Vc = Vgoal - params.volume_ratio;

		if (Vgoal < params.volume_ratio) Vgoal = params.volume_ratio;

		// update numeric stencil after density changed
		update_stencil();

		auto tic_pm = tictoc::getTag();
		// solve worst displacement by modified power method
		double c_worst = ModifPM(1000, true, 1, 1, true, false);
		double t_pm = tictoc::Duration<tictoc::s>(tic_pm, tictoc::getTag());
		tlist.emplace_back(t_pm);

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

	// write time usage on modified PM during optimization
	bio::write_vector(grids.getPath("tpm"), tlist);
}

void TestSuit::testOrdinaryTopopt(void)
{

	// set force
	grids[0]->reset_force();
	setForceSupport(getPreloadForce(), grids[0]->getForce());

	if (!grids.hasSupport()) {
		forceProject(grids[0]->getForce());
	}

	grids[0]->force2matlab("fn");

	grids.resetAllResidual();
	grids[0]->reset_displacement();
	grids.writeSupportForce(grids.getPath("fs"));

#if 1
	initDensities(params.volume_ratio);
	float Vgoal = params.volume_ratio;
#else
	initDensities(1);
	float Vgoal = 1;
#endif

	int itn = 0;

	snippet::converge_criteria stop_check(1, 5, 1e-3);

	std::vector<double> cRecord, volRecord;

	double Vc = Vgoal - params.volume_ratio;

	std::vector<float> tmodipm;

	while (itn++ < 100) {
		printf("\n* \033[32mITER %d \033[0m*\n", itn);
		Vgoal *= (1 - params.volume_decrease);
		Vc = Vgoal - params.volume_ratio;
		if (Vgoal < params.volume_ratio) Vgoal = params.volume_ratio;
		// update numeric stencil after density changed
		update_stencil();
		// solve displacement 
		//double c = grids.solveFEM();
		double rel_res = 1;
		int femit = 0;
		while (rel_res > 1e-2 && femit++ < 50) {
			rel_res = grids.v_cycle(1, 1);
		}
		double c = grids[0]->compliance();
		printf("-- c = %6.4e   r = %4.2lf%%\n", c, rel_res * 100);
		if (isnan(c) || abs(c) < 1e-11) { printf("\033[31m-- Error compliance\033[0m\n"); exit(-1); }
		cRecord.emplace_back(c); volRecord.emplace_back(Vgoal);
		if (stop_check.update(c, &Vc) && Vgoal <= params.volume_ratio) break;
		grids.log(itn);
		// compute sensitivity
		computeSensitivity();
		// update density
		updateDensities(Vgoal);
	}

	printf("\n=   finished   =\n");

	// write result density field
	grids.writeDensity(grids.getPath("out.vdb"));

	// write worst compliance record during optimization
	bio::write_vector(grids.getPath("c"), cRecord);

	// write volume record during optimization
	bio::write_vector(grids.getPath("vrec"), volRecord);
}

std::pair<double, Eigen::VectorXd> SpectraFindLargestEigenPair(const Eigen::MatrixXd& mat) {
	// construct matrix operation object using the wrapper class DenseSymMatProd
	Spectra::DenseSymMatProd<double> op(mat);

	// construct eigen solver object, requesting the largest eigenvalues
	Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, 4);

	// initialize and compute
	eigs.init();
	// nconv is the number of converged eigenvalues
	int nconv = eigs.compute(Spectra::SortRule::LargestAlge);

	std::cout << "nconv = " << nconv << std::endl;

	// retrieve results
	Eigen::VectorXd evalues;
	Eigen::Matrix<double, -1, -1> evectors;
	if (eigs.info() == Spectra::CompInfo::Successful) {
		evalues = eigs.eigenvalues();
		evectors = eigs.eigenvectors();
	}
	std::pair<double, Eigen::VectorXd> result;
	result.first = evalues[0];
	result.second = evectors.col(0);
	std::cout << "eigenvalues = " << result.first << std::endl;
	return result;
}

void testSpectra(void) {
	Eigen::MatrixXd A = Eigen::MatrixXd::Random(1000, 1000);
	Eigen::MatrixXd M = A * A.transpose();
	std::pair<double, Eigen::VectorXd> eigenpair;
	_TIC("t_specta")
	eigenpair = SpectraFindLargestEigenPair(M);
	_TOC;
	std::cout << "-- [Spectra]  eig = " << eigenpair.first << ", time = " << tictoc::get_record("t_spectra") << std::endl;

	_TIC("t_pm")
	eigenpair = TestSuit::powerMethod(M);
	_TOC;
	std::cout << "-- [PM]  eig = " << eigenpair.first << ", time = " << tictoc::get_record("t_pm") << std::endl;
}

void TestSuit::testDifferentEigenSolvers(void)
{
	grids.testShell();
	if (FLAGS_inputdensity == "") {
		initDensities(1);
	} else {
		grids.readDensity(FLAGS_inputdensity);
	}

	std::ofstream ofs(grids.getPath("record.txt"));

	ofs << "n_elements = " << grids[0]->n_gselements << std::endl;

	// update grid stencil
	grids.update_stencil();

	const std::vector<int>& loadnodes = getLoadNodes();
	Eigen::Matrix<double, -1, -1> C(loadnodes.size() * 3, loadnodes.size() * 3);

	// compute volume
	double vol = grids[0]->volumeRatio();

	// solve C
	_TIC("t_cpm")
	aggregatedSystem(loadnodes, C);
	_TOC

	// solve largest eigenvector
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig;
	_TIC("t_cpm")
	eig.compute((C + C.transpose()) / 2);
	_TOC

	double t_computeC = tictoc::get_record("t_cpm");

	std::pair<double, Eigen::VectorXd> vv;
	_TIC("t_cpm")
	vv = powerMethod(C);
	_TOC

	double c_worst = vv.first;
	auto& w_worst = vv.second;
	Eigen::Matrix<double, -1, 1> f_worst = assembleForce(loadnodes, w_worst.data());
	setForce(f_worst);
	grids.writeSupportForce(grids.getPath("fs_cpm"));
	printf("--[CPM] c_worst = %6.4e, time = %4.2lf\n", c_worst, tictoc::get_record("t_cpm"));
	ofs << "[CPM] c_worst = " << c_worst << ", time = " << tictoc::get_record("t_cpm") << std::endl;

	double c_wmodi;
	_TIC("t_mpm")
	c_wmodi = modifiedPM();
	_TOC
	grids.writeSupportForce(grids.getPath("fs_mpm"));
	printf("--[MPM] c_wmodi = %6.2e, time = %4.2lf\n", c_wmodi, tictoc::get_record("t_mpm"));
	ofs << "[MPM] c_wmodi = " << c_wmodi << ", time = " << tictoc::get_record("t_mpm") << std::endl;

	double c_waccu;
	_TIC("t_accupm")
	c_waccu = ModifPM(100, true, 1, 1, true, false);
	_TOC
	grids.writeSupportForce(grids.getPath("fs_accupm"));
	printf("--[AccuPM] c_worst = %6.2e, time = %4.2lf\n", c_waccu, tictoc::get_record("t_accupm"));
	ofs << "[AccuPM] c_worst = " << c_waccu << ", time = " << tictoc::get_record("t_accupm") << std::endl;

	std::cout << "t_cpm = " << tictoc::get_record("t_cpm") << " ms" << std::endl;
	std::cout << "t_mpm = " << tictoc::get_record("t_mpm") << " ms" << std::endl;
	std::cout << "t_accupm = " << tictoc::get_record("t_accupm") << " ms" << std::endl;

	ofs << "t_cpm = " << tictoc::get_record("t_cpm") << " ms" << std::endl;
	ofs << "t_mpm = " << tictoc::get_record("t_mpm") << " ms" << std::endl;
	ofs << "t_accupm = " << tictoc::get_record("t_accupm") << " ms" << std::endl;
	ofs.close();
}

void TestSuit::testDifferentInitForce(void)
{
	std::vector<std::string> vdbfiles;

	if (FLAGS_inputdensity == "") {
		printf("-- No vdb files given\n");
		initDensities(1);
		vdbfiles.emplace_back("");
	}
	else {
		printf("-- using vdb directory %s\n", FLAGS_inputdensity.c_str());
		std::vector<std::string> listfiles = dir_utils::listFile(FLAGS_inputdensity.c_str());
		std::cout << "-- list " << listfiles.size() << " files" << std::endl;
		vdbfiles = dir_utils::filterFiles(listfiles, ".vdb");
		std::cout << "-- Read " << vdbfiles.size() << " vdb files" << std::endl;
	}

	for (int j = 0; j < vdbfiles.size(); j++) {
		if (vdbfiles[j] != "") {
			std::cout << "-- Reading " << vdbfiles[j] << std::endl;
			grids.readDensity(vdbfiles[j]);
		}
		update_stencil();
		std::vector<int> itn_list;
		for (int k = 0; k < 20; k++) {
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

			int max_itn = 500;

			int itn = 0;

			printf("\033[32m[ModiPM]\n\033[0m");

			std::vector<float>  cworstrecord;

			std::vector<float> fchrecord;

			cworstrecord.emplace_back(0);

			snippet::SerialChar<double> fchserial;

			double rel_res = 1;

			auto start_tag = tictoc::getTag();

			std::vector<float> timeRecord;

			// 1e-5
			while (itn++ < max_itn && (fch > 1e-4 || rel_res > 1e-2)) {
				rel_res = 1;

				// do v_cycle
				int vitn = 0;

				rel_res = grids.v_cycle();

				// compute and record compliance
				cworstrecord.emplace_back(grids[0]->compliance());

				// remove rigid displacement
				//if (!grids.hasSupport()) displacementProject(grids[0]->getDisplacement());

				// project to balanced load on load region
				grids[0]->v3_copy(grids[0]->getDisplacement(), grids[0]->getForce());
				forceProject(grids[0]->getForce());

				// normalize projected force
				grids[0]->unitizeForce();

				// compute change of force
				fch = grids[0]->supportForceCh() / grids[0]->supportForceNorm();

				// record fch
				fchrecord.emplace_back(fch);

				// check fch serial
				fchserial.add(fch);

				// update support force
				getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

				timeRecord.emplace_back(tictoc::Duration<tictoc::s>(start_tag, tictoc::getTag()));

				// output residual information
				printf("--[%d] r_rel %6.2lf%%, fch %2.2lf%%\n", itn, rel_res * 100, fch * 100);
			}

			itn_list.emplace_back(itn);

			grids[0]->_keyvalues["tmodpm"] = tictoc::Duration<tictoc::s>(start_tag, tictoc::getTag());

			// reduce relative residual 
			itn = 0;
#if 0
			while (rel_res > 1e-3 && itn++ < 50) { rel_res = grids.v_cycle(); printf("-- r_rel %6.2lf%%\n", rel_res * 100); }
#endif

			//grids.writeSupportForce(grids.getPath("fs"));

			double worstCompliance = grids[0]->compliance();

			if (isnan(worstCompliance)) { printf("\033[31m-- NaN occurred !\033[0m\n"); exit(-1); }

			grids[0]->_keyvalues["mu"] = worstCompliance;

			printf("-- Worst Compliance %6.3e\n", worstCompliance);

			char modifpmname[1000];
			sprintf_s(modifpmname, "%s_rand%d", dir_utils::path2filename(vdbfiles[j]).c_str(), k);

			bio::write_vector(grids.getPath(std::string(modifpmname) + "_cworst"), cworstrecord);
			bio::write_vector(grids.getPath(std::string(modifpmname) + "_fch"), fchrecord);
			bio::write_vector(grids.getPath(std::string(modifpmname) + "_time"), timeRecord);
			bio::write_vector(grids.getPath(snippet::formated("%s_rand", dir_utils::path2filename(vdbfiles[j]).c_str()) + "_itn"), itn_list);
			grids.writeSupportForce(grids.getPath(std::string(modifpmname)) + "_fsworst");
			//grids.writeDisplacement(grids.getPath(std::string(modifpmname)) + "_uworst");
		}
	}
}

void TestSuit::testMemoryUsage(void)
{
	initDensities(FLAGS_volume_ratio);

	update_stencil();

	grids[0]->randForce();

	forceProject(grids[0]->getForce());

	grids[0]->reset_displacement();

	solveFem();

	computeSensitivity();

	updateDensities(FLAGS_volume_ratio);

	std::cout << "-- " << grids[0]->n_gselements << " elements" << std::endl;

	size_t gmem = gpu_manager.size();
	std::cout << "-- gm " << gmem / 1024 / 1024 << " MB" << std::endl;

	size_t projmem = projectionGetMem();
	std::cout << "-- projection " << projmem / 1024 / 1024 << " MB" << std::endl;

	size_t temp_buf_size = grid::Grid::_tmp_buf_size;
	std::cout << "-- temp buf size " << temp_buf_size / 1024 / 1024 << " MB" << std::endl;

	size_t totaluse = gmem + projmem + temp_buf_size;
	std::cout << "-- total memory usage " << totaluse / 1024 / 1024 << " MB" << std::endl;

	std::ofstream ofs(grids.getPath("mem.txt"));

	ofs << grids[0]->n_gselements << " elements" << std::endl;
	ofs << "gm " << gmem / 1024 / 1024 << " MB, " << (double)(gmem / 1024 / 1024) / 1024 << " GB" << std::endl;
	ofs << "proj " << projmem / 1024 / 1024 << " MB, " << (double)(projmem / 1024 / 1024) / 1024 << " GB" << std::endl;
	ofs << "temp " << temp_buf_size / 1024 / 1024 << " MB, " << (double)(temp_buf_size / 1024 / 1024) / 1024 << " GB" << std::endl;
	ofs << "total " << totaluse / 1024 / 1024 << " MB, " << (double)(totaluse / 1024 / 1024) / 1024 << " GB" << std::endl;
	ofs.close();
}

void TestSuit::testReplayModiPM(void)
{
	std::vector<std::string> vdbfiles;
	if (FLAGS_inputdensity != "") {
		vdbfiles = dir_utils::matchFiles(dir_utils::listFile(FLAGS_inputdensity), "density\\d+.vdb");
		printf("-- read %d vdb files\n", vdbfiles.size());
	}
	else {
		std::cout << "\033[31m" << "No vdb files given" << "\033[0m" << std::endl;
		exit(-1);
	}

	std::vector<std::string> filevdb;
	std::vector<double> t_vdb;

	for (int i = 0; i < vdbfiles.size(); i++) {
		std::cout << "-- using vdb file " << vdbfiles[i] << std::endl;
		grids.readDensity(vdbfiles[i]);

		filevdb.emplace_back(dir_utils::path2filename(vdbfiles[i]));

		grids.update_stencil();

		auto t0 = tictoc::getTag();

		modifiedPM();

		auto t1 = tictoc::getTag();

		double t_dur = tictoc::Duration<tictoc::ms>(t0, t1);

		t_vdb.emplace_back(t_dur);

		std::ofstream ofs(grids.getPath("replay_time.txt"));

		for (int k = 0; k < i + 1; k++) {
			ofs << "vdb " << filevdb[k] << ", " << "time " << t_vdb[k] << std::endl;
		}

		ofs.close();
	}
}

void TestSuit::testModifiedPMError(void)
{
	std::vector<std::string> vdbfiles;
	if (FLAGS_inputdensity == "") {
		std::cout << "\033[31m" << "No input density given" << "\033[0m" << std::endl;
	}
	else {
		std::vector<std::string> listfiles = dir_utils::listFile(FLAGS_inputdensity);
		std::cout << "-- list " << listfiles.size() << " files" << std::endl;
		vdbfiles = dir_utils::filterFiles(listfiles, ".vdb");
		std::cout << "-- Read " << vdbfiles.size() << " vdb files" << std::endl;
	}

	std::vector<int> loadnodes = getLoadNodes();
	Eigen::MatrixXd C(loadnodes.size() * 3, loadnodes.size() * 3);

	for (int k = 0; k < vdbfiles.size(); k++) {
		std::cout << "-- Reading file " << vdbfiles[k] << std::endl;

		grids.readDensity(vdbfiles[k]);

		std::string fn = dir_utils::path2filename(vdbfiles[k]);

		update_stencil();

		auto t0 = tictoc::getTag();
		aggregatedSystem(loadnodes, C, 1e-4);
		auto dv = SpectraFindLargestEigenPair(C);
		auto t1 = tictoc::getTag();

		std::cout << "-- c_worst = " << dv.first << std::endl;

		Eigen::Matrix<double, -1, 1> f_worst = assembleForce(loadnodes, dv.second.data());
		setForce(f_worst);
		grids.writeSupportForce(grids.getPath(fn + "_fs_true"));

		std::ofstream ofs(grids.getPath(fn + "_log.txt"));
		ofs << "time " << tictoc::Duration<tictoc::ms>(t0, t1) << " ms, " << "c_worst " << dv.first << std::endl;

		ofs.close();
	}
}

void TestSuit::testDistributeForceOpt(void)
{
	// set force
	grids[0]->reset_force();
	//setForceSupport(getPreloadForce(), grids[0]->getForce());
	setForceSupport(getForceNormal(), grids[0]->getForce());

	if (!grids.hasSupport()) {
		forceProject(grids[0]->getForce());
	}

	grids.writeSupportForce(grids.getPath("fs"));

#if 1
	initDensities(params.volume_ratio);
	float Vgoal = params.volume_ratio;
#else
	initDensities(1);
	float Vgoal = 1;
#endif

	int itn = 0;

	snippet::converge_criteria stop_check(1, 2, 5e-3);

	std::vector<double> cRecord, volRecord;

	double Vc = Vgoal - params.volume_ratio;

	std::vector<float> tmodipm;

	double Md = 1, MdThres = 0.08;


	while (itn++ < 100) {
		printf("\n* \033[32mITER %d \033[0m*\n", itn);
		Vgoal *= (1 - params.volume_decrease);
		Vc = Vgoal - params.volume_ratio;
		if (Vgoal < params.volume_ratio) Vgoal = params.volume_ratio;
		// update numeric stencil after density changed
		update_stencil();
		// solve displacement 
		//double c = grids.solveFEM();
		double rel_res = 1;
		int femit = 0;
		while (rel_res > 1e-2 && femit++ < 50) {
			rel_res = grids.v_cycle(1, 1);
		}
		double c = grids[0]->compliance();
		printf("-- c = %6.4e   r = %4.2lf%%  md = %4.2lf%%\n", c, rel_res * 100, Md * 100);
		if (isnan(c) || abs(c) < 1e-11) { printf("\033[31m-- Error compliance\033[0m\n"); exit(-1); }
		cRecord.emplace_back(c); volRecord.emplace_back(Vgoal);
		if (stop_check.update(c, &Vc) && Vgoal <= params.volume_ratio && Md < MdThres) break;
		grids.log(itn);
		// compute sensitivity
		computeSensitivity();
		// update density
		updateDensities(Vgoal);

		Md = grids[0]->densityDiscretiness();
	}

	printf("\n=   finished   =\n");

	// write result density field
	grids.writeDensity(grids.getPath("out.vdb"));

	// write worst compliance record during optimization
	bio::write_vector(grids.getPath("c"), cRecord);

	// write volume record during optimization
	bio::write_vector(grids.getPath("vrec"), volRecord);

}

void TestSuit::extractMeshFromDensity(void)
{
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec<3, int>> trias;
	std::vector<glm::vec<4, int>> quads;
	openvdb_wrapper_t<float>::meshFromFile(FLAGS_inputdensity, vertices, trias, quads, 0.5, true);
}

void TestSuit::testModifiedPM(void)
{
	//initDensities(1);
	std::string inputrho = FLAGS_inputdensity;
	if (inputrho != "") {
		grids.readDensity(inputrho);
	} else {
		initDensities(params.volume_ratio);
	}

	// update stencil
	grids.update_stencil();

	// generate random force
	grids[0]->randForce();

	// project force to balanced load on load region
	forceProject(grids[0]->getForce());

	// normalize force
	grids[0]->unitizeForce();

	// reset displacement
	grids[0]->reset_displacement();

	// DEBUG
	grids[0]->force2matlab("finit");

	getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

	grids.writeSupportForce(grids.getPath("fsinit"));

	double fch = 1;

	int max_itn = 500;

	int itn = 0;

	printf("\033[32m[ModiPM]\n\033[0m");

	std::vector<float>  cworstrecord;

	std::vector<float> fchrecord;

	std::vector<float> resrecord;

	//cworstrecord.emplace_back(0);

	snippet::SerialChar<double> fchserial;

	double rel_res = 1;

	auto start_tag = tictoc::getTag();

	std::vector<float> timeRecord;

	// 1e-5
	while (itn++ < max_itn && (fch > 1e-4 || rel_res > 1e-2)) {
		rel_res = 1;

		// do v_cycle
		int vitn = 0;

		rel_res = grids.v_cycle(1, 1);

		// compute and record compliance
		cworstrecord.emplace_back(grids[0]->compliance());

		grids.writeSupportForce(grids.getPath(snippet::formated("pmit%d_fs", itn)));

		resrecord.emplace_back(rel_res);

		// remove rigid displacement
		//if (!grids.hasSupport()) displacementProject(grids[0]->getDisplacement());

		// project to balanced load on load region
		grids[0]->v3_copy(grids[0]->getDisplacement(), grids[0]->getForce());
		forceProject(grids[0]->getForce());

		// normalize projected force
		grids[0]->unitizeForce();

		// compute change of force
		fch = grids[0]->supportForceCh() / grids[0]->supportForceNorm();

		// record fch
		fchrecord.emplace_back(fch);

		// check fch serial
		fchserial.add(fch);

		// update support force
		getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

		timeRecord.emplace_back(tictoc::Duration<tictoc::s>(start_tag, tictoc::getTag()));

		// output residual information
		printf("-- r_rel %6.2lf%%, fch %2.2lf%%, c = %6.4e\n", rel_res * 100, fch * 100, *cworstrecord.rbegin());

	}

	grids[0]->_keyvalues["tmodpm"] = tictoc::Duration<tictoc::s>(start_tag, tictoc::getTag());

	// reduce relative residual 
	itn = 0;
#if 0
	while (rel_res > 1e-3 && itn++ < 50) { rel_res = grids.v_cycle(); printf("-- r_rel %6.2lf%%\n", rel_res * 100); }
#endif

	//grids.writeSupportForce(grids.getPath("fs"));

	double worstCompliance = grids[0]->compliance();

	if (isnan(worstCompliance)) { printf("\033[31m-- NaN occurred !\033[0m\n"); exit(-1); }

	grids[0]->_keyvalues["mu"] = worstCompliance;

	printf("-- Worst Compliance %6.3e\n", worstCompliance);

	bio::write_vector(grids.getPath("cworst"), cworstrecord);
	bio::write_vector(grids.getPath("fch"), fchrecord);
	bio::write_vector(grids.getPath("res"), resrecord);
	bio::write_vector(grids.getPath("time"), timeRecord);
}

void TestSuit::verifyModifiedPM(void)
{
	initDensities(1);

	update_stencil();

	const std::vector<int>& loadnodes = getLoadNodes();

	Eigen::Matrix<double, -1, -1> C(loadnodes.size() * 3, loadnodes.size() * 3);

	// compute worst case by aggregated system
	printf("-- Solving Aggregated system...\n");
	aggregatedSystem(loadnodes, C);
	C = (C + C.transpose()) / 2;
	printf("-- Solving eigen pair...\n");
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, -1, -1>> eigs(C);
	Eigen::Matrix<double, -1, 1> w_worst = eigs.eigenvectors().rowwise().reverse().col(0);
	double c_worst = eigs.eigenvalues().reverse()[0];
	double c_secworst = eigs.eigenvalues().reverse()[1];
	printf("-- [Aggre]   c_worst = %6.2e (%6.2e)\n", c_worst, c_secworst);

	// assemble worst force 
	Eigen::Matrix<double, -1, 1> f_worst = assembleForce(loadnodes, w_worst.data());
	// solve worst displacement
	setForce(f_worst);
	Eigen::Matrix<double, -1, 1> uworst = solveFem();
	// compute compliance
	double fu = f_worst.dot(uworst);
	// write displacement of aggregated C to u_agg
	grids.writeDisplacement(grids.getPath("u_agg"));
	grids.writeSupportForce(grids.getPath("fs_agg"));

	printf("-- Solving by modified power method...\n");
	grids[0]->randForce();
	// project force to balanced load on load region
	forceProject(grids[0]->getForce());
	// normalize force
	grids[0]->unitizeForce();
	// reset displacement
	grids[0]->reset_displacement();
	double c_pm = ModifPM(1, 0, 1, 1, true, false);
	getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());
	grids.writeSupportForce(grids.getPath("fs_pm"));
	grids.writeDisplacement(grids.getPath("u_pm"));

	printf("-- [ModifPM] c_worst = %6.2e \n", c_pm);

}

void TestSuit::testFEM(void)
{
	if (FLAGS_inputdensity == "") {
		initDensities(1);
	} else {
		grids.readDensity(FLAGS_inputdensity);
	}

	update_stencil();

	setForceSupport(getPreloadForce(), grids[0]->getForce());

	grids[0]->reset_displacement();
	
	solveFem();

	auto m = mesh_utils::ReadMesh(FLAGS_testmesh);

	auto p4 = mesh_utils::FlattenVertex(m);

	std::vector<double> clist;
	std::vector<double> vonstress;
	stressAndComplianceOnVertex(p4, clist, vonstress);

	bio::write_vector(grids.getPath("clist"), clist);
	bio::write_vector(grids.getPath("vonlist"), vonstress);

	return;

}

void TestSuit::testUnitLoad(void)
{
	if (FLAGS_inputdensity == "") {
		initDensities(1);
	}else {
		std::cout << "-- Reading file " << FLAGS_inputdensity << std::endl;
		grids.readDensity(FLAGS_inputdensity);
	}

	update_stencil();

	setForceSupport(getForceNormal(), grids[0]->getForce());

	grids[0]->unitizeForce();

	grids[0]->reset_displacement();

	grids.writeSupportForce(grids.getPath("fs"));

	solveFem();

	double cload = grids[0]->compliance();

	std::cout << "-- compliance " << cload << std::endl;
}

void TestSuit::testBESO(void)
{
	initDensities(1);

	
}

void TestSuit::testSurfVoxels(void)
{
	grids.writeSurfaceElement(grids.getPath("surfvoxelcenter"));
}

void TestSuit::testGStimes(void)
{
	int nv = 1;
	int pregs = 1;
	
	initDensities(1);

	int n_gs = 1;
	int i = 1;
	do {
		gsPM(nv, n_gs, n_gs);
		n_gs = i * 5; i++;
	} while (n_gs < 31);
}

double TestSuit::gsPM(int n_vcycle, int presmooth, int postsmooth)
{
	// update stencil
	grids.update_stencil();

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

	int max_itn = 500;

	int itn = 0;

	printf("\033[32m[ModiPM]\n\033[0m");

	std::vector<float>  cworstrecord;

	std::vector<float> fchrecord;

	std::vector<float> resrecord;

	cworstrecord.emplace_back(0);

	resrecord.emplace_back(1);

	snippet::SerialChar<double> fchserial;

	double rel_res = 1;

	auto start_tag = tictoc::getTag();

	std::vector<float> timeRecord;

	// 1e-5
	while (itn++ < max_itn && (fch > 1e-4 || rel_res > 1e-2)) {
		rel_res = 1;

		// do v_cycle
		int vitn = 0;

		while (vitn++ < n_vcycle && rel_res > 1e-2) {
			rel_res = grids.v_cycle(presmooth, postsmooth);
		}

		// record residual
		resrecord.emplace_back(rel_res);

		// compute and record compliance
		cworstrecord.emplace_back(grids[0]->compliance());

		// project to balanced load on load region
		grids[0]->v3_copy(grids[0]->getDisplacement(), grids[0]->getForce());
		forceProject(grids[0]->getForce());

		// normalize projected force
		grids[0]->unitizeForce();

		// compute change of force
		fch = grids[0]->supportForceCh() / grids[0]->supportForceNorm();

		// record fch
		fchrecord.emplace_back(fch);

		// check fch serial
		fchserial.add(fch);

		// update support force
		getForceSupport(grids[0]->getForce(), grids[0]->getSupportForce());

		timeRecord.emplace_back(tictoc::Duration<tictoc::s>(start_tag, tictoc::getTag()));

		// output residual information
		printf("-- r_rel %6.2lf%%, fch %2.2lf%%\n", rel_res * 100, fch * 100);
	}

	grids[0]->_keyvalues["tmodpm"] = tictoc::Duration<tictoc::s>(start_tag, tictoc::getTag());

	// reduce relative residual 
	itn = 0;
	while (rel_res > 1e-3 && itn++ < 50) { rel_res = grids.v_cycle(); printf("-- r_rel %6.2lf%%\n", rel_res * 100); }

	grids.writeSupportForce(grids.getPath("fs"));

	double worstCompliance = grids[0]->compliance();

	if (isnan(worstCompliance)) { printf("\033[31m-- NaN occurred !\033[0m\n"); exit(-1); }

	grids[0]->_keyvalues["mu"] = worstCompliance;

	printf("-- Worst Compliance %6.3e\n", worstCompliance);

	char modifpmname[1000];

	sprintf_s(modifpmname, "[modipm]%dV%dpre%dpost", n_vcycle, presmooth, postsmooth);

	bio::write_vector(grids.getPath(std::string(modifpmname) + "_cworst"), cworstrecord);
	bio::write_vector(grids.getPath(std::string(modifpmname) + "_fch"), fchrecord);
	bio::write_vector(grids.getPath(std::string(modifpmname) + "_time"), timeRecord);
	bio::write_vector(grids.getPath(std::string(modifpmname) + "_res"), resrecord);
	grids.writeSupportForce(grids.getPath(std::string(modifpmname)) + "fsworst");
	grids.writeDisplacement(grids.getPath(std::string(modifpmname)) + "uworst");

	return worstCompliance;
}

void aggregatedSystem(const std::vector<int>& loadnodes, Eigen::MatrixXd& C, double err_tol /*= 1e-2*/, Eigen::MatrixXd* pZ /*= nullptr*/) {
	auto& Z = *pZ;
	if (pZ != nullptr) {
		Z.resize(grids[0]->n_nodes(), loadnodes.size() * 3);
		Z.fill(0);
	}
	C.resize(loadnodes.size() * 3, loadnodes.size() * 3);
	C.fill(0);
	grids[0]->reset_force();
	grids.resetAllResidual();
	grids[0]->reset_displacement();
	// compute Aggregated system
	int lastloadid = 0;
	for (int i = 0; i < loadnodes.size(); i++) {
		printf("load nodes %d (%d)", i, loadnodes[i]);
		int loadid = loadnodes[i];
		// clear last load buf
		for (int j = 0; j < 3; j++) {
			double f = 0;
			gpu_manager_t::upload_buf(grids[0]->getForce()[j] + lastloadid, &f, sizeof(double));
		}
		for (int j = 0; j < 3; j++) {
			double fj[3] = { 0,0,0 };
			fj[j] = 1;
			// upload fj
			for (int k = 0; k < 3; k++) {
				gpu_manager_t::upload_buf(grids[0]->getForce()[k] + loadid, fj + k, sizeof(double));
			}

			// project force if support exists
			if (!grids.hasSupport()) { forceProject(grids[0]->getForce()); }

			double rel_res = 1;
			int nv = 0;
			printf("\n");
			while (rel_res > err_tol && nv++ < 50) {
				rel_res = grids.v_cycle();
				if (nv % 5 == 1) {
					printf("\r-- rel_res :  %6.2lf%%", rel_res * 100);
				}
			}
			printf("\r-- rel_res :  %6.2lf", rel_res * 100);
			std::vector<double> u[3];
			for (int k = 0; k < 3; k++) {
				u[k].resize(grids[0]->n_nodes());
				gpu_manager_t::download_buf(u[k].data(), grids[0]->getDisplacement()[k], sizeof(double)*grids[0]->n_nodes());
			}

			// download displacement to Z
			if (pZ != nullptr) {
				for (int i = 0; i < grids[0]->n_nodes(); i++) {
					Z(i * 3, i * 3 + j) = u[0][i];
					Z(i * 3 + 1, i * 3 + j) = u[1][i];
					Z(i * 3 + 2, i * 3 + j) = u[2][i];
				}
			}

			// download displacement to C
			for (int k = 0; k < loadnodes.size(); k++) {
				C(k * 3 + 0, i * 3 + j) = u[0][loadnodes[k]];
				C(k * 3 + 1, i * 3 + j) = u[1][loadnodes[k]];
				C(k * 3 + 2, i * 3 + j) = u[2][loadnodes[k]];
			}
		}
		printf("\n");
		lastloadid = loadid;
	}// compute Aggregated system

	eigen2ConnectedMatlab("C", C);
}

Eigen::Matrix<double, -1, 1> assembleForce(const std::vector<int>& nodeids, double* pforce) {
	Eigen::Matrix<double, -1, 1> ftotal;
	ftotal.resize(grids[0]->n_nodes() * 3, 1);
	ftotal.fill(0);
	for (int i = 0; i < nodeids.size(); i++) {
		int nodeid = nodeids[i];
		for (int j = 0; j < 3; j++) {
			ftotal[nodeid * 3 + j] = pforce[i * 3 + j];
		}
	}
	return ftotal;
}

void setForce(Eigen::Matrix<double, -1, 1>& f) {
	std::vector<double> fv[3];
	for (int i = 0; i < 3; i++) {
		fv[i].resize(grids[0]->n_nodes());
	}
	for (int i = 0; i < grids[0]->n_nodes(); i++) {
		fv[0][i] = f[i * 3];
		fv[1][i] = f[i * 3 + 1];
		fv[2][i] = f[i * 3 + 2];
	}
	for (int i = 0; i < 3; i++) {
		gpu_manager_t::upload_buf(grids[0]->getForce()[i], fv[i].data(), sizeof(double) * grids[0]->n_nodes());
	}
}

Eigen::Matrix<double, -1, 1> solveFem(void) {
	Eigen::Matrix<double, -1, 1> u;
	double rel_res = 1;
	int itn = 0;
	while (rel_res > 1e-2 && itn++ < 200) {
		rel_res = grids.v_cycle();
		printf("\r--[%d] rel_res = %6.2lf%%",itn, rel_res*100);
	}
	printf("\n");
	std::vector<double> uh[3];
	for (int i = 0; i < 3; i++) {
		uh[i].resize(grids[0]->n_nodes());
		gpu_manager_t::download_buf(uh[i].data(), grids[0]->getDisplacement()[i], sizeof(double)*grids[0]->n_nodes());
	}
	u.resize(grids[0]->n_nodes() * 3);
	for (int i = 0; i < grids[0]->n_nodes(); i++) {
		u[i * 3] = uh[0][i];
		u[i * 3 + 1] = uh[1][i];
		u[i * 3 + 2] = uh[2][i];
	}
	return u;
}

std::pair<double, Eigen::Matrix<double, -1, 1>> TestSuit::powerMethod(const Eigen::Matrix<double, -1, -1>& C) {
	Eigen::Matrix<double, -1, 1> v(C.rows(), 1);
	v.setRandom();
	v.normalize();
	Eigen::Matrix<double, -1, 1> vlast = v;

	printf("\033[33m[PM]\033[33m\n");

	double eig = 1, lasteig = 0;

	double ech = 1;
	double vch = 1;
	int itn = 0;
	while ((vch > 1e-4 /*|| ech > 1e-4*/) && itn++ < 1e4) {
		lasteig = eig;
		vlast = v;
		v = C * vlast;
		eig = v.norm();
		v /= v.norm();
		ech = abs(eig - lasteig) / eig;
		vch = (vlast - v).norm();
		if (itn % 20 == 0) {
			printf("\r-- [%d] ech %6.2e,  vch %6.2e\n", itn, ech, vch);
		}
	}
	printf("\n");

	return { eig,v };
}

void TestSuit::test2011(void)
{
	initDensities(1);
	
	const std::vector<int>& loadnodes = getLoadNodes();

	Eigen::Matrix<double, -1, -1> C(loadnodes.size() * 3, loadnodes.size() * 3);

	//  
	MMA::mma_t mma(grids[0]->n_gselements, 1);
	mma.init(params.min_rho, 1);
	float volScale = 1e3;
	float sensScale = 1e5;
	gv::gVector dv(grids[0]->n_gselements, volScale / grids[0]->n_gselements);
	gv::gVector v(1, volScale*(1 - params.volume_ratio));

	std::vector<double> clist;

	int itn = 0;
	while (itn++<100) {
		printf("\033[32m* * * * * * * * * * * * * * iter %d * * * * * * * * * * * * * * * * *\033[0m\n", itn);

		// update density from mma
		setDensity(mma.get_x().data());

		grids.log(itn);

		gpu_manager_t::pass_dev_buf_to_matlab("rho", grids[0]->getRho(), grids[0]->n_rho());

		// update grid stencil
		grids.update_stencil();

		// compute volume
		double vol = grids[0]->volumeRatio();
		v[0] = volScale * (vol - params.volume_ratio);

		// solve C
		aggregatedSystem(loadnodes, C);

		// solve largest eigenvector
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig;
		eig.compute((C + C.transpose()) / 2);

		printf("-- solving eigenvalues...\n");
#if 1
		Eigen::Matrix<double, -1, 1> w_worst = eig.eigenvectors().rowwise().reverse().col(0);
		double c_worst = eig.eigenvalues().reverse()[0];
		double c_secworst = eig.eigenvalues().reverse()[1];
		printf("-- c_worst = %6.2e (%6.2e)\n", c_worst, c_secworst);
#else
		auto vv = powerMethod(C);
		double c_worst = vv.first;
		auto& w_worst = vv.second;
		printf("-- c_worst = %6.4e\n", c_worst);
#endif

#if 0
		double c_wmodi = modifiedPM();
		printf("-- c_wmodi = %6.2e\n", c_wmodi);
#endif

		clist.emplace_back(c_worst);

		// assemble worst force 
		Eigen::Matrix<double, -1, 1> f_worst = assembleForce(loadnodes, w_worst.data());

		// solve worst displacement
		setForce(f_worst);
		Eigen::Matrix<double, -1, 1> uworst = solveFem();

		double fu = f_worst.dot(uworst);

		// compute sensitivity
		computeSensitivity();

		scaleVector(grids[0]->getSens(), grids[0]->n_gselements, sensScale);

		gpu_manager_t::pass_dev_buf_to_matlab("sens", grids[0]->getSens(), grids[0]->n_gselements);

		mma.update(grids[0]->getSens(), &dv.data(), v.data());

		// DEBUG
		if (itn % 5 == 0) {
			grids.writeDensity(grids.getPath("out.vdb"));
			grids.writeSensitivity(grids.getPath("sens.vdb"));
		}

		bio::write_vector(grids.getPath("clist"), clist);
	}

	printf("\n=   finished   =\n");

	// write result density field
	grids.writeDensity(grids.getPath("out.vdb"));
}

void TestSuit::testtest2011(void)
{
	initDensities(1);
	
	const std::vector<int>& loadnodes = getLoadNodes();

	Eigen::Matrix<double, -1, -1> C(loadnodes.size() * 3, loadnodes.size() * 3);

	//  
	MMA::mma_t mma(grids[0]->n_gselements, 1);
	mma.init(params.min_rho, 1);
	float volScale = 1e3;
	float sensScale = 1e5;
	gv::gVector dv(grids[0]->n_gselements, volScale / grids[0]->n_gselements);
	gv::gVector v(1, volScale*(1 - params.volume_ratio));

	int itn = 0;
	while (itn++<100) {
		printf("\033[32m* * * * * * * * * * * * * * iter %d * * * * * * * * * * * * * * * * *\033[0m\n", itn);

		// update density from mma
		setDensity(mma.get_x().data());

		grids.log(itn);

		gpu_manager_t::pass_dev_buf_to_matlab("rho", grids[0]->getRho(), grids[0]->n_rho());

		// update grid stencil
		grids.update_stencil();

		// compute volume
		double vol = grids[0]->volumeRatio();
		v[0] = volScale * (vol - params.volume_ratio);

		ModifPM(1, 1, 1, 1, true, false);

		// compute sensitivity
		computeSensitivity();

		scaleVector(grids[0]->getSens(), grids[0]->n_gselements, sensScale);

		gpu_manager_t::pass_dev_buf_to_matlab("sens", grids[0]->getSens(), grids[0]->n_gselements);

		mma.update(grids[0]->getSens(), &dv.data(), v.data());

		// DEBUG
		if (itn % 5 == 0) {
			grids.writeDensity(grids.getPath("out.vdb"));
			grids.writeSensitivity(grids.getPath("sens.vdb"));
		}
	}

	printf("\n=   finished   =\n");

	// write result density field
	grids.writeDensity(grids.getPath("out.vdb"));

}

void TestSuit::testKernels(void)
{
	initDensities(1);
	update_stencil();
	grids.test_kernels();
}

void TestSuit::test2019(void)
{
	initDensities(1);

	const std::vector<int>& loadnodes = getLoadNodes();

	Eigen::Matrix<double, -1, -1> C(loadnodes.size() * 3, loadnodes.size() * 3);
	
	float volScale = 1e-3;
	
	double Vgoal = 1;

	int itn = 0;
	while (itn++ < 100) {
		// update grid stencil
		grids.update_stencil();

		Vgoal *= (1 - params.volume_decrease);

		double Vc = Vgoal - params.volume_ratio;

		if (Vgoal < params.volume_ratio) Vgoal = params.volume_ratio;

		// compute volume
		double vol = grids[0]->volumeRatio();

		// solve C
		aggregatedSystem(loadnodes, C);

		// solve largest eigenvector
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig;
		eig.compute(C);
		Eigen::Matrix<double, -1, 1> w_worst = eig.eigenvectors().col(0);

		// assemble worst force 
		Eigen::Matrix<double, -1, 1> f_worst = assembleForce(loadnodes, w_worst.data());

		// solve worst displacement
		setForce(f_worst);
		if (!grids.hasSupport()) forceProject(grids[0]->getForce());

		// solve FEM
		Eigen::Matrix<double, -1, 1> uworst = solveFem();

		// compute sensitivity
		computeSensitivity();

		// update variable
		updateDensities(Vgoal);
	}

	printf("\n=   finished   =\n");

	// test
	Eigen::Matrix<double, -1, 1> vtest;
	double nrm = vtest.dot(Eigen::Map<Eigen::Matrix<double, -1, 1>>(vtest.data(), vtest.rows()));

	// write result density field
	grids.writeDensity(grids.getPath("out.vdb"));
}

extern void stressAndComplianceOnVertex_impl(double elen, const std::vector<int>& vlexid, const std::vector<glm::vec4>& inclusionPos, std::vector<double>& clist, std::vector<double>& vonlist);

void TestSuit::stressAndComplianceOnVertex(const std::vector<glm::vec4>& p4list, std::vector<double>& clist, std::vector<double>& vonlist)
{
	auto& vsat = grids.vrtsatlist[0];
	auto& esat = grids.elesatlist[0];
	int ereso = grids[0]->_ereso;
	int vreso = grids[0]->_ereso + 1;
	float origin[3] = { grids[0]->_box[0][0], grids[0]->_box[0][1], grids[0]->_box[0][2] };
	float elen = grids[0]->elementLength();

	int invalideid = 0;

	std::vector<int> vlocate(p4list.size(), -1);
	std::vector<glm::vec4> inclusionOffset(p4list.size());

	for (int i = 0; i < p4list.size(); i++) {
		auto p4 = p4list[i];
		int p4pos[3] = { (p4[0] - origin[0]) / elen, (p4[1] - origin[1]) / elen, (p4[2] - origin[2]) / elen };
		int p4id = p4pos[0] + p4pos[1] * ereso + p4pos[2] * ereso * ereso;
		int eid;
		if (p4pos[0] >= ereso || p4pos[0] < 0 || p4pos[1] >= ereso || p4pos[1] < 0 || p4pos[2] >= ereso || p4pos[2] < 0) {
			eid = -1;
		} else {
			eid = esat(p4id);
		}
		if (eid == -1) {
			int newpos[3] = { p4pos[0],p4pos[1],p4pos[2] };
			for (int dx = -1; dx < 2; dx++) {
				newpos[0] = p4pos[0] + dx;
				if (newpos[0] >= ereso || newpos[0] < 0) continue;
				for (int dy = -1; dy < 2; dy++) {
					newpos[1] = p4pos[1] + dy;
					if (newpos[1] >= ereso || newpos[1] < 0) continue;
					for (int dz = -1; dz < 2; dz++) {
						newpos[2] = p4pos[2] + dz;
						if (newpos[2] >= ereso || newpos[2] < 0) continue;
						int newpid = newpos[0] + newpos[1] * ereso + newpos[2] * ereso * ereso;
						eid = esat(newpid);
						if (eid != -1) {
							inclusionOffset[i] = glm::vec4(
								p4[0] - origin[0] - elen * newpos[0],
								p4[1] - origin[1] - elen * newpos[1],
								p4[2] - origin[2] - elen * newpos[2], 0
							);
							p4pos[0] = newpos[0];
							p4pos[1] = newpos[1];
							p4pos[2] = newpos[2];
							goto __validid;
						}
					}
				}
			}
		}

		if (eid == -1) { invalideid++; continue; }

		__validid:
		int vid = p4pos[0] + p4pos[1] * vreso + p4pos[2] * vreso * vreso;
		vlocate[i] = vsat(vid);
		if (vlocate[i] == -1) {
			printf("\033[31mInvalid Vertex on elements\033[0m\n");
			exit(-1);
		}
		for (int j = 0; j < 4; j++) {
			inclusionOffset[i][j] = (std::clamp)(inclusionOffset[i][j], 0.f, elen);
		}
	}

	printf("-- %d vertices are out of range, total %d\n", invalideid, vlocate.size());

	stressAndComplianceOnVertex_impl(elen, vlocate, inclusionOffset, clist, vonlist);
}

