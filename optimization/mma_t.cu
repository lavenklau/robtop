#include "mma_t.h"
#include "string"
#include "vector"
#include"cuda_runtime.h"
#include"cusparse_v2.h"
#include"lib.cuh"
#include"gpuVector.cuh"
#include "algorithm"

namespace MMA {

using namespace gv;


bool MMA::mma_subproblem_t::init(gv::gVector* df, std::vector<gv::gVector*>& dg, gv::gVector* g)
{
	if (!initialized()) {
		p.resize(dg.size() + 1);
		q.resize(dg.size() + 1);
		for (int i = 0; i < p.size(); i++) {
			p[i] = new gv::gVector(df->size());
			q[i] = new gv::gVector(df->size());
		}
		plambda = gVector(df->size());
		qlambda = gVector(df->size());

		Psi = gVector(df->size());

		G.resize(dg.size());
		for (int i = 0; i < G.size(); i++) {
			G[i] = new gVector(df->size());
		}

		//d = gVector(mma.y.size());

		gproxy = gVector(mma.n_constrain());

		b = gVector(mma.n_constrain());

		// initialize description
		auto desc_stat = cusparseCreateMatDescr(&cuSolver.descr);
		if (desc_stat != CUSPARSE_STATUS_SUCCESS) {
			throw std::string("sparse matrix context create failed !");
		}
		cusparseSetMatType(cuSolver.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(cuSolver.descr, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(cuSolver.descr, CUSPARSE_FILL_MODE_LOWER);
		cusparseSetMatDiagType(cuSolver.descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

		// create sparse matrix context
		auto stat = cusolverSpCreate(&cuSolver.spHandle);
		if (stat != CUSOLVER_STATUS_SUCCESS) {
			throw std::string("cusolver context create failed !");
		}

		// allocate gpu memory for row indices
		int n = mma.n_dim();
		int m = mma.n_constrain();
		int n_elements = n + m + 1 + n * m * 2 + m * 4 + m * m;
		cudaMalloc(&cuSolver.row_ptr, sizeof(int)*(n + m + 1 + m + 1));
		cudaMalloc(&cuSolver.col_ptr, sizeof(int)*(n_elements));
		cudaMalloc(&cuSolver.val_ptr, sizeof(gv::Scalar)*n_elements);
		cudaMalloc(&cuSolver.b_ptr, sizeof(gv::Scalar)*(n + m + 1 + m));

		cuSolver.n_nonzeros = n_elements;
		cuSolver.nrows = n + m + 1 + m;
	}

	// compute p , q
	gVector u_x2 = (mma.asym_u - mma.x)*(mma.asym_u - mma.x);
	gVector x_l2 = (mma.x - mma.asym_l)*(mma.x - mma.asym_l);
	//u_x2.toMatlab("u_x2");
	//x_l2.toMatlab("x_l2");
	for (int i = 0; i < dg.size() + 1; i++) {
		//p[i] = new gVector(df->size());
		//q[i] = new gVector(df->size());
		gVector& diff_f = (i == 0 ? *df : *dg[i - 1]);
		//diff_f.toMatlab("diff_f");
		*p[i] = u_x2 * (1.001 * (diff_f.max(0)) + 0.001*(-diff_f).max(0) + 1e-5 / (mma.xmax - mma.xmin));
		*q[i] = x_l2 * (0.001 * (diff_f.max(0)) + 1.001*(-diff_f).max(0) + 1e-5 / (mma.xmax - mma.xmin));
	}

	// compute value of constrain proxy
	for (int i = 0; i < mma.n_constrain(); i++) {
		gproxy[i] = p[i + 1]->dot(1 / (mma.asym_u - mma.x)) + q[i + 1]->dot(1 / (mma.x - mma.asym_l));
	}

	// compute b of proxy 
	b = gproxy - *g;

	// reset auxiliary variable
	reset();

	// compute plambda, qlambda
	std::vector<Scalar> lambda;
	lambda.resize(dg.size());
	mma.lambda.download(lambda.data());
	plambda = *p[0];
	qlambda = *q[0];
	for (int i = 0; i < lambda.size(); i++) {
		plambda += lambda[i] * *p[i + 1];
		qlambda += lambda[i] * *q[i + 1];
	}

	// compute Psi
	Psi = 2 * plambda / ((mma.asym_u - mma.x) ^ 3) + 2 * qlambda / ((mma.x - mma.asym_l) ^ 3);

	// compute G
	for (int i = 0; i < G.size(); i++) {
		*G[i] = (*p[i + 1]) / ((mma.asym_u - mma.x) * (mma.asym_u - mma.x)) - (*q[i + 1]) / ((mma.x - mma.asym_l) * (mma.x - mma.asym_l));
	}

	// compute d
	//std::vector<Scalar> dhost(p.size() - 1);
	//for (int i = 0; i < dhost.size(); i++) {
	//	dhost[i] = (*p[i + 1] / (mma.asym_u - mma.x) + *q[i + 1] / (mma.x - mma.asym_l)).sum();
	//}
	//d = gVector(dhost.size());
	//d.set(dhost.data());
}

MMA::mma_subproblem_t::cuSolver_t::~cuSolver_t()
{
	if (row_ptr != nullptr) {
		cudaFree(row_ptr);
		cudaFree(col_ptr);
		cudaFree(val_ptr);
		cudaFree(b_ptr);
		cusolverSpDestroy(spHandle);
	}
}

void MMA::mma_t::update(
	gv::gVector& new_x, gv::gVector& new_y, Scalar new_z, gv::gVector& new_lambda,
	gv::gVector& new_xi, gv::gVector& new_eta,
	gv::gVector& new_mu, Scalar new_zeta, gv::gVector& new_s
)
{
	x.swap(new_x);
	y.swap(new_y);
	z = (new_z);
	lambda.swap(new_lambda);
	xi.swap(new_xi);
	eta.swap(new_eta);
	mu.swap(new_mu);
	zeta = (new_zeta);
	s.swap(new_s);
}

void mma_t::get_x(Scalar* dst)
{
	cudaMemcpy(dst, x.data(), sizeof(Scalar)*x.size(), cudaMemcpyDeviceToDevice);
}


__device__ Scalar clamp_asym(Scalar val, Scalar low_value, Scalar up_value) {
	if (val > up_value) return up_value;
	if (val < low_value) return low_value;
	return val;
}

// dx0  : change of x this time,  
// dx_1 : change of x last time
void mma_t::adjust_asym(gv::gVector& dx0, gv::gVector& dx_1)
{
	const Scalar* p1 = dx0.data(), *p2 = dx_1.data();
	Scalar* al = asym_l.data(), *au = asym_u.data();
	Scalar* valpha = alpha.data(), *vbeta = beta.data();
	Scalar* xptr = x.data();
	Scalar* dxptr = dx0.data();
	Scalar* xmax_ptr = xmax.data();
	Scalar* xmin_ptr = xmin.data();

	// adjust asymptotes
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, dx0.size(), 512);
	traverse_noret << <grid_size, block_size >> > (dx0.size(), [=] __device__(int eid) {
		Scalar gamma = 1;
		Scalar d1 = p1[eid], d2 = p2[eid];
		if (d1*d2 > 0) gamma = 1.2;
		if (d1*d2 < 0) gamma = 0.7;
		Scalar alvalue = al[eid], auvalue = au[eid];
		Scalar xval = xptr[eid];
		Scalar xmax_val = xmax_ptr[eid];
		Scalar xmin_val = xmin_ptr[eid];
		Scalar dxval = dxptr[eid];
		// xval is already updated
		Scalar new_al = xval /*+ dxval */ - gamma * (xval - dxval - alvalue);
		Scalar new_au = xval/* + dxval */ + gamma * (auvalue - xval + dxval);
		Scalar val_alpha = valpha[eid], val_beta = vbeta[eid];

		Scalar lower, upper;

		// clamp lower asymptotes
		lower = xval - 10 * (xmax_val - xmin_val);
		upper = xval - 0.01*(xmax_val - xmin_val);
		new_al = clamp_asym(new_al, lower, upper);
		// clamp upper asymptotes
		lower = xval + 0.01 * (xmax_val - xmin_val);
		upper = xval + 10*(xmax_val - xmin_val);
		new_au = clamp_asym(new_au, lower, upper);

		// adjust asymptotes
		al[eid] = new_al;
		au[eid] = new_au;

		// adjust local bounds
		Scalar newalpha = xmin_val;
		Scalar newalpha_low = new_al + 0.1f*(xval - new_al);
		if (newalpha < newalpha_low) newalpha = newalpha_low;
		newalpha_low = xval - 0.5*(xmax_val - xmin_val);
		if (newalpha < newalpha_low) newalpha = newalpha_low;
		
		Scalar newbeta = xmax_val;
		Scalar newbeta_up = new_au - 0.1f*(new_au - xval);
		if (newbeta > newbeta_up) newbeta = newbeta_up;
		newbeta_up = xval + 0.5*(xmax_val - xmin_val);
		if (newbeta > newbeta_up) newbeta = newbeta_up;

		valpha[eid] = newalpha;
		vbeta[eid] = newbeta;
	});
	cudaDeviceSynchronize();
	cuda_error_check;

}

gv::gVector MMA::mma_subproblem_t::solveLinearSystem(
	gv::gVector& Dx, gv::gVector& Dy, std::vector<gv::gVector*>& Gvecs,
	gv::Scalar zetadz, gv::gVector& a, 
	gv::gVector& Dlambda, gv::gVector& deltax, gv::gVector& deltay, gv::Scalar deltaz, gv::gVector& deltaLambda)
{
	std::vector<Eigen::Triplet<gv::Scalar>> triplist;

	gVector::toMatlab("Gvecs", Gvecs);
	Dx.toMatlab("Dx");
	Dy.toMatlab("Dy");
	a.toMatlab("a");
	Dlambda.toMatlab("Dlambda");

	int ndim = mma.n_dim();
	int nconstrain = mma.n_constrain();

	gVector Dlambda_y = Dlambda + 1 / Dy;

	gVector hatdelta_lambda_y = deltaLambda + deltay / Dy;

	hatdelta_lambda_y.toMatlab("hatdelta_lambda_y");
	Dlambda_y.toMatlab("Dlambda_y");

	std::vector<std::vector<Scalar>> A11;
	std::vector<Scalar> b1(nconstrain);
	std::vector<Scalar> A12(nconstrain);
	Scalar A22 = -zetadz;
	Scalar b2 = deltaz;

	a.download(A12.data());

	A11.resize(nconstrain);
	for (int i = 0; i < A11.size(); i++) {
		A11[i].resize(nconstrain, 0);
	}

	std::vector<Scalar> hostDlambday(nconstrain);
	std::vector<Scalar> hostdelta_lambda_y(nconstrain);
	Dlambda_y.download(hostDlambday.data());
	hatdelta_lambda_y.download(hostdelta_lambda_y.data());
	
	for (int i = 0; i < nconstrain; i++) {
		A11[i][i] += hostDlambday[i];
	}

	Scalar g;
	for (int i = 0; i < nconstrain; i++) {
		for (int j = 0; j < nconstrain; j++) {
			g = (*Gvecs[i]).dot((*Gvecs[j]) / Dx);
			A11[i][j] += g;
		}
	}

	//std::cout << "hostdelta_lambda_y = " << std::endl;
	for (int i = 0; i < nconstrain; i++) {
		g = (*Gvecs[i]).dot(deltax / Dx);
		//std::cout << hostdelta_lambda_y[i] << std::endl;
		b1[i] = hostdelta_lambda_y[i] - g;
	}

	
	std::vector<Scalar> dlambda_dz;

	largeVarlessConstrainLinearSolve(A11, A12, A22, b1, b2, dlambda_dz);

	gVector dx = -deltax / Dx;

	Scalar dz = dlambda_dz[nconstrain];

	for (int i = 0; i < nconstrain; i++) {
		dx -= dlambda_dz[i] * (*Gvecs[i]) / Dx;
	}

	gVector dlambda(nconstrain);
	dlambda.set(dlambda_dz.data());

	gVector dy = dlambda / Dy - deltay / Dy;

	gv::gVector dxi = mma.epsilon / (mma.x - mma.alpha) - dx / (mma.x - mma.alpha)*mma.xi - mma.xi;
	gv::gVector deta = mma.epsilon / (mma.beta - mma.x) + mma.eta / (mma.beta - mma.x) * dx - mma.eta;
	gv::gVector dmu = mma.epsilon / mma.y - dy * mma.mu / mma.y - mma.mu;
	Scalar      dzeta = mma.epsilon / mma.z - mma.zeta - (mma.zeta / mma.z) * dz;
	gv::gVector ds = mma.epsilon / mma.lambda - mma.s / mma.lambda*dlambda - mma.s;

	gVector dw(mma.n_w());
	
	mma.get_w(dx, dy, dz, dlambda, dxi, deta, dmu, dzeta, ds, dw);
	
	dx.toMatlab("dx");
	dy.toMatlab("dy");
	std::cout << "dz = " << dz << std::endl;
	dlambda.toMatlab("dlambda");
	dxi.toMatlab("dxi");
	deta.toMatlab("deta");
	dmu.toMatlab("dmu");
	std::cout << "dzeta = " << dzeta << std::endl;
	ds.toMatlab("ds");


	return dw;
}

std::pair<gv::Scalar, gv::Scalar> MMA::mma_subproblem_t::solve(gv::gVector* df, std::vector<gv::gVector*>& dg, gv::gVector* g, gv::gVector& dxResult)
{
	gv::gVector curx = mma.x;

	init(df, dg, g);

	Scalar err = 1e30, lasterr = 0;

	std::pair<Scalar, Scalar> err_step;

	gv::gVector lastx, Dx, Dy, Dlambda;
	gv::gVector varphi_x;
	gv::gVector hat_deltax, hat_deltay, hat_deltalambda;
	Scalar hat_deltaz;
	gv::gVector dw;
	gv::gVector full_dw(mma.n_w());
	gv::gVector w(mma.n_w());

	int nSubIter = 0;

	bool overflowed = false;
	//mma.toMatlab();

	bool err_stays = false;

	int err_stays_counter = 0;

	Scalar lambda_damp = 1;

	while (mma.epsilon > 1e-6) {

		err = 1e30;

		lambda_damp = 1;

		err_stays = false;

		err_stays_counter = 0;

		while (err > 0.9*mma.epsilon) {
		
			mma.get_w(w);

			lastx = mma.x;

			Psi = 2 * plambda / ((mma.asym_u - mma.x) ^ 3) + 2 * qlambda / ((mma.x - mma.asym_l) ^ 3);

			Dx = Psi + mma.xi / (mma.x - mma.alpha) + mma.eta / (mma.beta - mma.x);

			Dy = mma.d + mma.mu / mma.y;

			Dlambda = mma.s / mma.lambda;

			for (int i = 0; i < G.size(); i++) {
				*G[i] = (*p[i + 1]) / ((mma.asym_u - mma.x) * (mma.asym_u - mma.x)) - (*q[i + 1]) / ((mma.x - mma.asym_l) * (mma.x - mma.asym_l));
			}

			for (int i = 0; i < mma.n_constrain(); i++) {
				gproxy[i] = p[i + 1]->dot(1 / (mma.asym_u - mma.x)) + q[i + 1]->dot(1 / (mma.x - mma.asym_l));
			}

			varphi_x = plambda / ((mma.asym_u - mma.x)*(mma.asym_u - mma.x)) - qlambda / ((mma.x - mma.asym_l)*(mma.x - mma.asym_l));
			hat_deltax = -(mma.epsilon / (mma.x - mma.alpha) - mma.epsilon / (mma.beta - mma.x) - varphi_x);
			hat_deltay = -(mma.lambda + mma.epsilon / mma.y - mma.c - mma.d * mma.y);
			Scalar hat_deltaz = -(mma.lambda.dot(mma.a) + mma.epsilon / mma.z - mma.a0);

			hat_deltalambda = -(mma.z*mma.a + mma.y + b - mma.epsilon / mma.lambda - gproxy);

			dw = solveLinearSystem(Dx, Dy, G, mma.zeta / mma.z, mma.a, Dlambda, hat_deltax, hat_deltay, hat_deltaz, hat_deltalambda);

			//whole_dw(dw, full_dw);

#ifdef __MMA_WITH_MATLAB
			//hat_deltax.toMatlab("deltax");
			//hat_deltay.toMatlab("deltay");
			//printf("hat_deltaz = %f\n", hat_deltaz);
			//hat_deltalambda.toMatlab("deltalambda");
			//dw.toMatlab("simp_dw");
			//dw.toMatlab("dw");
			//w.toMatlab("w");
#endif

			Scalar max_t1inv = ((-1 / asym_clamp_factor) * (get_dx(dw) / (mma.x - mma.alpha))).max();
			Scalar max_t2inv = ((1 / asym_clamp_factor) * (get_dx(dw) / (mma.beta - mma.x))).max();
			Scalar max_t3inv = ((-1 / asym_clamp_factor)*get_artivar(dw) / get_artivar(w)).max();
			printf("max_tinv = (%f, %f, %f)\n", max_t1inv, max_t2inv, max_t3inv);
			Scalar max_t = 1 / (std::max)((std::max)(max_t1inv, max_t2inv), (std::max)(max_t3inv, Scalar{ 1 }));

#ifdef __MMA_WITH_MATLAB
			//((mma.alpha - mma.x) / get_dx(dw)).toMatlab("t1b");
			//((mma.beta - mma.x) / get_dx(dw)).toMatlab("t2b");
#endif

			printf("\033[32msearching step ... initial step = %f,\033[0m\033[34m epsilon = %f\033[0m\n", max_t, mma.epsilon);
			std::cout << "==========================================================================================" << std::endl;

			std::cout << "\033[31mrelambda damping " << lambda_damp << "\033[0m" << std::endl;

			err_step = search_step(dw, max_t, lambda_damp);

			err = err_step.first;

			std::cout << "\033[31merr = " << err << "\033[0m" << ", lasterr = " << lasterr << std::endl;

			if ((err - lasterr) / err < 1e-5) {
				err_stays_counter++;
			}
			else {
				err_stays_counter = 0;
			}

			lasterr = err;

			if (err_stays_counter > 6) {
				err_stays = true;
				err_stays_counter = 0;
			}

			if (err_stays) {
				lambda_damp *= 10;
				err_stays = false;
			}

			printf("\033[32musing step = %f\033[0m", err_step.second, mma.epsilon);

			if (nSubIter++ > 80) {
				printf("\033[31mIteration on subproblem is overflowed !\n\033[0m");
				printf("\033[31m   -]epsilon = %f\n\033[0m", mma.epsilon);
				printf("\033[31m   -]error   = %f\n\033[0m", err);
				printf("\033[31m   -]step    = %f\n\033[0m", err_step.second);

				printf("\033[31m   -]asym_l  = (%f", Scalar(mma.asym_l.min()));
				//for (int i = 1; i < mma.asym_l.size(); i++) {
					printf(", %f", Scalar(mma.asym_l.max()));
				//}
				printf(")\n\033[0m");

				printf("\033[31m   -]asym_u  = (%f", Scalar(mma.asym_u.min()));
				//for (int i = 1; i < mma.asym_u.size(); i++) {
					printf(", %f", Scalar(mma.asym_u.max()));
				//}
				printf(")\n\033[0m");

				overflowed = true;
				break;
			}
		}

		if (overflowed) {
			break;
		}

		mma.epsilon *= 0.1;
	}

	gv::gVector cur_dx = mma.x - curx;

	mma.adjust_asym(cur_dx, mma.lastdx);

	//cur_dx.toMatlab("cur_dx");
	//mma.xmax.toMatlab("xmax");
	//mma.xmin.toMatlab("xmin");
	//mma.lastdx.toMatlab("lastdx");
	//mma.asym_l.toMatlab("new_L");
	//mma.asym_u.toMatlab("new_u");

	dxResult = cur_dx;

	return err_step;
}

std::pair<Scalar, Scalar> MMA::mma_subproblem_t::search_step(gv::gVector& dw, Scalar initial_step, Scalar lambda_damp)
{
	std::cout << "backtracking line searching..." << std::endl;
	dx = get_dx(dw);
	dy = get_dy(dw);
	dz = get_dz(dw);
	dlambda = get_dlambda(dw);

	dxi = mma.epsilon / (mma.x - mma.alpha) - dx / (mma.x - mma.alpha)*mma.xi - mma.xi;
	deta = mma.epsilon / (mma.beta - mma.x) + mma.eta / (mma.beta - mma.x) * dx - mma.eta;
	dmu = mma.epsilon / mma.y - dy * mma.mu / mma.y - mma.mu;
	dzeta = mma.epsilon / mma.z - mma.zeta - (mma.zeta / mma.z) * dz;
	ds = mma.epsilon / mma.lambda - mma.s / mma.lambda*dlambda - mma.s;

	Scalar step = initial_step;

#ifdef __MMA_WITH_MATLAB
	dx.toMatlab("ls_dx");
	dy.toMatlab("ls_dy");
	std::cout << "ls_dz = " << dz << std::endl;
	dlambda.toMatlab("ls_dlambda");
	dxi.toMatlab("ls_dxi");
	deta.toMatlab("ls_deta");
	dmu.toMatlab("ls_dmu");
	std::cout << "ls_dzeta = " << dzeta << std::endl;
	ds.toMatlab("ls_ds");
#endif

	auto two_inf_err = kkt_err(mma.x, mma.y, mma.z, mma.lambda, mma.xi, mma.eta, mma.mu, mma.zeta, mma.s);

	std::cout << "old : ||delta||_2 = " << two_inf_err.first << ", " << "||delta||_inf = " << two_inf_err.second << std::endl;

	Scalar old_err = two_inf_err.first;

	Scalar new_err = 1e30;
	Scalar new_inf_err = 1e30;

	while (step > 1e-12 && new_err > old_err) {
		std::cout << "trying step " << step << std::endl;
		std::cout << "----------------------" << std::endl;

		new_x = mma.x + dx * step;
		new_y = mma.y + dy * step;
		new_z = mma.z + dz * step;
		new_lambda = mma.lambda + dlambda * step;
		new_xi = mma.xi + dxi * step;
		new_eta = mma.eta + deta * step;
		new_mu = mma.mu + dmu * step;
		new_zeta = mma.zeta + dzeta * step;
		new_s = mma.s + ds * step;

		auto new_two_inf_err = kkt_err(new_x, new_y, new_z, new_lambda, new_xi, new_eta, new_mu, new_zeta, new_s, lambda_damp);

		std::cout << "new : ||delta||_2 = " << new_two_inf_err.first << ", " << "||delta||_inf = " << new_two_inf_err.second << std::endl;


		new_err = new_two_inf_err.first;
		new_inf_err = new_two_inf_err.second;
		if (new_err < old_err) {
			// update variable
			mma.update(new_x, new_y, new_z, new_lambda, new_xi, new_eta, new_mu, new_zeta, new_s);

			break;
		}
		step /= 2;

		std::cout << std::endl;
	}
	
	return std::pair<Scalar, Scalar>(new_inf_err, step);
}

// return <2-norm, inf-norm> of delta
std::pair<Scalar, Scalar> MMA::mma_subproblem_t::kkt_err(
	gv::gVector& new_x, gv::gVector& new_y, Scalar new_z, gv::gVector& new_lambda,
	gv::gVector& new_xi, gv::gVector& new_eta, gv::gVector& new_mu,
	Scalar new_zeta, gv::gVector& new_s,
	Scalar lambda_damp
)
{
	Scalar err_sum = 0, err_max = -10;

	update_pqlambda(new_lambda);

	kkt_err_tmp.resize(mma.n_dim());
	int tag = 0;

	auto rex = gv::gVectorMap(kkt_err_tmp.data(), mma.n_dim());
	rex = plambda / ((mma.asym_u - new_x)*(mma.asym_u - new_x)) - qlambda / ((new_x - mma.asym_l)*(new_x - mma.asym_l)) - new_xi + new_eta;
	rex.toMatlab("rex");
	err_sum += rex.sqrnorm();
	err_max = (std::max)(err_max, rex.infnorm());

	auto rey = gv::gVectorMap(kkt_err_tmp.data(), mma.n_constrain());
	rey = mma.c + mma.d * new_y - new_lambda - new_mu;
	rey.toMatlab("rey");
	err_sum += rey.sqrnorm();
	err_max = (std::max)(err_max, rey.infnorm());

	err_sum += pow(mma.a0 - new_zeta - new_lambda.dot(mma.a), 2);
	std::cout << "rez = " << mma.a0 - new_zeta - new_lambda.dot(mma.a) << std::endl;
	err_max = (std::max)(err_max, abs(mma.a0 - new_zeta - new_lambda.dot(mma.a)));

	gv::gVector g(mma.n_constrain());
	for (int i = 0; i < g.size(); i++) {
		g[i] = ((*p[i + 1]) / (mma.asym_u - new_x) + (*q[i + 1]) / (new_x - mma.asym_l)).sum();
	}

	auto relam = gv::gVectorMap(kkt_err_tmp.data(), mma.n_constrain());
	//tag += mma.n_constrain();
	relam = g - mma.a*new_z - new_y + new_s - b;
	relam.toMatlab("relam");
	err_sum += relam.sqrnorm() / (lambda_damp*lambda_damp);
	err_max = (std::max)(err_max, relam.infnorm() / lambda_damp);

	auto rexsi = gv::gVectorMap(kkt_err_tmp.data(), mma.n_dim());
	//tag += mma.n_dim();
	rexsi = new_xi * (new_x - mma.alpha) - mma.epsilon;
	rexsi.toMatlab("rexsi");
	err_sum += rexsi.sqrnorm();
	err_max = (std::max)(err_max, rexsi.infnorm());

	auto reeta = gv::gVectorMap(kkt_err_tmp.data(), mma.n_dim());
	reeta = new_eta * (mma.beta - new_x) - mma.epsilon;
	reeta.toMatlab("reeta");
	err_sum += reeta.sqrnorm();
	err_max = (std::max)(err_max, reeta.infnorm());

	auto remu = gv::gVectorMap(kkt_err_tmp.data(), mma.n_constrain());
	remu = new_mu * new_y - mma.epsilon;
	remu.toMatlab("remu");
	err_sum += remu.sqrnorm();
	err_max = (std::max)(err_max, remu.infnorm());

	err_sum += pow(new_zeta*new_z - mma.epsilon, 2);
	std::cout << "rezet = " << new_zeta * new_z - mma.epsilon << std::endl;
	err_max = (std::max)(err_max, abs(new_zeta*new_z - mma.epsilon));

	auto res = gv::gVectorMap(kkt_err_tmp.data(), mma.n_constrain());
	res = new_lambda * new_s - mma.epsilon;
	res.toMatlab("res");
	err_sum += res.sqrnorm();
	err_max = (std::max)(err_max, res.infnorm());

#ifdef __MMA_WITH_MATLAB
	//g.toMatlab("g");
	//b.toMatlab("b");
	//new_x.toMatlab("new_x");
	//new_y.toMatlab("new_y");
	//gv::gVector(1, new_z).toMatlab("new_z");
	//new_lambda.toMatlab("new_lambda");
	//new_xi.toMatlab("new_xi");
	//new_eta.toMatlab("new_eta");
	//new_mu.toMatlab("new_mu");
	//gv::gVector(1, new_zeta).toMatlab("new_zeta");
	//new_s.toMatlab("new_s");

	//gv::gVector total_delta = plambda / ((mma.asym_u - new_x)*(mma.asym_u - new_x)) - qlambda / ((new_x - mma.asym_l)*(new_x - mma.asym_l)) - new_xi + new_eta;
	//total_delta.concate(mma.c + mma.d * new_y - new_lambda - new_mu);
	//total_delta.concate(mma.a0 - new_zeta - new_lambda.dot(mma.a));
	//total_delta.concate(g - mma.a*new_z - new_y + new_s - b);
	//total_delta.concate(new_xi * (new_x - mma.alpha) - mma.epsilon);
	//total_delta.concate(new_eta*(mma.beta - new_x) - mma.epsilon);
	//total_delta.concate(new_mu*new_y - mma.epsilon);
	//total_delta.concate(new_zeta*new_z - mma.epsilon);
	//total_delta.concate(new_lambda*new_s - mma.epsilon);
	//total_delta.toMatlab("deltaAll");
#endif

	return std::pair<Scalar, Scalar>(sqrt(err_sum), err_max);
}

void MMA::mma_subproblem_t::cuSolver_t::update(int* host_row_ptr, int* host_col_ptr, gv::Scalar* host_val_ptr, gv::Scalar* host_b_ptr)
{
	cudaMemcpy(row_ptr, host_row_ptr, sizeof(int)*n_row_indices(), cudaMemcpyHostToDevice);
	cudaMemcpy(col_ptr, host_col_ptr, sizeof(int)*n_elements(), cudaMemcpyHostToDevice);
	cudaMemcpy(val_ptr, host_val_ptr, sizeof(gv::Scalar)*n_elements(), cudaMemcpyHostToDevice);
	cudaMemcpy(b_ptr, host_b_ptr, sizeof(gv::Scalar)*nrows, cudaMemcpyHostToDevice);
}


void MMA::mma_subproblem_t::update_pqlambda(gv::gVector& newLambda)
{
	std::vector<Scalar> lambda;
	lambda.resize(mma.n_constrain());
	newLambda.download(lambda.data());
	plambda = *p[0];
	qlambda = *q[0];
	for (int i = 0; i < lambda.size(); i++) {
		plambda += lambda[i] * *p[i + 1];
		qlambda += lambda[i] * *q[i + 1];
	}
}

void MMA::mma_subproblem_t::reset(void)
{
	mma.x = (mma.alpha + mma.beta) / 2;
	mma.epsilon = 1;
	mma.zeta = mma.z = 1;
	mma.y.set(1);
	mma.lambda.set(1);
	mma.s.set(1);
	mma.xi = (1 / (mma.x - mma.alpha)).max(1);
	mma.eta = (1 / (mma.beta - mma.x)).max(1);
	mma.mu = (mma.c / 2).max(1);
}


void MMA::mma_subproblem_t::whole_dw(gv::gVector& simple_dw, gv::gVector& dw)
{
	dw.resize(mma.n_w());
	Scalar* ptr = simple_dw.data();
	gv::gVectorMap xmap(ptr, mma.x.size());
	ptr += mma.x.size();
	gv::gVectorMap ymap(ptr, mma.y.size());
	ptr += mma.y.size();
	ptr += 1;
	gv::gVectorMap lambdamap(ptr, mma.lambda.size());

	//int offset = 0;
	//gv::gVectorMap(dw.data() + offset, mma.x.size()) = gv::gVector(xmap);
	//offset += mma.x.size();
	//gv::gVectorMap(dw.data() + offset, mma.y.size()) = gv::gVector(ymap);
	gv::gVector dx(xmap);
	gv::gVector dy(ymap);
	Scalar dz = simple_dw[mma.x.size() + mma.y.size()];
	gv::gVector dlambda(lambdamap);

	gv::gVector dxi = mma.epsilon / (mma.x - mma.alpha) - dx / (mma.x - mma.alpha)*mma.xi - mma.xi;
	gv::gVector deta = mma.epsilon / (mma.beta - mma.x) + mma.eta / (mma.beta - mma.x) * dx - mma.eta;
	gv::gVector dmu = mma.epsilon / mma.y - dy * mma.mu / mma.y - mma.mu;
	Scalar      dzeta = mma.epsilon / mma.z - mma.zeta - (mma.zeta / mma.z) * dz;
	gv::gVector ds = mma.epsilon / mma.lambda - mma.s / mma.lambda*dlambda - mma.s;

	mma.get_w(dx, dy, dz, dlambda, dxi, deta, dmu, dzeta, ds, dw);

	return;
}

void mma_t::clamp_asymptotes(void)
{
	gVector stride = xmax - xmin;
	gv::gVector lower = x - 10 * stride;
	gv::gVector upper = x - 0.01*stride;
	asym_l.clamp(lower, upper);

	lower = x + 0.01 * stride;
	upper = x + 10 * stride;
	asym_u.clamp(lower, upper);

	// adjust bounds
	alpha = xmin.max((asym_l + 0.1*(x - asym_l)).max(x - 0.5*stride));
	beta = xmax.min((asym_u - 0.1*(asym_u - x)).min(x + 0.5*stride));
}

void MMA::mma_subproblem_t::cuSolver_t::solve(gv::gVector& result)
{
	result = gv::gVector(nrows);
	int singular_flag;
	auto solve_stat = cusolverSpScsrlsvqr(spHandle, nrows, n_elements(), descr, val_ptr, row_ptr, col_ptr, b_ptr, 1e-7, 0, result.data(), &singular_flag);
	if (solve_stat != CUSOLVER_STATUS_SUCCESS) {
		printf("\033[31m cusolver failed, error code  %d \033[0m\n", solve_stat);
	}
}


gv::gVector MMA::mma_subproblem_t::get_dx(const gv::gVector& dw)
{
	return dw.slice(0, mma.n_dim());
}

gv::gVector MMA::mma_subproblem_t::get_dy(gv::gVector& dw)
{
	return dw.slice(mma.n_dim(), mma.n_dim() + mma.n_constrain());
}

Scalar MMA::mma_subproblem_t::get_dz(gv::gVector& dw)
{
	int start_id = mma.n_dim() + mma.n_constrain();
	return dw[start_id];
}

gv::gVector MMA::mma_subproblem_t::get_dlambda(gv::gVector& dw)
{
	int start_id = mma.n_dim() + mma.n_constrain() + 1;
	int end_id = start_id + mma.n_constrain();
	return dw.slice(start_id, end_id);
}

void mma_t::init_subproblem_variable(void) {
	x = (alpha + beta) / 2;
	xi = x - alpha;
	xi.invInPlace();
	xi.maximize(1);
	eta = beta - x;
	eta.invInPlace();
	eta.maximize(1);
	mu = c / 2;
	mu.maximize(1);
}

void mma_t::set_bound(Scalar* lower_bound, Scalar* upper_bound)
{
	xmin.set(lower_bound);
	xmax.set(upper_bound);
	alpha.set(lower_bound);
	beta.set(upper_bound);
	asym_l = alpha;
	asym_u = beta;
	x = (alpha + beta) / 2;
	clamp_asymptotes();
}

void mma_t::set_bound(Scalar uniform_lower_bound, Scalar uniform_upper_bound)
{
	xmin.set(uniform_lower_bound);
	xmax.set(uniform_upper_bound);
	alpha.set(uniform_lower_bound);
	beta.set(uniform_upper_bound);
	asym_l = alpha;
	asym_u = beta;
	x = (alpha + beta) / 2;
	clamp_asymptotes();
}

void mma_t::get_w(gv::gVector& w)
{
	w.resize(n_w());

	Scalar* ptr = w.data();
	int offset = 0;
	cudaMemcpy(ptr + offset, x.data(), sizeof(Scalar)*x.size(), cudaMemcpyDeviceToDevice);
	offset += x.size();

	cudaMemcpy(ptr + offset, y.data(), sizeof(Scalar)*y.size(), cudaMemcpyDeviceToDevice);
	offset += y.size();

	w[offset] = z;
	offset += 1;

	cudaMemcpy(ptr + offset, lambda.data(), sizeof(Scalar)*lambda.size(), cudaMemcpyDeviceToDevice);
	offset += lambda.size();

	cudaMemcpy(ptr + offset, xi.data(), sizeof(Scalar)*xi.size(), cudaMemcpyDeviceToDevice);
	offset += xi.size();

	cudaMemcpy(ptr + offset, eta.data(), sizeof(Scalar)*eta.size(), cudaMemcpyDeviceToDevice);
	offset += eta.size();

	cudaMemcpy(ptr + offset, mu.data(), sizeof(Scalar)*mu.size(), cudaMemcpyDeviceToDevice);
	offset += mu.size();

	w[offset] = zeta;
	offset += 1;

	cudaMemcpy(ptr + offset, s.data(), sizeof(Scalar)*s.size(), cudaMemcpyDeviceToDevice);

	return;
}

void mma_t::get_w(
	gv::gVector& new_x, gv::gVector& new_y, Scalar new_z, gv::gVector& new_lambda,
	gv::gVector& new_xi, gv::gVector& new_eta,
	gv::gVector& new_mu,
	Scalar new_zeta, 
	gv::gVector& new_s,
	gv::gVector& new_w)
{
	new_w.resize(n_w());

	Scalar* ptr = new_w.data();
	int offset = 0;
	cudaMemcpy(ptr + offset, new_x.data(), sizeof(Scalar)*new_x.size(), cudaMemcpyDeviceToDevice);
	offset += new_x.size();

	cudaMemcpy(ptr + offset, new_y.data(), sizeof(Scalar)*new_y.size(), cudaMemcpyDeviceToDevice);
	offset += new_y.size();

	new_w[offset] = new_z;
	offset += 1;

	cudaMemcpy(ptr + offset, new_lambda.data(), sizeof(Scalar)*new_lambda.size(), cudaMemcpyDeviceToDevice);
	offset += new_lambda.size();

	cudaMemcpy(ptr + offset, new_xi.data(), sizeof(Scalar)*new_xi.size(), cudaMemcpyDeviceToDevice);
	offset += new_xi.size();

	cudaMemcpy(ptr + offset, new_eta.data(), sizeof(Scalar)*new_eta.size(), cudaMemcpyDeviceToDevice);
	offset += new_eta.size();

	cudaMemcpy(ptr + offset, new_mu.data(), sizeof(Scalar)*new_mu.size(), cudaMemcpyDeviceToDevice);
	offset += new_mu.size();

	new_w[offset] = new_zeta;
	offset += 1;

	cudaMemcpy(ptr + offset, new_s.data(), sizeof(Scalar)*new_s.size(), cudaMemcpyDeviceToDevice);

	return;
}

};


