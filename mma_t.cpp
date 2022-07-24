#include "mma_t.h"
#include "Eigen/Sparse"
#include "Eigen/Eigen"
//#include "mycommon.h"
#include "functional"
#ifdef __MMA_WITH_MATLAB
#include "matlab_utils.h"
#endif

using namespace MMA;
using namespace gv;

gv::gVectorMap MMA::mma_subproblem_t::get_artivar(gv::gVector& w) const
{
	return gVectorMap(w.data() + mma.n_dim(), w.size() - mma.n_dim());
}

bool mma_t::update(Scalar* dev_df, Scalar** dg, Scalar* dev_g)
{
	toMatlab();

	gv::gVector vdf = gVector::Map(dev_df, n_dim());
	std::vector<gVector> vdg(n_constrain());
	std::vector<gVector*> vdg_ptr(n_constrain());
	for (int i = 0; i < n_constrain(); i++) {
		vdg[i] = (gVector::Map(dg[i], n_dim()));
		vdg_ptr[i] = &vdg[i];
	}

	gv::gVector vg = gVector::Map(dev_g, n_constrain());

	vdf.toMatlab("df");
	//gv::gVector::toMatlab("dg", dg, n_dim());
	vg.toMatlab("g");

	auto err_step = subproblem.solve(&vdf, vdg_ptr, &vg, dx);

	subproblem.toMatlab();

	dx.swap(lastdx);

	if (lastdx.norm() < 1e-3) {
		stop_counter++;
	}

	if (stop_counter > 10) {
		return true;
	}

	return false;
}


bool MMA::mma_subproblem_t::initialized(void)
{
	return !q.empty();
}

void MMA::mma_subproblem_t::clear()
{
	for (int i = 0; i < p.size(); i++) {
		p[i]->clear();
	}

	for (int i = 0; i < q.size(); i++) {
		q[i]->clear();
	}

	plambda.clear();
	qlambda.clear();

	for (int i = 0; i < G.size(); i++) {
		G[i]->clear();
	}
}

MMA::mma_subproblem_t::~mma_subproblem_t()
{
	clear();
}


void MMA::mma_subproblem_t::largeVarlessConstrainLinearSolve(std::vector<std::vector<Scalar>>& A11, std::vector<Scalar>& A12, Scalar A13, std::vector<Scalar>& b1, Scalar b2, std::vector<Scalar>& result)
{
	int nc = mma.n_constrain();

	Eigen::Matrix<Scalar, -1, -1> A(nc + 1, nc + 1);
	Eigen::Matrix<Scalar, -1, -1> b(nc + 1, 1);

	for (int i = 0; i < A11.size(); i++) {
		for (int j = 0; j < A11.size(); j++) {
			A(i, j) = A11[i][j];
		}
	}

	for (int i = 0; i < nc; i++) {
		A(i, nc) = A12[i];
		A(nc, i) = A12[i];
	}

	A(nc, nc) = A13;

	memcpy(b.data(), b1.data(), b1.size() * sizeof(Scalar));

	b(nc, 0) = b2;

	Eigen::Matrix<Scalar, -1, -1> x = A.colPivHouseholderQr().solve(b);

	result.resize(x.size());

#ifdef __MMA_WITH_MATLAB
	eigen2ConnectedMatlab("A", A);
	eigen2ConnectedMatlab("b", b);
	eigen2ConnectedMatlab("x", x);
#endif

	memcpy(result.data(), x.data(), result.size() * sizeof(Scalar));
}

void MMA::mma_subproblem_t::toMatlab(void)
{
#if defined(__MMA_WITH_MATLAB) &&  defined(ENABLE_MATLAB)
	// pass p,q to matlab
	Eigen::Matrix<Scalar, -1, -1> pmat(p[0]->size(), p.size()), qmat(p[0]->size(), p.size());
	for (int i = 0; i < p.size(); i++) {
		p[i]->download(pmat.col(i).data());
		q[i]->download(qmat.col(i).data());
	}
	eigen2ConnectedMatlab("p", pmat);
	eigen2ConnectedMatlab("q", qmat);
	
	// pass p/qlambda to matlab
	plambda.toMatlab("plambda");
	qlambda.toMatlab("qlambda");

	// pass Psi to matlab
	Psi.toMatlab("Psi");

	// pass G to matlab
	Eigen::Matrix<Scalar, -1, -1> Gmat(G[0]->size(), G.size());
	for (int i = 0; i < G.size(); i++) {
		G[i]->download(Gmat.col(i).data());
	}
	eigen2ConnectedMatlab("G", Gmat);

	mma.d.toMatlab("d");

	gproxy.toMatlab("gproxy");

	b.toMatlab("b");
#endif
}

namespace MMA {

	//void gen_constrain_func(int num_func, std::vector<std::function<Scalar(std::vector<Scalar>&)>>& funcs) {
	//	for (int i = 0; i < num_func; i++)
	//	{
	//		funcs.push_back([=](std::vector<Scalar>& input) {
	//			Scalar result;
	//			return result;
	//		});
	//	}
	//		
	//}

	void test_mma(void) {
		// test 1-dim
		if (0)
		{
			auto f = [](Scalar x) {
				return x * exp(-x * x) + sin(x);
			};
			auto df = [](Scalar x) {
				return exp(-x * x) * (1 - 2 * x * x) + cos(x);
			};
			auto g = [](Scalar x) {
				return (x - 1.5f)*(x - 1.5f) - 1;
			};
			auto dg = [](Scalar x) {
				return 2 * x - 3;
			};
			mma_t solver(1, 1, 1e-3);
			solver.init(0.5, 2);
			auto& xlist = solver.get_x();
			bool converged = false;

			gv::gVector fdiff(1);
			gv::gVector gval(1);
			std::vector<Scalar*> gdiff(1);
			gv::gVector dgv(1);
			gdiff[0] = dgv.data();

			while (!converged) {

				Scalar new_x = xlist[0];

				printf("current x     = %f\n", new_x);

				printf("        df(x) = %f\n", df(new_x));

				printf("        g(x)  = %f\n", g(new_x));

				printf("        dg(x) = %f\n", dg(new_x));

				fdiff[0] = df(new_x);
				gval[0] = g(new_x);
				dgv[0] = dg(new_x);

				converged = solver.update(fdiff.data(), gdiff.data(), gval.data());
			}
		}

		// test 2-dim
		if (1)
		{
			auto f = [](Scalar x, Scalar y) {
				return x / 2 + y * sqrt(3) / 2 + exp(10 * pow(x - y, 2));
			};
			auto df = [](Scalar x, Scalar y) {
				std::vector<Scalar> ret;
				ret.push_back(0.5 + 20 * (x - y)*exp(10 * pow(x - y, 2)));
				ret.push_back(sqrt(3) / 2 + 20 * (y - x)*exp(10 * pow(x - y, 2)));
				return ret;
			};
			auto g = [](Scalar x, Scalar y) {
				std::vector<Scalar> ret;
				ret.push_back(x*x + y * y - 1);
				ret.push_back(x*x - 3. / 4);
				ret.push_back(y*y - 3. / 4);
				ret.push_back(pow(x + y, 2) - 1);
				return ret;
			};
			auto dg = [](Scalar x, Scalar y) {
				std::vector<std::vector<Scalar>> ret;
				ret.resize(4);
				ret[0].push_back(2 * x);
				ret[0].push_back(2 * y);

				ret[1].push_back(2 * x);
				ret[1].push_back(0);

				ret[2].push_back(0);
				ret[2].push_back(2 * y);

				ret[3].push_back(2 * (x + y));
				ret[3].push_back(2 * (x + y));
				return ret;
			};

			mma_t solver(2, 4);

			//solver.set_constrain_amplifier(1e3, 1e2);

			solver.init(-1, 1);

			gv::gVector fdiff(2);
			gv::gVector gval(4);
			std::vector<gv::gVector> gdiffval(4);
			std::vector<Scalar*> gdiff(4);
			for (int i = 0; i < 4; i++) {
				gdiffval[i] = gv::gVector(2);
				gdiff[i] = gdiffval[i].data();
			}

			gv::gVector cur_x(2);

			bool converged = false;
			while (!converged) {
				cur_x = solver.get_x();
				
				printf("current : \n");
				printf("x = (%f, %f)\n", Scalar(cur_x[0]), Scalar(cur_x[1]));

				// compute difference of current df
				auto diff_f = df(cur_x[0], cur_x[1]);
				for (int i = 0; i < diff_f.size(); i++) {
					fdiff[i] = diff_f[i];
				}
				printf("Df = (%f, %f)\n", diff_f[0], diff_f[1]);

				// compute difference of current g
				auto diff_g = dg(cur_x[0], cur_x[1]);
				for (int i = 0; i < diff_g.size(); i++) {
					for (int j = 0; j < 2; j++) {
						gdiffval[i][j] = diff_g[i][j];
					}
				}

				printf("Dg = [(%f, %f),\n", diff_g[0][0], diff_g[0][1]);
				printf("      (%f, %f),\n", diff_g[1][0], diff_g[1][1]);
				printf("      (%f, %f),\n", diff_g[2][0], diff_g[2][1]);
				printf("      (%f, %f)]\n", diff_g[3][0], diff_g[3][1]);

				// compute current constrain value 
				auto gv = g(cur_x[0], cur_x[1]);
				for (int i = 0; i < gv.size(); i++) {
					gval[i] = gv[i];
				}

				printf("g = [ %f,\n", gv[0]);
				printf("      %f,\n", gv[1]);
				printf("      %f,\n", gv[2]);
				printf("      %f ]\n", gv[3]);
				
				if (solver.update(fdiff.data(), gdiff.data(), gval.data())) {
					break;
				}
			}
		}
	}

	void mma_t::set_constrain_amplifier(Scalar prefer_c, Scalar prefer_d)
	{
		c.set(prefer_c);
		d.set(prefer_d);
	}

	void mma_t::toMatlab(void)
	{
#ifdef __MMA_WITH_MATLAB
		x.toMatlab("x");
		y.toMatlab("y");
		lambda.toMatlab("lambda");
		xi.toMatlab("xi");
		eta.toMatlab("eta");
		mu.toMatlab("mu");
		s.toMatlab("s");
		alpha.toMatlab("alpha");
		beta.toMatlab("beta");
		asym_l.toMatlab("asym_l");
		asym_u.toMatlab("asym_u");
#endif
	}


	int mma_t::n_w(void)
	{
		int n = n_dim();
		int m = n_constrain();
		return n + m + 1 + m + n + n + m + 1 + m;
	}

}

