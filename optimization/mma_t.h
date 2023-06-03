#pragma once

#ifndef __MY_MMA_T_H
#define __MY_MMA_T_H

#include "gpuVector.h"
#include "vector"
#include "cusparse.h"
#include "cusolverSp.h"
#include "Eigen/Sparse"

#ifdef __GVECTOR_WITH_MATLAB
#define __MMA_WITH_MATLAB
#endif


namespace MMA {


	typedef float Scalar;

	constexpr Scalar asym_clamp_factor = 0.99f;

	class mma_subproblem_t;

	class mma_t;

class mma_subproblem_t {
public:
	typedef gv::Scalar Scalar;
private:
	std::vector<gv::gVector*> p;
	std::vector<gv::gVector*> q;
	gv::gVector plambda;
	gv::gVector qlambda;

	//gv::gVector varphi_x;

	gv::gVector Psi;

	std::vector<gv::gVector*> G;

	//gv::gVector d;

	mma_t& mma;

	gv::gVector gproxy;

	gv::gVector b;

	Eigen::SparseMatrix<gv::Scalar, Eigen::RowMajor> A;

	// subproblem line search variable
	gv::gVector dx;
	gv::gVector dy;
	Scalar      dz;
	gv::gVector dlambda;

	gv::gVector dxi;
	gv::gVector deta;
	gv::gVector dmu;
	Scalar      dzeta;
	gv::gVector ds;

	gv::gVector new_x;
	gv::gVector new_y;
	Scalar      new_z;
	gv::gVector new_lambda;
	gv::gVector new_xi;
	gv::gVector new_eta;
	gv::gVector new_mu;
	Scalar      new_zeta;
	gv::gVector new_s;

	gv::gVector kkt_err_tmp;

	struct cuSolver_t{
		cusolverSpHandle_t spHandle;
		// sparse matrix handle

		cusparseMatDescr_t descr;

		int n_nonzeros;

		int nrows;

		int* row_ptr = nullptr;
		// device row pointer

		int* col_ptr = nullptr;
		// device col pointer

		gv::Scalar* val_ptr = nullptr;
		// device value pointer

		gv::Scalar* b_ptr = nullptr;
		// device right side value pointer

		~cuSolver_t();

		void update(int* hot_row_ptr, int* host_col_ptr, gv::Scalar* host_val_ptr, gv::Scalar* host_b_ptr);

		void solve(gv::gVector& result);


		int n_elements(void) { return n_nonzeros; }

		int n_row_indices(void) { return nrows + 1; }
	} cuSolver;

	friend class mma_t;

private:
	gv::gVector get_dx(const gv::gVector& dw);
	gv::gVector get_dy(gv::gVector& dw);
	Scalar get_dz(gv::gVector& dw);
	gv::gVector get_dlambda(gv::gVector& dw);

	gv::gVectorMap get_artivar(gv::gVector& w) const;

	bool init(gv::gVector* df, std::vector<gv::gVector*>& dg, gv::gVector* g);

	void update_pqlambda(gv::gVector& newLambda);

	void whole_dw(gv::gVector& simple_dw, gv::gVector& dw);

public:

	bool initialized(void);;

	void clear();

	mma_subproblem_t(mma_t& solver) :mma(solver) {};

	~mma_subproblem_t();


	void reset(void);

	std::pair<Scalar, Scalar> solve(gv::gVector* df, std::vector<gv::gVector*>& dg, gv::gVector* g, gv::gVector& dx);

	std::pair<Scalar, Scalar> search_step(gv::gVector& dw, Scalar initial_step, Scalar lambda_damp = 1);

	std::pair<Scalar, Scalar> kkt_err(
		gv::gVector& new_x, gv::gVector& new_y, Scalar new_z, gv::gVector& new_lambda,
		gv::gVector& new_xi, gv::gVector& new_eta, gv::gVector& new_mu,
		Scalar new_zeta, gv::gVector& new_s, Scalar lambda_damp = 1
	);

	gv::gVector solveLinearSystem(gv::gVector& Dx, gv::gVector& Dy, std::vector<gv::gVector*>& G, gv::Scalar zetadz, gv::gVector& a, gv::gVector& Dlambda,
		gv::gVector& deltax, gv::gVector& deltay, gv::Scalar deltaz, gv::gVector& deltaLambda);

	void largeVarlessConstrainLinearSolve(std::vector<std::vector<Scalar>>& A11, std::vector<Scalar>& A12, Scalar A13, std::vector<Scalar>& b1, Scalar b2, std::vector<Scalar>& result);

	void toMatlab(void);
};


class mma_t {
private:	
	// dim n
	gv::gVector x, xmin, asym_l, asym_u, alpha, beta, xmax; // design variable and its moving asymptotes and lower and upper bound for x
	gv::gVector lastdx, dx;  // last change of design variable
	gv::gVector xi; // multiplier of lower bound constrain for x, xi * ( x - alpha ) >= 0
	gv::gVector eta; // multiplier of upper bound constrain for x, eta * ( beta - x ) >=0

	Scalar z; // artificial variable for min max problem
	Scalar zeta; // multiplier of z , - zeta * z = 0

	// dim m
	gv::gVector y; // relax variable in inequality constrain in initial problem, g(x) - a * z <= y
	gv::gVector lambda; // multiplier of inequality constrain in initial problem lambda * (g - a * z - y) ==0
	gv::gVector mu; // multiplier of y >=0, - mu * y = 0 

	gv::gVector s; // slackness variable for inequality constrain in subproblem , g - a * z - y + s - b = 0

	// optimize parameter
	Scalar a0;
	gv::gVector a; // amplifier of z in inequality constrain 
	gv::gVector c, d; // punish amplifier of y in inequality constrain

	// slackness error in subproblem
	Scalar epsilon;

	//
	Scalar epsilon_tol = 1e-7;

	// 
	int stop_counter = 0;

	friend class mma_subproblem_t;

	mma_subproblem_t subproblem;

private:
	void set_bound(Scalar* lower_bound, Scalar* upper_bound);

	void set_bound(Scalar uniform_lower_bound, Scalar uniform_upper_bound);

	void init_subproblem_variable(void);
public:
	mma_t(int ndim, int nConstrain, Scalar err_tol = 1e-7)
		: x(ndim), lastdx(ndim, 0), dx(ndim, 0), asym_l(ndim), asym_u(ndim), alpha(ndim), beta(ndim), xmin(ndim), xmax(ndim), xi(ndim), eta(ndim),
		y(nConstrain, 1), lambda(nConstrain, 1), mu(nConstrain), s(nConstrain, 1),
		z(1), zeta(1),
		a0{ 1 }, a(nConstrain, 0),
		c(nConstrain, 10), d(nConstrain, 1),
		epsilon{ 1 },
		subproblem(*this),
		epsilon_tol(err_tol)
	{
		gv::gVector::Init((std::max)(ndim, nConstrain));
	}

	void set_constrain_amplifier(Scalar prefer_c, Scalar prefer_d);

	void init(Scalar* lower_bound, Scalar* upper_bound) {
		set_bound(lower_bound, upper_bound);
		init_subproblem_variable();
	}

	void init(Scalar lower_bound, Scalar upper_bound) {
		set_bound(lower_bound, upper_bound);
		init_subproblem_variable();
		toMatlab();
	}

	void toMatlab(void);

	void update(
		gv::gVector& new_x, gv::gVector& new_y, Scalar new_z, gv::gVector& new_lambda,
		gv::gVector& new_xi, gv::gVector& new_eta, gv::gVector& new_mu,
		Scalar new_zeta, gv::gVector& new_s
	);

	gv::gVector& get_x(void) {
		return x;
	}

	void get_x(Scalar* dst);

	void get_w(gv::gVector& w);

	void get_w(gv::gVector& new_x, gv::gVector& new_y, Scalar new_z, gv::gVector& new_lambda,
		gv::gVector& new_xi, gv::gVector& new_eta, gv::gVector& new_mu,
		Scalar new_zeta, gv::gVector& new_s, gv::gVector& w
	);

	int n_dim(void) { return x.size(); }

	int n_constrain(void) { return y.size(); }

	int n_w(void);

	bool update(Scalar* dev_df, Scalar** dev_dg, Scalar* dev_g);
	// return if converged
	/*
		df : derivative of objective function
		dg : derivative of constrain function, dg1_1,dg1_2,dg1_3,...,dg1_n,dg2_1,dg2_2,...
		g  : constrain function value at current x
	*/

	void adjust_asym(gv::gVector& dx1, gv::gVector& dx2);

	void clamp_asymptotes(void);
};

extern void test_mma(void);

}


#endif
