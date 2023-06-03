#include "templateMatrix.h"
#include "snippet.h"
#include "math.h"
#include "matlab_utils.h"


/* initialize elastic matrix for further computation */
Scalar mu = default_poisson_ratio;
Scalar E = default_youngs_modulus;
Eigen::Matrix<Scalar, 6, 6> elastic_matrix;
Eigen::Matrix<Scalar, 24, 24> Ke;

Scalar* g_Ke;

Eigen::Matrix<Scalar, 3, 1> dN(int i, Scalar elen, Eigen::Matrix<Scalar, 3, 1>& param) {
	if (i > 7 || i < 0) throw std::runtime_error("");
	int id[3] = { (i % 2),   (i / 2 % 2),   (i / 4) };
	Scalar r[3];
	for (int k = 0; k < 3; k++) {
		r[k] = (id[k] ? param[k] : (1 - param[k]));
	}

	Eigen::Matrix<Scalar, 3, 1> dn;
	for (int k = 0; k < 3; k++) {
		snippet::Zp<3> j;
		dn[k] = (id[k] ? 1.0 / elen : -1.0 / elen) * r[j[k + 1]] * r[j[k + 2]];
	}
	return dn;
}

void initTemplateMatrix(
	Scalar element_len, gpu_manager_t& gm,
	Scalar ymodu /*= default_younds_modulu*/, Scalar ps_ratio /*= default_poisson_ratio*/)
{
	mu = ps_ratio;
	E = ymodu;
	elastic_matrix << 1 - mu, mu, mu, 0, 0, 0,
		mu, 1 - mu, mu, 0, 0, 0,
		mu, mu, 1 - mu, 0, 0, 0,
		0, 0, 0, (1 - 2 * mu) / 2, 0, 0,
		0, 0, 0, 0, (1 - 2 * mu) / 2, 0,
		0, 0, 0, 0, 0, (1 - 2 * mu) / 2;
	elastic_matrix *= E / ((1 + mu)*(1 - 2 * mu));

	Eigen::Matrix<Scalar, 3, 1> gs_points[8];
	
	double p = sqrt(3) / 3;

	for (int i = 0; i < 8; i++) {
		int x = 2 * (i % 2) - 1;
		int y = 2 * (i / 2 % 2) - 1;
		int z = 2 * (i / 4) - 1;
		gs_points[i][0] = (x * p + 1) / 2;
		gs_points[i][1] = (y * p + 1) / 2;
		gs_points[i][2] = (z * p + 1) / 2;
	}

	
	Ke.fill(0);
	// Gauss Quadrature Point
	for (int i = 0; i < 8; i++) {

		Eigen::Matrix<Scalar, 3, 1> grad_N[8];

		// Element Vertex Point
		for (int k = 0; k < 8; k++) {
			grad_N[k] = dN(k, element_len, gs_points[i]);
		}

		Eigen::Matrix<Scalar, 6, 24> B;

		B.fill(0);

		for (int a = 0; a < 3; a++) {
			int offset = a;
			for (int b = 0; b < 8; b++) {
				B(a, offset) = grad_N[b][a];
				offset += 3;
			}
		}
		int offset = 0;
		/// torsional strain tau
		for (int b = 0; b < 8; b++) {
			/// tau_yz
			B(3, offset + 1) = grad_N[b].z();
			B(3, offset + 2) = grad_N[b].y();
			/// tau_xz
			B(4, offset) = grad_N[b].z();
			B(4, offset + 2) = grad_N[b].x();
			/// tau_xy
			B(5, offset) = grad_N[b].y();
			B(5, offset + 1) = grad_N[b].x();

			offset += 3;
		}
		Ke += B.transpose() * elastic_matrix * B;
	}

	Ke *= pow(element_len / 2, 3);

	eigen2ConnectedMatlab("KE", Ke);
	//g_Ke = (Scalar*)gm.add_buf("template matrix buf ", sizeof(Ke), Ke.data());

	// DEBUG
	// compute rigid motion on element
	Eigen::Matrix<double, 24, 6> RE;
	for (int i = 0; i < 8; i++) {
		RE.block<3, 3>(i * 3, 0) = Eigen::Matrix<double, 3, 3>::Identity();
		Eigen::Matrix3d phat;
		int p[3] = { i % 2, i / 2 % 2, i / 2 / 2 };
		phat << 0, -p[2], p[1],
			p[2], 0, -p[0],
			-p[1], p[0], 0;
		RE.block<3, 3>(i * 3, 3) = phat;
	}
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < i; j++) {
			RE.col(i) -= RE.col(i).dot(RE.col(j)) * RE.col(j);
		}
		RE.col(i).normalize();
	}

	eigen2ConnectedMatlab("RE", RE);
}

const Eigen::Matrix<Scalar, 24, 24>& getTemplateMatrix(void)
{
	return Ke;
}

const Scalar* getTemplateMatrixElements(void)
{
	return Ke.data();
}

Scalar* getDeviceTemplateMatrix(void) {
	return g_Ke;
}

