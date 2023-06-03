#include "projection.h"
#include "iostream"
#include "Eigen/Sparse"
#include "snippet.h"
#include "CGALDefinition.h"
#include "matlab_utils.h"
#include "binaryIO.h"

std::vector<int> _loadnodes;

std::vector<Eigen::Matrix<double, 3, 1>> _loadpos;

std::vector<Eigen::Matrix<double, 3, 1>> _loadnormals;

std::vector<Eigen::Matrix<double, 3, 1>> _loadforce;

Eigen::SparseMatrix<double> _Nd;

Eigen::Matrix<double, -1, -1> _Rd;

Eigen::Matrix<double, -1, 1> _f;

Eigen::Matrix<double, -1, 1> _wNd;

Eigen::Matrix<double, -1, 1> _wRd;

Eigen::Matrix<double, -1, 6> _R;

Eigen::Matrix<double, -1, 1> _Ru[6][3];

const int* _vlex2gs_dev;

const int* _nodeflag;

extern int _n_gsnodes;

int _n_nodes;

extern grid::HierarchyGrid grids;

std::ostream& log() {
	std::cout << "[\033[33mProjection\033[0m]";
	return std::cout;
}

void setNodes(BitSAT<unsigned int>& vbits, int vreso, const std::vector<int>& lex2gs, const int* vlex2gs_dev, const int* nodeflag, int n_gs)
{
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 3; j++) {
			_Ru[i][j].resize(n_gs, 1);
			_Ru[i][j].fill(0);
		}
	}

	_R.resize(n_gs * 3, 6);
	_R.fill(0);
	int vreso2 = pow(vreso, 2);
#pragma omp parallel for
	for (int i = 0; i < vbits._bitArray.size(); i++) {
		unsigned int word = vbits._bitArray[i];
		if (word == 0) continue;
		int vidoffset = vbits._chunkSat[i];
		int vidword = 0;
		for (int j = 0; j < BitCount<unsigned int>::value; j++) {
			if (read_bit(word, j)) {
				int vid = vidword + vidoffset;
				int gsvid = lex2gs[vid];
				int bitid = i * BitCount<unsigned int>::value + j;
				int  p[3] = { bitid % vreso, bitid % vreso2 / vreso,bitid / vreso / vreso };
				Eigen::Matrix<double, 3, 3> phat;
				phat << 0, -p[2], p[1],
					p[2], 0, -p[0],
					-p[1], p[0], 0;
				_R.block<3, 3>(gsvid * 3, 0) = Eigen::Matrix<double, 3, 3>::Identity();
				_R.block<3, 3>(gsvid * 3, 3) = phat;

				//for (int k = 0; k < 3; k++) {
				//	_Ru[k][k][gsvid] = 1;
				//	_Ru[k + 3][0][gsvid] = phat(0, k);
				//	_Ru[k + 3][1][gsvid] = phat(1, k);
				//	_Ru[k + 3][2][gsvid] = phat(2, k);
				//}


				vidword++;
			}
		}
	}


	// Gram-Schmitt Orthogonalization
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < i; j++) {
			_R.col(i) -= _R.col(i).dot(_R.col(j))*_R.col(j);
		}
		_R.col(i).normalize();
	}

	eigen2ConnectedMatlab("R", _R);

	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < n_gs; k++) {
				_Ru[i][j][k] = _R(k * 3 + j, i);
			}
		}
	}

	_n_nodes = vbits.total();
	_vlex2gs_dev = vlex2gs_dev;
	_n_gsnodes = n_gs;

	//_R.resize(1, 6);
#if 0
	Eigen::Matrix<double, -1, 6> pR[3];
	for (int i = 0; i < 3; i++) pR[i].resize(n_gs, 6);
	for (int i = 0; i < _R.rows(); i++) {
		int vid = i / 3;
		int cid = i % 3;
		pR[cid].row(vid) = _R.row(i);
	}

	double* Rdata[3] = { pR[0].data(),pR[1].data(),pR[2].data() };
	uploadRigidMatrix(Rdata, n_gs);
	uploadNodeFlags(nodeflag, n_gs);
#endif
}

void setLoadNodes(
	const std::vector<int>& loadnodes,
	const std::vector<Eigen::Matrix<double, 3, 1>>& loadpos,
	const std::vector<Eigen::Matrix<double, 3, 1>>& loadnormal,
	const std::vector<Eigen::Matrix<double, 3, 1>>& loadforce) {
	_loadnodes = loadnodes;
	_loadnormals = loadnormal;
	_loadpos = loadpos;

	// sort loadnodes
	auto permu = snippet::sort_permutation(_loadnodes, std::less<int>());
	_loadnodes = snippet::apply_permutation(_loadnodes, permu);
	_loadnormals = snippet::apply_permutation(_loadnormals, permu);
	_loadpos = snippet::apply_permutation(_loadpos, permu);
	_loadforce = snippet::apply_permutation(loadforce, permu);
	

	// DEBUG
	{
		std::vector<double> posflat(_loadnodes.size() * 3);
		for (int i = 0; i < _loadpos.size(); i++) {
			auto v = _loadpos[i];
			posflat[i * 3] = v[0];
			posflat[i * 3 + 1] = v[1];
			posflat[i * 3 + 2] = v[2];
		}
		bio::write_vector(grids.getPath("loadpos"), posflat);

		bio::write_vector(grids.getPath("nodeids"), _loadnodes);

		std::vector<double> normalflat(_loadnormals.size() * 3);
		for (int i = 0; i < _loadnormals.size(); i++) {
			auto v = _loadnormals[i];
			normalflat[i * 3] = v[0];
			normalflat[i * 3 + 1] = v[1];
			normalflat[i * 3 + 2] = v[2];
		}
		bio::write_vector(grids.getPath("loadnormal"), normalflat);
	}

	std::vector<double> vnormal[3];

	for (int i = 0; i < _loadnormals.size(); i++) {
		vnormal[0].emplace_back(_loadnormals[i][0]);
		vnormal[1].emplace_back(_loadnormals[i][1]);
		vnormal[2].emplace_back(_loadnormals[i][2]);
	}

	std::vector<Eigen::Triplet<double>> triplist;

	std::vector<double> vtangent[2][3];

	for (int i = 0; i < _loadnodes.size(); i++) {
		Kernel::Plane_3 tagentplane(Point(_loadpos[i][0], _loadpos[i][1], _loadpos[i][2]), Kernel::Vector_3(_loadnormals[i][0], _loadnormals[i][1], _loadnormals[i][2]));
		Kernel::Vector_3 v0 = tagentplane.base1();
		double v0len = sqrt(v0.squared_length());
		Kernel::Vector_3 v1 = tagentplane.base2();
		double v1len = sqrt(v1.squared_length());

		for (int k = 0; k < 3; k++) {
			vtangent[0][k].emplace_back(v0[k] / v0len);
			vtangent[1][k].emplace_back(v1[k] / v1len);
		}
		if (CGAL::scalar_product(v0 / v0len, Kernel::Vector_3(_loadnormals[i][0], _loadnormals[i][1], _loadnormals[i][2])) > 1e-2
			|| CGAL::scalar_product(v1 / v1len, Kernel::Vector_3(_loadnormals[i][0], _loadnormals[i][1], _loadnormals[i][2])) > 1e-2
			) {
			printf("-- wrong base\n");
		}

		if (!grids.isForceFree()) {
			for (int j = 0; j < 3; j++) {
				triplist.emplace_back(i * 2, i * 3 + j, v0[j] / v0len);
				triplist.emplace_back(i * 2 + 1, i * 3 + j, v1[j] / v1len);
			}
		}
	}


	_Nd.resize(_loadnodes.size() * 2, _loadnodes.size() * 3);
	_Nd.setFromTriplets(triplist.begin(), triplist.end());

	_Rd.resize(_loadnodes.size() * 3, 6);

	for (int i = 0; i < _loadnodes.size(); i++) {
		auto p = _loadpos[i];
		Eigen::Matrix<double, 3, 3> phat;
		phat << 0, -p[2], p[1],
			p[2], 0, -p[0],
			-p[1], p[0], 0;
		_Rd.block<3, 3>(i * 3, 0) = Eigen::Matrix<double, 3, 3>::Identity();
		_Rd.block<3, 3>(i * 3, 3) = phat;
	}

	_Nd.makeCompressed();

	//Eigen::SparseMatrix<double> Nd1 = _Nd;
	//Nd1.conservativeResize(Nd1.cols(), Nd1.cols());
	eigen2ConnectedMatlab("Rd", _Rd);
	eigen2ConnectedMatlab("Nd", _Nd);
	//getMatEngine().eval("spy(Nd);");

	// remove component in Nd
	if (!grids.isForceFree()) {
		_Rd -= _Nd.transpose()*(_Nd*_Rd);
	}

	// Gram-Schmitt Orthonormal
	bool invalid[6] = { false };
	for (int i = 0; i < 6; i++) {
		double oldres = _Rd.col(i).norm();
		for (int j = 0; j < i; j++) {
			if (invalid[j]) continue;
			double w = _Rd.col(i).dot(_Rd.col(j));
			_Rd.col(i) -= w * _Rd.col(j);
		}
		double res = _Rd.col(i).norm();
		log() << "Rigid motion " << i << " res " << res << "(" << oldres << ")" << std::endl;
		if (res / oldres < 1e-6 || oldres < 1e-6) {
			log() << "Motion deprecated" << std::endl;
			invalid[i] = true;
		}
		_Rd.col(i) /= res;
	}

	int validcounter = 0;
	for (int i = 0; i < 6; i++) {
		if (!invalid[i] ) {
			_Rd.col(validcounter++) = _Rd.col(i);
		}
	}
	
	_Rd.conservativeResize(_loadnodes.size() * 3, validcounter);

	eigen2ConnectedMatlab("Rd", _Rd);

	_f.resize(_loadnodes.size() * 3, 1);

	uploadLoadNodes(_loadnodes, vtangent, vnormal);

	setLoadForce(_loadforce);
}

void setLoadForce(std::vector<Eigen::Matrix<double, 3, 1>> flist)
{
	std::vector<double> fhost[3];
	for (int i = 0; i < 3; i++) fhost[i].resize(getLoadNodes().size());
	for (int i = 0; i < flist.size(); i++) {
		fhost[0][i] = flist[i][0];
		fhost[1][i] = flist[i][1];
		fhost[2][i] = flist[i][2];
	}
	double* fhostptr[3] = { fhost[0].data(),fhost[1].data(),fhost[2].data() };
	uploadLoadForce(fhostptr);
}

const std::vector<int>& getLoadNodes(void)
{
	return _loadnodes;
}

void forceProject(std::vector<double> f[3])
{
	// FOR DEBUG
	//{
	//	Eigen::Matrix<double, -1, 3> f3;
	//	f3.resize(f->size(), 3);
	//	for (int i = 0; i < 3; i++) {
	//		memcpy(f3.col(i).data(), f[i].data(), sizeof(double) * f[i].size());
	//	}
	//	eigen2ConnectedMatlab("fsupport", f3);

	//	std::vector<double> flat(_loadnodes.size() * 3);
	//	for (int i = 0; i < _loadnodes.size(); i++) {
	//		flat[i * 3] = f[0][i];
	//		flat[i * 3 + 1] = f[1][i];
	//		flat[i * 3 + 2] = f[2][i];
	//	}
	//	bio::write_vector("./out/fsupport", flat);
	//}

	// flatten vector
	for (int i = 0; i < _loadnodes.size(); i++) {
		for (int j = 0; j < 3; j++) {
			_f[i * 3 + j] = f[j][i];
		}
	}

	 //DEBUG
	eigen2ConnectedMatlab("fs", _f);

	_wNd = _Nd * _f;
	_wRd = _Rd.transpose()*_f;

	eigen2ConnectedMatlab("wRd", _wRd);

	// DEBUG
	//eigen2ConnectedMatlab("wNd", _wNd);

	if (grids.isForceFree() && !grids.hasSupport()) {
		_f -= _Rd * _wRd;
	}
	else if (grids.isForceFree() && grids.hasSupport()) {
		//_f.fill(0);
	}
	else if (!grids.isForceFree() && !grids.hasSupport()) {
		_f -= _Nd.transpose() * _wNd + _Rd * _wRd;
	}
	else if (!grids.isForceFree() && grids.hasSupport()) {
		_f -= _Nd.transpose() * _wNd;
	}

	// stack vector
	for (int i = 0; i < _loadnodes.size(); i++) {
		for (int j = 0; j < 3; j++) {
			f[j][i] = _f[i * 3 + j];
		}
	}

	// DEBUG
	//{
	//	std::vector<double> flat(_loadnodes.size() * 3);
	//	for (int i = 0; i < _loadnodes.size(); i++) {
	//		flat[i * 3] = f[0][i];
	//		flat[i * 3 + 1] = f[1][i];
	//		flat[i * 3 + 2] = f[2][i];
	//	}
	//	bio::write_vector("./out/fsproj", flat);
	//}

	// DEBUG
	eigen2ConnectedMatlab("fsproj", _f);
}

void writeSupportForce(const std::string& filename, double const * const f_dev[3])
{
	std::vector<double> fs[3];
	getForceSupport(f_dev, fs);
	bio::write_vectors(filename, fs, false);
}

int n_loadnodes(void)
{
	return _loadnodes.size();
}

void displacementProject(double* u_dev[3])
{
	// DEBUG
	//grids[0]->v3_toMatlab("u_dev", u_dev);

	double* uRdev[3];
	grids[0]->v3_create(uRdev);

	double s = 0;
	for (int k = 0; k < 6; k++) {
		for (int j = 0; j < 3; j++) {
			gpu_manager_t::upload_buf(uRdev[j], _Ru[k][j].data(), sizeof(double) * _n_gsnodes);
		}
		s = grids[0]->v3_dot(u_dev, uRdev);
		//std::cout << "-- s" << k << " = " << s << std::endl;
		grids[0]->v3_minus(u_dev, s, uRdev);
	}

	grids[0]->v3_destroy(uRdev);

	//Eigen::Matrix<double, -1, 1> vhost(_n_gsnodes * 3, 1);
	//std::vector<double> uhost(_n_gsnodes);
	//for (int i = 0; i < 3; i++) {
	//	gpu_manager_t::download_buf(uhost.data(), u_dev[i], sizeof(double)*_n_gsnodes);
	//	for (int j = 0; j < _n_gsnodes; j++) {
	//		vhost[j * 3 + i] = uhost[j];
	//	}
	//}

	//vhost -= _R * (_R.transpose() * vhost);

	//for (int i = 0; i < 3; i++) {
	//	for (int j = 0; j < _n_gsnodes; j++) {
	//		uhost[j] = vhost[j * 3 + i];
	//	}
	//	gpu_manager_t::upload_buf(u_dev[i], uhost.data(), sizeof(double) * _n_gsnodes);
	//}
}
