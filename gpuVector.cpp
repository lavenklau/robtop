#include "gpuVector.h"
#include "iostream"

#ifdef __GVECTOR_WITH_MATLAB
#include "matlab_utils.h"
#include "Eigen\Eigen"
#endif

using namespace gv;

void gVector::toMatlab(const char* vName)
 {
#if defined(__GVECTOR_WITH_MATLAB) && defined(ENABLE_MATLAB)
	 Eigen::Matrix<Scalar, -1, 1> vec;
	 vec.resize(size(), 1);
	 download(vec.data());
	 eigen2ConnectedMatlab(vName, vec);
#endif
 }

void gv::gVector::resize(size_t dim)
{
	build(dim);
}

gv::gVectorMap::~gVectorMap(void)
{
	//std::cout << "destructing gvector map " << data() << " with size " << size() << std::endl;
	_Get_data() = nullptr;
	_Get_size() = 0;
}

gv::gElementProxy gv::gVector::operator[](int eid)
{
	return gv::gElementProxy(_data + eid);
}

void gv::gVector::toMatlab(const char* vName, std::vector<gVector*>& vecs)
{
#if defined(__GVECTOR_WITH_MATLAB) && defined(ENABLE_MATLAB)
	int dim = vecs[0]->size();
	for (int i = 0; i < vecs.size(); i++) {
		if (vecs[i]->size() != dim) {
			std::cout << "\033[31m" << "Warning : gVector list has different dimension !" << "\033[0m" << std::endl;
			return;
		}
	}

	Eigen::Matrix<Scalar, -1, -1> vcs(dim, vecs.size());

	for (int i = 0; i < vecs.size(); i++) {
		vecs[i]->download(vcs.col(i).data());
	}

	eigen2ConnectedMatlab(vName, vcs);

#endif
}

void gv::gVector::toMatlab(const char* vName, std::vector<Scalar*>& vecs, int dim)
{
#if defined(__GVECTOR_WITH_MATLAB) && defined(ENABLE_MATLAB)
	Eigen::Matrix<Scalar, -1, -1> vcs(dim, vecs.size());
	
	for (int i = 0; i < vecs.size(); i++) {
		gv::gVectorMap(vecs[i], dim).download(vcs.col(i).data());
	}

	eigen2ConnectedMatlab(vName, vcs);
#endif
}

