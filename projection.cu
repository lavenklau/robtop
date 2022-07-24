#include "projection.h"
#include "lib.cuh"
#include "snippet.h"

#include "helper_cuda.h"

//#include "cublas.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION undefined!
#elif (CUDART_VERSION >= 11060)
#define check_cublas( sta ) if(sta!=CUBLAS_STATUS_SUCCESS) { printf("\033[31mcuBLAS error at line %d, file %s\n error name : %s\n\033[0m",__LINE__,__FILE__,cublasGetStatusName(sta));}
#else 
#define check_cublas( sta ) if(sta!=CUBLAS_STATUS_SUCCESS) { printf("\033[31mcuBLAS error at line %d, file %s\n error name : %s\n\033[0m",__LINE__,__FILE__,_cudaGetErrorEnum(sta));}
#endif

//#define check_cublas( sta ) if(sta!=CUBLAS_STATUS_SUCCESS) { printf("\033[31mcuBLAS error at line %d, file %s\n error name : %s\n\033[0m",__LINE__,__FILE__,cublasGetStatusName(sta));}

double* _Rdata[3];
double* _Rudata[3];
cublasHandle_t cublas_handle;

extern grid::HierarchyGrid grids;

int _n_gsnodes;

cublasStatus_t sta;

extern const int* _vlex2gs_dev;

extern const int* _nodeflag;

extern int _n_nodes;

int* gloadnodes;

double* gvtangent[2][3];

double* gvnormal[3];

double* fload[3];

gBitSAT<unsigned int> vid2loadid;

extern std::vector<int> _loadnodes;

__constant__ double* gLoadtangent[2][3];
__constant__ double* gLoadnormal[3];

void uploadRigidMatrix(double* pdata[3], int n_gs)
{
	_n_gsnodes = n_gs;
	sta = cublasCreate(&cublas_handle);
	
	check_cublas(sta);

	for (int i = 0; i < 3; i++) {
		cudaMalloc(&_Rdata[i], sizeof(double) * 6 * n_gs);
		cudaMemcpy(_Rdata[i], pdata[i], sizeof(double) * 6 * n_gs, cudaMemcpyHostToDevice);
		cudaMalloc(&_Rudata[i], sizeof(double) * 6);
		init_array(_Rudata[i], 0., 6);
	}
	cuda_error_check;
}

void uploadNodeFlags(const int* nodeflags, int n_gs)
{
	_n_gsnodes = n_gs;
	cudaMalloc(&_nodeflag, sizeof(int) * _n_nodes);
	cudaMemcpy(const_cast<int*>(_nodeflag), nodeflags, sizeof(int) * _n_nodes, cudaMemcpyHostToDevice);
}

void uploadLoadNodes(const std::vector<int>& loadnodes, std::vector<double> vtang[2][3], std::vector<double> vnormal[3])
{
	// upload loadnodes indices
	cudaMalloc(&gloadnodes, sizeof(int) * loadnodes.size());
	cudaMemcpy(gloadnodes, loadnodes.data(), sizeof(int) * loadnodes.size(), cudaMemcpyHostToDevice);
	cuda_error_check;

	// upload load nodes tangent vector and normal vectors
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			cudaMalloc(&gvtangent[j][i], sizeof(double) * vtang[j][i].size());
			cudaMemcpy(gvtangent[j][i], vtang[j][i].data(), sizeof(double) * vtang[j][i].size(), cudaMemcpyHostToDevice);
		}
		cudaMalloc(&gvnormal[i], sizeof(double) * vnormal[i].size());
		cudaMemcpy(gvnormal[i], vnormal[i].data(), sizeof(double) * vnormal[i].size(), cudaMemcpyHostToDevice);
	}

	// upload pointer to tangent vector and normal vectors to constant memory
	cudaMemcpyToSymbol(gLoadtangent, &gvtangent[0][0], sizeof(gLoadtangent));
	cudaMemcpyToSymbol(gLoadnormal, &gvnormal[0], sizeof(gLoadnormal));

	// build load nodes sat
	int nbitword = snippet::Round<grid::BitCount<unsigned int>::value>(_n_gsnodes) / grid::BitCount<unsigned int>::value;
	cudaMalloc(&vid2loadid._bitarray, nbitword * sizeof(unsigned int));
	cudaMalloc(&vid2loadid._chunksat, (nbitword + 1) * sizeof(int));
	init_array(const_cast<unsigned int*>(vid2loadid._bitarray), (unsigned int)(0), nbitword);
	init_array(const_cast<int*>(vid2loadid._chunksat), int{ 0 }, nbitword + 1);
	// set bit of load nodes on device
	auto loadsat = vid2loadid;
	int* loadids = gloadnodes;
	auto set_loadbit = [=] __device__(int tid) {
		atomic_set_gbit(loadsat._bitarray, loadids[tid]);
	};
	size_t grid_size, block_size; make_kernel_param(&grid_size, &block_size, n_loadnodes(), 512);
	traverse_noret << <grid_size, block_size >> > (n_loadnodes(), set_loadbit);
	cudaDeviceSynchronize(); cuda_error_check;
	// compute sat on host and upload back to device
	{
		std::vector<unsigned int> loadbits(nbitword);
		cudaMemcpy(loadbits.data(), vid2loadid._bitarray, sizeof(unsigned int) * nbitword, cudaMemcpyDeviceToHost);
		grid::BitSAT<unsigned int> hostsat(loadbits);
		cudaMemcpy(const_cast<int*>(vid2loadid._chunksat), hostsat._chunkSat.data(), sizeof(int) * hostsat._chunkSat.size(), cudaMemcpyHostToDevice);
	}
	// finished SAT computation

	// DEBUG check tangent and normal 
}

void uploadLoadForce(double const* const fhost[3])
{
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&fload[i], sizeof(double) * getLoadNodes().size());
		cudaMemcpy(fload[i], fhost[i], sizeof(double) * getLoadNodes().size(), cudaMemcpyHostToDevice);
	}
}

//void displacementProject(double* u_dev[3])
//{
//	printf("\033[31mDisplacement Projection is deprecated on device\033[0m\n");
//	throw std::runtime_error("Deprecated branch");
//	// DEBUG
//	//grids[0]->v3_toMatlab("u_dev", u_dev);
//
//	double a = 1, b = 0;
//	for (int i = 0; i < 3; i++) {
//		sta = cublasDgemv(cublas_handle, CUBLAS_OP_T, _n_gsnodes, 6, &a, _Rdata[i], _n_gsnodes, u_dev[i], 1, &b, _Rudata[i], 1);
//		check_cublas(sta);
//	}
//
//	sta = cublasDaxpy(cublas_handle, 6, &a, _Rudata[1], 1, _Rudata[0], 1);
//	check_cublas(sta);
//
//	sta = cublasDaxpy(cublas_handle, 6, &a, _Rudata[2], 1, _Rudata[0], 1);
//	check_cublas(sta);
//
//	a = -1; b = 1;
//	for (int i = 0; i < 3; i++) {
//		sta = cublasDgemv(cublas_handle, CUBLAS_OP_N, _n_gsnodes, 6, &a, _Rdata[i], _n_gsnodes, _Rudata[0], 1, &b, u_dev[i], 1);
//		check_cublas(sta);
//	}
//	cuda_error_check;
//
//	// DEBUG
//	//grids[0]->v3_toMatlab("up_dev", u_dev);
//}

void uploadRigidDisplacement(double* udst[3], int k) {
	for (int i = 0; i < 3; i++) {
		cudaMemcpy(udst[i], _Rdata[i] + k * _n_gsnodes, sizeof(double) * _n_gsnodes, cudaMemcpyDeviceToDevice);
	}
}

void forceProject(double* f_dev[3])
{
	// DEBUG
	//grids[0]->v3_toMatlab("f_dev", f_dev);
	bool freeforce = grids.isForceFree();

	double* fsupport[3];
	Grid::getTempBufArray(fsupport, 3, n_loadnodes());

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_loadnodes(), 512);

	devArray_t<double*, 3> fs{ fsupport[0],fsupport[1],fsupport[2] };
	devArray_t<double*, 3> f{ f_dev[0],f_dev[1],f_dev[2] };

	int* loadnodes_g = gloadnodes;
	const int* lex2gs_g = _vlex2gs_dev;

	// gather forces on load nodes
	auto kernel = [=] __device__(int tid) {
		int vload = loadnodes_g[tid];
		//int vgs = lex2gs_g[vload];
		int vgs = vload;
		for (int i = 0; i < 3; i++) {
			fs[i][tid] = f[i][vgs];
		}
	};

	traverse_noret << <grid_size, block_size >> > (n_loadnodes(), kernel);
	cudaDeviceSynchronize();
	cuda_error_check;

	// copy f support to host
	std::vector<double> fshost[3];
	for (int i = 0; i < 3; i++) {
		fshost[i].resize(n_loadnodes());
		cudaMemcpy(fshost[i].data(), fs[i], sizeof(double) * n_loadnodes(), cudaMemcpyDeviceToHost);
	}

	cuda_error_check;

	// project f support on host
	forceProject(fshost);

	// upload projected f support to devce
	for (int i = 0; i < 3; i++) cudaMemcpy(fs[i], fshost[i].data(), sizeof(double)* fshost[i].size(), cudaMemcpyHostToDevice);

	// clear force on nodes outside load region (reset all force to zero)
	for (int i = 0; i < 3; i++)  init_array(f[i], double{ 0 }, _n_gsnodes);

	// replace projected support force to original force vector
	auto kernel_upload = [=] __device__(int tid) {
		int vload = loadnodes_g[tid];
		//int vgs = lex2gs_g[vload];
		int vgs = vload;
		for (int i = 0; i < 3; i++) {
			f[i][vgs] = fs[i][tid];
		}
	};
	
	traverse_noret << <grid_size, block_size >> > (n_loadnodes(), kernel_upload);
	cudaDeviceSynchronize();
	cuda_error_check;

	// DEBUG
	//grids[0]->v3_toMatlab("fp_dev", f_dev);
}

// compute N^T * N * f or N * f
void forceProjectComplementary(double* f_dev[3], bool Ncoords)
{
	bool freeforce = grids.isForceFree();

	double* fsupport[4];
	Grid::getTempBufArray(fsupport, 4, n_loadnodes());

	// extract support force
	getForceSupport(f_dev, fsupport);

	// project support force to tangent space
	devArray_t<double*, 3> v0{ gvtangent[0][0],gvtangent[0][1],gvtangent[0][2] };
	devArray_t<double*, 3> v1{ gvtangent[1][0],gvtangent[1][1],gvtangent[1][2] };
	devArray_t<double*, 3> fs{ fsupport[0],fsupport[1],fsupport[2] };
	auto kernel = [=] __device__(int tid) {
		if (freeforce) {
			fs[0][tid] = 0; fs[1][tid] = 0; fs[2][tid] = 0;
			return;
		}
		double w0 = v0[0][tid] * fs[0][tid] + v0[1][tid] * fs[1][tid] + v0[2][tid] * fs[2][tid];
		double w1 = v1[0][tid] * fs[0][tid] + v1[1][tid] * fs[1][tid] + v1[2][tid] * fs[2][tid];
		if (Ncoords) {
			fs[0][tid] = w0; fs[1][tid] = w1; fs[2][tid] = 0;
		}
		else {
			for (int i = 0; i < 3; i++) {
				fs[i][tid] = w0 * v0[i][tid] + w1 * v1[i][tid];
			}
		}
	};
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_loadnodes(), 512);
	traverse_noret << <grid_size, block_size >> > (n_loadnodes(), kernel);
	cudaDeviceSynchronize();
	cuda_error_check;

	// substitute support force in original force 
	setForceSupport(fsupport, f_dev);
}

void forceRestoreProjection(double* f_dev[3])
{
	bool freeforce = grids.isForceFree();
	double* fsupport[4];
	Grid::getTempBufArray(fsupport, 4, n_loadnodes());

	// extract support force
	getForceSupport(f_dev, fsupport);

	// restore support force from projection coordinates
	devArray_t<double*, 3> v0{ gvtangent[0][0],gvtangent[0][1],gvtangent[0][2] };
	devArray_t<double*, 3> v1{ gvtangent[1][0],gvtangent[1][1],gvtangent[1][2] };
	devArray_t<double*, 3> fs{ fsupport[0],fsupport[1],fsupport[2] };
	auto restorekernel = [=] __device__(int tid) {
		double w0 = fs[0][tid] ;
		double w1 = fs[1][tid];
		if (freeforce) {
			for (int i = 0; i < 3; i++) fs[i][tid] = 0;
		}
		else {
			for (int i = 0; i < 3; i++)  fs[i][tid] = w0 * v0[i][tid] + w1 * v1[i][tid];
		}
	};
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_loadnodes(), 512);
	traverse_noret << <grid_size, block_size >> > (n_loadnodes(), restorekernel);
	cudaDeviceSynchronize();
	cuda_error_check;	

	// substitute support force in original force 
	setForceSupport(fsupport, f_dev);
}

void getForceSupport(double const * const f_dev[3], double* fsup[3])
{
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_loadnodes(), 512);

	devArray_t<double*, 3> fs{ fsup[0],fsup[1],fsup[2] };
	devArray_t<const double*, 3> f{ f_dev[0],f_dev[1],f_dev[2] };

	int* loadnodes_g = gloadnodes;
	const int* lex2gs_g = _vlex2gs_dev;

	// gather forces on load nodes
	auto kernel = [=] __device__(int tid) {
		int vload = loadnodes_g[tid];
		//int vgs = lex2gs_g[vload];
		int vgs = vload;
		for (int i = 0; i < 3; i++) {
			fs[i][tid] = f[i][vgs];
		}
	};

	traverse_noret << <grid_size, block_size >> > (n_loadnodes(), kernel);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void getForceSupport(double const * const f_dev[3], std::vector<double> fs[3])
{
	double* fsupport[3];
	Grid::getTempBufArray(fsupport, 3, n_loadnodes());
	getForceSupport(f_dev, fsupport);
	for (int i = 0; i < 3; i++) {
		fs[i].resize(n_loadnodes());
		cudaMemcpy(fs[i].data(), fsupport[i], sizeof(double) * n_loadnodes(), cudaMemcpyDeviceToHost);
	}
}

void setForceSupport(double const * const fsup[3], double* f_dev[3])
{
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_loadnodes(), 512);

	devArray_t<const double*, 3> fs{ fsup[0],fsup[1],fsup[2] };
	devArray_t<double*, 3> f{ f_dev[0],f_dev[1],f_dev[2] };

	int* loadnodes_g = gloadnodes;
	const int* lex2gs_g = _vlex2gs_dev;

	// gather forces on load nodes
	auto kernel = [=] __device__(int tid) {
		int vload = loadnodes_g[tid];
		//int vgs = lex2gs_g[vload];
		int vgs = vload;
		for (int i = 0; i < 3; i++) {
			f[i][vgs] = fs[i][tid];
		}
	};

	traverse_noret << <grid_size, block_size >> > (n_loadnodes(), kernel);
	cudaDeviceSynchronize();
	cuda_error_check;

}

void cleanProjection(void)
{
	sta = cublasDestroy(cublas_handle);
	check_cublas(sta);

	for (int i = 0; i < 3; i++) {
		if (_Rdata[i] != nullptr) cudaFree(_Rdata[i]);
		if (_Rudata[i] != nullptr) cudaFree(_Rudata[i]);
		//if (gloadnodes != nullptr) cudaFree(gloadnodes);
	}

	cudaFree(gloadnodes);
	
	cuda_error_check;
}

double** getPreloadForce(void)
{
	return fload;
}

double** getForceNormal(void)
{
	return gvnormal;
}

size_t projectionGetMem(void)
{
	size_t memsize = 0;
	if (!grids.hasSupport()) {
		memsize += _n_gsnodes * 3 * sizeof(double) * 6;
	}

	// gvtangent
	memsize += n_loadnodes() * 3 * 2 * sizeof(double);

	// gvnormal
	memsize += n_loadnodes() * 3 * sizeof(double);

	// gloadnodse
	memsize += n_loadnodes() * sizeof(int);

	return memsize;
}


