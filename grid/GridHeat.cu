#include "Grid.h"
//#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "templateMatrix.h"
#include "lib.cuh"
#include "projection.h"
#include "tictoc.h"

#define DIRICHLET_DIAGONAL_WEIGHT 1e6f
//#define DIRICHLET_DIAGONAL_WEIGHT 1

using namespace grid;
using namespace culib;

extern __constant__ double gTemplateMatrix[24][24];
extern __constant__ double gHeatMatrix[8][8];
extern __constant__ int* gV2E[8];
extern __constant__ int* gV2Vfine[27];
extern __constant__ int* gV2Vcoarse[8];
extern __constant__ int* gV2V[27];
extern __constant__ int* gVfine2Vfine[27];
extern __constant__ int* gV2VfineC[64];// vertex to fine grid element center 
extern __constant__ int* gVfine2Efine[8];
extern __constant__ int* gVfine2Effine[8];
extern __constant__ float power_penalty[1];
extern __constant__ double* gU[3];
extern __constant__ double* gF[3];
extern __constant__ double* gR[3];
extern __constant__ double* gUworst[3];
extern __constant__ double* gFworst[3];
extern __constant__ double* gRfine[3];
extern __constant__ double* gUcoarse[3];
extern __constant__ int gUpCoarse[3];
extern __constant__ int gDownCoarse[3];
extern __constant__ int gGS_num[8];
extern __constant__ int gmode[1];
extern __constant__ int* gVflag[1];
extern __constant__ int* gEflag[1];
extern __constant__ int gLayerid[1];
extern __constant__ int gDEBUG[1];
extern __constant__ ScalarT* gT;
extern __constant__ ScalarT* gFT;
extern __constant__ ScalarT* gRTfine;
extern __constant__ ScalarT* gTcoarse;
extern __constant__ ScalarT* gRT;

extern __constant__ double* gLoadtangent[2][3];
extern __constant__ double* gLoadnormal[3];

template<typename T>
__device__ void loadTemplateMatrix(volatile T KE[8][8]) {
	int nbase = 0;
	while(nbase + threadIdx.x < 64){
		int row = (nbase + threadIdx.x) / 8;
		int col = (nbase + threadIdx.x) % 8;
		KE[row][col] = gHeatMatrix[row][col];
		nbase += blockDim.x;
	}
	__syncthreads();
}


template<typename T, int BlockSize = 32 * 13>
__global__ void gs_relax_heat_kernel(int n_vgstotal, int nv_gsset, devArray_t<T, 27> tstencil, int gs_offset) {
	// GraftArray<double, 27, 9> stencil(rxstencil, n_vgstotal);
    auto& stencil = tstencil;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ double sumAT[1][13][32];
	int warpId = threadIdx.x / 32;
	int node_id_in_block = threadIdx.x % 32;
	int workId = node_id_in_block;
	int gs_vertex_id = blockIdx.x * 32 + node_id_in_block;

	int offset = gs_offset;

	ScalarT AT = 0;

	int node_id;

	int flag;

	bool invalid_node = true;

	if (gs_vertex_id < nv_gsset) {
		node_id = offset + gs_vertex_id;
		flag = gVflag[0][node_id];
		invalid_node = flag & Grid::Bitmask::mask_invalid;
		if (invalid_node) goto _blockSum;
		for (auto i : { 0,14 }) {
			double uT;
			int neigh_th = warpId + i;
			int neigh = gV2V[neigh_th][node_id];	
			if (neigh == -1) continue;
			uT= gT[neigh];
            AT += stencil[neigh_th][node_id] * uT;
		}
	}

_blockSum:

    sumAT[0][warpId][node_id_in_block] = AT;

    __syncthreads();

	// gather all part
	if (warpId < 7) {
        int addId = warpId + 7;
        if (addId < 13) {
            sumAT[0][warpId][node_id_in_block] += sumAT[0][addId][node_id_in_block];
        }
    }
	__syncthreads();
	if (warpId < 4) {
        int addId = warpId + 4;
        if (addId < 7) {
            sumAT[0][warpId][node_id_in_block] += sumAT[0][addId][node_id_in_block];
        }
    }
	__syncthreads();
	if (warpId < 2) {
        int addId = warpId + 2;
        sumAT[0][warpId][node_id_in_block] += sumAT[0][addId][node_id_in_block];
    }
	__syncthreads();
	if (warpId < 1) {
        int addId = warpId + 1;
        AT = sumAT[0][warpId][node_id_in_block] + sumAT[0][addId][node_id_in_block];
    }
	//__syncthreads();

	if (gs_vertex_id < nv_gsset && !invalid_node) {
		double node_sum = 0;
        ScalarT uT = 0.; int rowOffset = 0;
        if (warpId == 0) {
			uT = gT[node_id];
			uT = (gFT[node_id] - AT) / stencil[13][node_id];
			gT[node_id] = uT;
		}
	}
}

// map 32 vertices to 8 warp, each warp use specific neighbor element (density rho_i)
template<int BlockSize = 32 * 8>
__global__ void gs_relax_Heat_OTFA_kernel(int nv_gs, int gs_offset, float* rholist) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ ScalarT KE[8][8];

	__shared__ ScalarT sumKeU[4][32];

	__shared__ ScalarT sumS[4][32];

	// load template matrix from constant memory to shared memory
	loadTemplateMatrix(KE);

	int warpId = threadIdx.x / 32;
	int warpTid = threadIdx.x % 32;

	ScalarT KT = { 0. };
	ScalarT S  = { 0. };

	bool invalid_node = false;
	// the id in a gs subset
	int vid = blockIdx.x * 32 + warpTid;

	// the id in total node set
	vid += gs_offset;

	int vi = 7 - warpId;
	ScalarT heatRho = 0;
	int eid;

	int flag = gVflag[0][vid];
	invalid_node |= flag & Grid::Bitmask::mask_invalid;

	if (invalid_node) goto _blocksum;

	eid = gV2E[warpId][vid];

	if (eid != -1)
		heatRho = rholist[eid];
	else
		goto _blocksum;

	if (gV2V[13][vid] == -1) {
		invalid_node = true;
		goto _blocksum;
	}

	// compute KU and S 
	for (int vj = 0; vj < 8; vj++) {
		// vjpos = epos + vjoffset
		int vjpos[3] = {
			vj % 2 + warpId % 2,
			vj % 4 / 2 + warpId % 4 / 2,
			vj / 4 + warpId / 4
		};
		int vj_lid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
		int vj_vid = gV2V[vj_lid][vid];
		if (vj_vid == -1) continue;
        double U = {gT[vj_vid]};
        if (vj_lid != 13) {
            KT += heatRho * KE[vi][vj] * U;
        }
        if (vj_lid == 13) {
            S = heatRho * KE[vi][vi];
        }
    }

_blocksum:

	if (warpId >= 4) {
        sumKeU[warpId - 4][warpTid] = KT;
        sumS[warpId - 4][warpTid] = S;
    }
	__syncthreads();

	if (warpId < 4) {
        sumKeU[warpId][warpTid] += KT;
        sumS[warpId][warpTid] += S;
    }
	__syncthreads();

	if (warpId < 2) {
        sumKeU[warpId][warpTid] += sumKeU[warpId + 2][warpTid];
        sumS[warpId][warpTid] += sumS[warpId + 2][warpTid];
    }
	__syncthreads();

	if (warpId < 1 && !invalid_node) {
        KT = sumKeU[0][warpTid] + sumKeU[1][warpTid];
        S = sumS[0][warpTid] + sumS[1][warpTid];

        ScalarT newT /*= {gUT[vid]}*/;
		newT = (gFT[vid] - KT) / S;
		if (flag & Grid::mask_sink_nodes) newT = 0;
		gT[vid] = newT; 
	}


}

void Grid::gs_relax_heat(int n_times)
{
	if (is_dummy()) return;
	use_grid();
	cuda_error_check;
	if (_layer == 0) {
		for (int n = 0; n < n_times; n++) {
			int gs_offset = 0;
			for (int i = 0; i < 8; i++) {
				constexpr int BlockSize = 32 * 8;
				size_t grid_size, block_size;
				make_kernel_param(&grid_size, &block_size, gs_num[i] * 8, BlockSize);
                gs_relax_Heat_OTFA_kernel<<<grid_size, block_size>>>(gs_num[i], gs_offset, _gbuf.rho_e);
                cudaDeviceSynchronize();
                cuda_error_check;
				gs_offset += gs_num[i];
			}
			cudaDeviceSynchronize();
			cuda_error_check;
		}
	}
	else {
		check_array_len(_gbuf.rxStencil, 27 * 9 * n_gsvertices);
        devArray_t<ScalarT *, 27> st;
        for (int i = 0; i < 27; i++) { st[i] = _gbuf.tStencil[i]; }

        for (int n = 0; n < n_times; n++) {
			int gs_offset = 0;
            for (int i = 0; i < 8; i++) {
				size_t grid_size, block_size;
				constexpr int BlockSize = 32 * 13;
				make_kernel_param(&grid_size, &block_size, gs_num[i] * 13, BlockSize);
				gs_relax_heat_kernel << <grid_size, block_size >> > (n_gsvertices, gs_num[i], st, gs_offset);
				cudaDeviceSynchronize();
				cuda_error_check;
				gs_offset += gs_num[i];
			}
			cudaDeviceSynchronize();
			cuda_error_check;
		}
	}
}


__global__ void update_heat_residual_OTFA_kernel(int nv, float* rholist) {

	__shared__ double KE[8][8];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	loadTemplateMatrix(KE);

	if (tid >= nv) return;

	int vid = tid;

	// add fixed flag check
	bool vsink[27], vload[27];
	int v2v[27];
	for (int i = 0; i < 27; i++) {
		v2v[i] = gV2V[i][vid];
		if (v2v[i] != -1) {
			int flag = gVflag[0][v2v[i]];
			vsink[i] = flag & grid::Grid::mask_sink_nodes;
			vload[i] = flag & grid::Grid::Bitmask::mask_loadnodes;
		}
	}

	double KT = 0.;
	for (int i = 0; i < 8; i++) {
		int eid = gV2E[i][vid];
		if (eid == -1) continue;
		double heatRho = rholist[eid];
		int vi = 7 - i;
		for (int vj = 0; vj < 8; vj++) {
			int vjpos[3] = {
				vj % 2 + i % 2,
				vj % 4 / 2 + i % 4 / 2,
				vj / 4 + i / 4
			};
			int vj_lid = vjpos[0] + vjpos[1] * 3 + vjpos[2] * 9;
			int vj_vid = v2v[vj_lid];
			if (vj_vid == -1) {
				// DEBUG
				printf("-- error in update residual otfa\n");
				continue;
			}
			double t = gT[vj_vid];
			if (vsink[vj_lid]) { t = 0; }
			KT += heatRho * KE[vi][vj] * t;
		}
	}

	double r = gFT[vid] - KT;

	if (vsink[13]) { r = 0; }

	gRT[vid] = r;
}

template <typename T>
__global__ void update_heat_residual_kernel(int nv, devArray_t<T *, 27> rxstencil) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) return;
	int vid = tid;

	double KT = { 0. };
	for (int i = 0; i < 27; i++) {
		int vj = gV2V[i][vid];
		if (vj == -1) continue;
		double t = gT[vj];
		KT += rxstencil[i][vid] * t;
	}

	gRT[vid] = gFT[vid] - KT;
}

__global__ void heatDiffusionStep_kernel(int nv, devArray_t<ScalarT *, 27> st ){
    
}

void grid::Grid::heatDiffusionStep(void) {
}

void grid::Grid::update_heat_residual(void) {
	if (is_dummy()) return;
	use_grid();
	size_t grid_size, block_size;
	if (_layer == 0)
	{
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 256);
		update_heat_residual_OTFA_kernel<<<grid_size, block_size>>>(n_gsvertices, _gbuf.rho_e);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else
	{
		make_kernel_param(&grid_size, &block_size, n_gsvertices * 13, 32 * 13);
		devArray_t<float*, 27> stencil;
		for (int i = 0; i < 27; i++) {
			stencil[i] = _gbuf.tStencil[i];
		}
		update_heat_residual_kernel<<<grid_size, block_size>>>(n_gsvertices, stencil);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}


__global__ void restrict_heat_residual_kernel(int nv) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= nv) return;

	ScalarT res = { 0.f };

	{
		int neigh = gV2Vfine[13][tid];
		if (neigh != -1) {
			res += gRTfine[neigh];
		}
	}

	// volume vertex
	for (int j : {0, 2, 6, 8, 18, 20, 24, 26}) {
		int neigh = gV2Vfine[j][tid];
		if (neigh != -1) {
			res += gRTfine[neigh] * (1.0f / 8);
		}
	}
	// face center
	for (int j : {4, 10, 12, 14, 16, 22}) {
		int neigh = gV2Vfine[j][tid];
		if (neigh != -1) {
			res += gRTfine[neigh] * (1.0f / 2);
		}
	}
	// edge center
	for (int j : {1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25}) {
		int neigh = gV2Vfine[j][tid];
		if (neigh != -1) {
			res += gRTfine[neigh] * (1.0f / 4);
		}
	}

__writeResidual:

	gFT[tid] = res;
}


__global__ void restrict_heat_residual_nondyadic_kernel(int nv) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ ScalarT W[4][4][4];
	__shared__ int* vfine2vfine[27];

	if (threadIdx.x < 64) {
		int k = threadIdx.x % 4;
		int j = threadIdx.x / 4 % 4;
		int i = threadIdx.x / 16;
		W[i][j][k] = ((4 - i)*(4 - j)*(4 - k)) / 64.0;
		if (threadIdx.x < 27) {
			vfine2vfine[threadIdx.x] = gVfine2Vfine[threadIdx.x];
		}
	}
	__syncthreads();

	if (tid >= nv) return;

	int vid = tid;

	int aFlag[(7 * 7 * 7) / (sizeof(int) * 8) + 1] = { 0 };

	ScalarT sumR = 0;

	auto* rfine =  gRTfine;

	for (int i = 0; i < 64; i++) {
		int vff = gV2VfineC[i][vid];
		if (vff == -1) continue;
		int basepos[3] = { i % 4 * 2 - 3,i % 16 / 4 * 2 - 3,i / 16 * 2 - 3 };
		for (int dx = -1; dx <= 1; dx++) {
			int xj = basepos[0] + dx;
			if (xj <= -4 || xj >= 4) continue;
			for (int dy = -1; dy <= 1; dy++) {
				int yj = basepos[1] + dy;
				if (yj <= -4 || yj >= 4) continue;
				for (int dz = -1; dz <= 1; dz++) {
					int zj = basepos[2] + dz;
					if (zj <= -4 || zj >= 4) continue;
					int jid = xj + 3 + (yj + 3) * 7 + (zj + 3) * 49;
					if (read_gbit(aFlag, jid)) continue;
					set_gbit(aFlag, jid);
					int djid = (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9;
					int vj_vid = vfine2vfine[djid][vff];
					if (vj_vid == -1) continue;
					ScalarT r = {rfine[vj_vid]};
					ScalarT weight = W[abs(xj)][abs(yj)][abs(zj)];
					sumR += weight * r;
				}
			}
		}
	}

	gFT[vid] = sumR;
}


void grid::Grid::restrict_heat_residual(void)
{
	use_grid();
	size_t grid_size, block_size;
	if (_layer == 2 && is_skip()) {
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 256);
		restrict_heat_residual_nondyadic_kernel << <grid_size, block_size >> > (n_gsvertices);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else if (_layer == 0) {
		throw std::runtime_error("Cannot restrict residual to finest layer");
	}
	else {
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
		restrict_heat_residual_kernel << <grid_size, block_size >> > (n_gsvertices);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}

__global__ void prolongate_heat_correction_kernel(int nv) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid >= nv) return;

	int vid = tid;

	ScalarT c = { 0. };

	ScalarT *pT = gTcoarse;

	int flag = gVflag[0][vid];
	if (flag& Grid::Bitmask::mask_invalid) return;

	int posInE[3] = {
		((flag & Grid::Bitmask::mask_xmod7) >> Grid::Bitmask::offset_xmod7) % 2,
		((flag & Grid::Bitmask::mask_ymod7) >> Grid::Bitmask::offset_ymod7) % 2,
		((flag & Grid::Bitmask::mask_zmod7) >> Grid::Bitmask::offset_zmod7) % 2
	};

	for (int i = 0; i < 8; i++) {
		int vcoarsepos[3] = { i % 2 * 2, i % 4 / 2 * 2, i / 4 * 2 };
		int wpos[3] = { abs(vcoarsepos[0] - posInE[0]), abs(vcoarsepos[1] - posInE[1]), abs(vcoarsepos[2] - posInE[2]) };
		if (wpos[0] >= 2 || wpos[1] >= 2 || wpos[2] >= 2) continue;
		ScalarT weight = (2 - wpos[0]) * (2 - wpos[1]) * (2 - wpos[2]) / 8.;
		int vcoarseid = gV2Vcoarse[i][vid];
		if (vcoarseid == -1) continue;
		c += weight * pT[vcoarseid];
	}

	if (flag & Grid::mask_sink_nodes) c = 0;

	gT[vid] += c;
}


__global__ void prolongate_heat_correction_nondyadic_kernel(int nv, int* vbitflag) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= nv) return;

	int vid = tid;

	ScalarT c = { 0. };

	int flagword = vbitflag[vid];

	if (flagword & Grid::Bitmask::mask_invalid) return;

	int posInE[3] = {
		(flagword & Grid::Bitmask::mask_xmod7) >> Grid::Bitmask::offset_xmod7,
		(flagword & Grid::Bitmask::mask_ymod7) >> Grid::Bitmask::offset_ymod7,
		(flagword & Grid::Bitmask::mask_zmod7) >> Grid::Bitmask::offset_zmod7
	};
	for (int i = 0; i < 3; i++) posInE[i] %= 4;

	int nei_counter = 0;
	// traverse vertex of coarse element which contains this fine vertex
	for (int i = 0; i < 8; i++) {
		int finepos[3] = { i % 2 * 4, i % 4 / 2 * 4, i / 4 * 4 };
		int wpos[3] = { abs(finepos[0] - posInE[0]), abs(finepos[1] - posInE[1]), abs(finepos[2] - posInE[2]) };
		if (wpos[0] >= 4 || wpos[1] >= 4 || wpos[2] >= 4) continue;
		ScalarT weight = (4 - wpos[0]) * (4 - wpos[1]) * (4 - wpos[2]) / 64.0;
		int coarseid = gV2Vcoarse[i][vid];
		if (coarseid == -1) continue;
		c += weight * gTcoarse[coarseid];
	}

	if (flagword & Grid::mask_sink_nodes) c = 0;

	gT[vid] += c;
}

void grid::Grid::prolongate_heat_correction(void)
{
	if (is_dummy()) return;
	use_grid();
	size_t grid_size, block_size;
	if (_layer == 0 && is_skip()) {
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
		prolongate_heat_correction_nondyadic_kernel << <grid_size, block_size >> > (n_gsvertices, _gbuf.vBitflag);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else {
		make_kernel_param(&grid_size, &block_size, n_gsvertices, 512);
		prolongate_heat_correction_kernel << <grid_size, block_size >> > (n_gsvertices);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}


double HierarchyGrid::v_cycle_heat(int pre_relax, int post_relax)
{
	int depth = n_grid() - 1;
	// downside
	for (int i = 0; i < depth + 1; i++) {
		if (_gridlayer[i]->is_dummy()) { continue; }
		if (i > 0) {
			//_gridlayer[i]->stencil2matlab("rxcoarse");
			_gridlayer[i]->fineGrid->update_heat_residual();
			//_gridlayer[i]->fineGrid->residual2matlab("rfine");
			_gridlayer[i]->restrict_heat_residual(); 
			//_gridlayer[i]->force2matlab("fcoarse");
			_gridlayer[i]->reset_heat_displacement();
		}
		if (i < n_grid() - 1) {
			_gridlayer[i]->gs_relax_heat(pre_relax);
			//_gridlayer[i]->displacement2matlab("u");
		}
		else {
			//_gridlayer[i]->force2matlab("f");
			_gridlayer[i]->solve_fem_host();
			//_gridlayer[i]->displacement2matlab("u");
		}
	}
	// DEBUG
	//_gridlayer[0]->displacement2matlab("u");
	// upside
	for (int i = depth - 1; i >= 0; i--) {
		if (_gridlayer[i]->is_dummy()) { continue; }
		//_gridlayer[i]->displacement2matlab("u");
		//_gridlayer[i]->update_residual();
		//printf("-- [%d] r = %lf%%\n", i, _gridlayer[i]->relative_residual() * 100);
		_gridlayer[i]->prolongate_heat_correction();
		//_gridlayer[i]->update_residual();
		//printf("-- [%d] rc=  %lf%%\n", i, _gridlayer[i]->relative_residual() * 100);
		//_gridlayer[i]->displacement2matlab("uc");
		//_gridlayer[i]->force2matlab("fc");
		_gridlayer[i]->gs_relax_heat(post_relax);
		//_gridlayer[i]->update_residual();
		//printf("-- [%d] rr=  %lf%%\n", i, _gridlayer[i]->relative_residual() * 100);
		//_gridlayer[i]->displacement2matlab("ur");
	}

	_gridlayer[0]->update_heat_residual();
	return _gridlayer[0]->relative_heat_residual();
}

void grid::Grid::reset_heat_displacement(void){
	cudaMemset(_gbuf.uT, 0, sizeof(ScalarT) * n_gsvertices);
}

void grid::Grid::reset_heat_residual(void){
	cudaMemset(_gbuf.rT, 0, sizeof(ScalarT) * n_gsvertices);
}

void grid::Grid::reset_heat_force(void){
	cudaMemset(_gbuf.fT, 0, sizeof(ScalarT) * n_gsvertices);
}

double grid::Grid::relative_heat_residual(void)
{
	double rel =
	 culib::norm(_gbuf.rT, n_gsvertices) / culib::norm(_gbuf.fT, n_gsvertices);
	return rel;
}


// on the fly assembly
template <typename T, int BlockSize = 32 * 9>
__global__ void restrict_heat_stencil_nondyadic_OTFA_kernel(
	int nv_coarse, devArray_t<T *, 27> rxCoarse, int nv_fine, float *rhofine, int *vfineflag, T dirich_weight)
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ T KE[8][8];
	__shared__ T W[4][4][4];

	// load template matrix from constant memory to shared memory
	loadTemplateMatrix(KE);

	// compute weight
	if (threadIdx.x < 64) {
		int i = threadIdx.x % 4;
		int j = threadIdx.x % 16 / 4;
		int k = threadIdx.x / 16;
		W[k][j][i] = (4 - i)*(4 - j)*(4 - k) / 64.f;
	}
	__syncthreads();
	
	// init coarseStencil
	T coarseStencil[27] = { 0. };

	int ke_id = tid / nv_coarse;

	int vid = tid % nv_coarse;

	if (ke_id >= 1) return;

	// traverse neighbor nodes of fine element center (which is the vertex on fine fine grid)
	for (int i = 0; i < 64; i++) {
		int i2[3] = { (i % 4) * 2 + 1 ,(i % 16 / 4) * 2 + 1 ,(i / 16) * 2 + 1 };

		// get fine element center vertex
		int vn = gV2VfineC[i][vid];

		if (vn == -1) continue;

		// should traverse 7x7x7 neigbor nodes, and sum their weighted stencil, to reduce bandwidth, we traverse 8x8x8 elements 
		// traverse the neighbor fine fine element of this vertex and assembly the element matrices
		for (int j = 0; j < 8; j++) {
			int efineid = gVfine2Efine[j][vn];

			if (efineid == -1) continue;

			float heatRho = rhofine[efineid];

			int epos[3] = { i2[0] + j % 2 - 1,i2[1] + j % 4 / 2 - 1,i2[2] + j / 4 - 1 };

			// prefecth the flag of eight vertex
			bool vsink[8];
			for (int k = 0; k < 8; k++) {
				int vklid = j % 2 + k % 2 +
					(j / 2 % 2 + k / 2 % 2) * 3 +
					(j / 4 + k / 4) * 9;
				int vkvid = gVfine2Vfine[vklid][vn];
				if (vkvid == -1) printf("-- error in stencil restriction\n");
				int vkflag = vfineflag[vkvid];
				vsink[k] = vkflag & Grid::Bitmask::mask_sink_nodes;
			}

			// traverse the vertex of neighbor element (rows of element matrix), compute the weight on this vertex 
			for (int ki = 0; ki < 8; ki++) {
				int vipos[3] = { epos[0] + ki % 2,epos[1] + ki % 4 / 2,epos[2] + ki / 4 };
				int wipos[3] = { abs(vipos[0] - 4),abs(vipos[1] - 4),abs(vipos[2] - 4) };
				if (wipos[0] >= 4 || wipos[1] >= 4 || wipos[2] >= 4) continue;
				double wi = W[wipos[0]][wipos[1]][wipos[2]];
				double w_ki = wi * heatRho;

				// traverse another vertex of neighbor element (cols of element matrix), get the 3x3 Ke and multiply the row weights
				for (int kj = 0; kj < 8; kj++) {
					int kjpos[3] = { epos[0] + kj % 2 , epos[1] + kj % 4 / 2 , epos[2] + kj / 4 };
					double wk = w_ki * KE[ki][kj];

					if (vsink[kj] || vsink[ki]) {
						wk = 0;
						if (ki == kj) {
							wk = wi * dirich_weight;
						}
					}

					//  the weighted element matrix should split to coarse vertex, traverse the coarse vertices and split 3x3 Ke to coarse vertex by splitting weights
					for (int vsplit = 0; vsplit < 27; vsplit++) {
						int vsplitpos[3] = { vsplit % 3 * 4, vsplit % 9 / 3 * 4,vsplit / 9 * 4 };
						int wjpos[3] = { abs(vsplitpos[0] - kjpos[0]), abs(vsplitpos[1] - kjpos[1]), abs(vsplitpos[2] - kjpos[2]) };
						if (wjpos[0] >= 4 || wjpos[1] >= 4 || wjpos[2] >= 4) continue;
						T wkw = wk * W[wjpos[0]][wjpos[1]][wjpos[2]];
						coarseStencil[vsplit] += wkw;
					}
				}
			}
		}
	}

	for (int i = 0; i < 27; i++) {
		rxCoarse[i][vid] = coarseStencil[i];
	}
}

template<typename T, int BlockSize = 32 * 9>
__global__ void restrict_heat_stencil_dyadic_kernel(
	int nv_coarse, devArray_t<T*,27> rxCoarse, int nv_fine, devArray_t<T*, 27> rxFine) {
	size_t tid = blockDim.x*blockIdx.x + threadIdx.x;
	int ke_id = tid / nv_coarse;
	int vid = tid % nv_coarse;

	if (ke_id >= 1) return;

	double coarseStencil[27] = { 0. };

	int warpid = threadIdx.x / 32;
	int warptid = threadIdx.x % 32;

	double w[4] = { 1.0,1.0 / 2,1.0 / 4,1.0 / 8 };
	for (int i = 0; i < 27; i++) {
		int neipos[3] = { i % 3 + 1 ,i % 9 / 3 + 1 ,i / 9 + 1 };

		int wneighpos[3] = { abs(neipos[0] - 2),abs(neipos[1] - 2),abs(neipos[2] - 2) };

		if (wneighpos[0] >= 2 || wneighpos[1] >= 2 || wneighpos[2] >= 2) continue;

		double weight = w[wneighpos[0] + wneighpos[1] + wneighpos[2]];

		int vn = gV2Vfine[i][vid];

		if (vn == -1) continue;

		// traverse fine stencil component (each neighbor vertex has a component)
		for (int j = 0; j < 27; j++) {

			double kij = rxFine[j][vn] * weight;

			// DEBUG
			if (gVfine2Vfine[j][vn] == -1) { if (kij != 0) { printf("-- error on stencil 1\n"); } continue; }

			int vjpos[3] = { neipos[0] + j % 3 - 1 ,neipos[1] + j % 9 / 3 - 1 ,neipos[2] + j / 9 - 1 };

			// traverse coarse vertices to scatter the stencil component to them
			for (int vsplit = 0; vsplit < 27; vsplit++) {
				int vsplitpos[3] = { vsplit % 3 * 2, vsplit % 9 / 3 * 2, vsplit / 9 * 2 };
				int wsplitpos[3] = { abs(vsplitpos[0] - vjpos[0]), abs(vsplitpos[1] - vjpos[1]), abs(vsplitpos[2] - vjpos[2]) };
				if (wsplitpos[0] >= 2 || wsplitpos[1] >= 2 || wsplitpos[2] >= 2) continue;
				double wsplit = w[wsplitpos[0] + wsplitpos[1] + wsplitpos[2]];
				coarseStencil[vsplit] += wsplit * kij;
			}
		}
	}

	for (int i = 0; i < 27; i++) {
		rxCoarse[i][vid] = coarseStencil[i];
	}
}


void grid::HierarchyGrid::restrict_heat_stencil(grid::Grid &dstcoarse, grid::Grid &srcfine) {
	dstcoarse.use_grid();
	if (abs(srcfine._layer - dstcoarse._layer) > 1) {
		devArray_t<ScalarT*, 27> coarseStencil;
		for (int i = 0; i < 27; i++) {
			coarseStencil[i] = dstcoarse._gbuf.tStencil[i];
		}
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, dstcoarse.n_gsvertices, 256);
		restrict_heat_stencil_nondyadic_OTFA_kernel<<<grid_size, block_size>>>(
			dstcoarse.n_gsvertices, coarseStencil, srcfine.n_gsvertices, srcfine._gbuf.rho_e, srcfine._gbuf.vBitflag, ScalarT(1.));
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else {
		devArray_t<ScalarT *, 27> coarseStencil, fineStencil;
		for (int i = 0; i < 27; i++) {
			coarseStencil[i] = dstcoarse._gbuf.tStencil[i];
			fineStencil[i] = srcfine._gbuf.tStencil[i];
		}
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, dstcoarse.n_gsvertices, 256);
		restrict_heat_stencil_dyadic_kernel<<<grid_size, block_size>>>(
			dstcoarse.n_gsvertices, coarseStencil, srcfine.n_gsvertices, fineStencil);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}