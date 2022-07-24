#include "test_utils.h"
#include "lib.cuh"

extern  __constant__  double gTemplateMatrix[24][24];
extern  __constant__ int* gV2E[8];
extern  __constant__ int* gV2Vfine[27];
extern  __constant__ int* gV2Vcoarse[8];
extern  __constant__ int* gV2V[27];
extern  __constant__ int* gVfine2Vfine[27];
extern  __constant__ int* gV2VfineC[64];// vertex to fine grid element center 
extern  __constant__ int* gVfine2Efine[8];
extern  __constant__ int* gVfine2Effine[8];
extern  __constant__ float power_penalty[1];
extern  __constant__ double* gU[3];
extern  __constant__ double* gF[3];
extern  __constant__ double* gR[3];
extern  __constant__ double* gUworst[3];
extern  __constant__ double* gFworst[3];
extern  __constant__ double* gRfine[3];
extern  __constant__ double* gUcoarse[3];
extern  __constant__ int gGS_num[8];
extern  __constant__ int gmode[1];
extern  __constant__ int* gVflag[1];
extern  __constant__ int* gEflag[1];
extern  __constant__ int gLayerid[1];
extern  __constant__ int gDEBUG[1];

void TestSuit::setDensity(float* newrho)
{
	cudaMemcpy(grids[0]->getRho(), newrho, sizeof(float) * grids[0]->n_rho(), cudaMemcpyDeviceToDevice);
	cuda_error_check;
}

void TestSuit::scaleVector(float* p_data, size_t len, float scale)
{
	array_t<float> vec_map(p_data, len);
	vec_map *= scale;
	cuda_error_check;
}


extern __device__ void loadTemplateMatrix(volatile double KE[24][24]);
//__device__ void loadTemplateMatrix(volatile double KE[24][24]) {
//	int i = threadIdx.x / 24;
//	int j = threadIdx.x % 24;
//	if (i < 24) {
//		KE[i][j] = gTemplateMatrix[i][j];
//	}
//	int nfill = blockDim.x;
//	while (nfill < 24 * 24) {
//		int kid = nfill + threadIdx.x;
//		i = kid / 24;
//		j = kid % 24;
//		if (i < 24) {
//			KE[i][j] = gTemplateMatrix[i][j];
//		}
//		nfill += blockDim.x;
//	}
//	__syncthreads();
//}

// map 32 vertices to 8 warp (1 block)
template<int BlockSize = 32 * 8>
__global__ void stressAndComplianceOnVertex_kernel(
	int nv, int* vgsid, float* rholist, double elen,
	glm::vec4* poffset, double* pc, double* pv,
	float E, float mv, float power_penal
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	 
	//__shared__ volatile double KE[24][24];

	//__shared__ volatile double SIGMA[6][4][32];

	__shared__ volatile double EPS[6][4][32];

	//initSharedMem(&SIGMA[0][0][0], sizeof(SIGMA) / sizeof(double));
	initSharedMem(&EPS[0][0][0], sizeof(EPS) / sizeof(double));
	//loadTemplateMatrix(KE);

	double epsi[6] = { 0. };

	int warpId = threadIdx.x / 32;
	int warpTid = threadIdx.x % 32;

	int id = (tid / BlockSize) * 32 + warpTid;

	int vid;
	float rho = 1;
	int eid;
	int nvpos[3];
	int nid;
	double u[3];
	int N[3];
	float r[3] = { 0. };
	float d[3] = { 0. };
	float invelen = 1.f / elen;
	glm::vec4 ploc;
	float oldd[3];

	if (id >= nv) {
		goto __blockSum;
	}

	vid = vgsid[id];

	if (vid == -1) goto __blockSum;
	
	// checkout inclusion element density
	eid = gV2E[7][vid];
	if (eid == -1) goto __blockSum;
#if 0
	rho = rholist[eid];
#endif

	nvpos[0] = 1 + warpId % 2; nvpos[1] = 1 + warpId / 2 % 2; nvpos[2] = 1 + warpId / 4;
	nid = nvpos[0] + nvpos[1] * 3 + nvpos[2] * 3 * 3;

	// relocate vid to its corresponding neighbor 
	vid = gV2V[nid][vid];

	ploc = poffset[id];

	u[0] = gU[0][vid]; u[1] = gU[1][vid]; u[2] = gU[2][vid];

	N[0] = warpId % 2; N[1] = warpId / 2 % 2; N[2] = warpId / 4;

	for (int i = 0; i < 3; i++) {
		if (!N[i]) {
			r[i] = (elen - ploc[i]) / elen;
			d[i] = -invelen;
		}
		else {
			r[i] = ploc[i] / elen;
			d[i] = invelen;
		}
	}

	oldd[0] = d[0]; oldd[1] = d[1]; oldd[2] = d[2];
	d[0] = oldd[0] * r[1]    * r[2];
	d[1] = r[0]    * oldd[1] * r[2];
	d[2] = r[0]    * r[1]    * oldd[2];


	/* N_i  
	    _ _ _ _ _ 
	   |d0  0   0 |
	   |0   d1  0 |
	   |0   0   d2|
	   |d1  d0  0 | 
	   |d2  0   d0|
	   |0   d2  d1|
	*/

	epsi[0] = d[0] * u[0];
	epsi[1] = d[1] * u[1];
	epsi[2] = d[2] * u[2];
	epsi[3] = d[1] * u[0] + d[0] * u[1];
	epsi[4] = d[2] * u[0] + d[0] * u[2];
	epsi[5] = d[2] * u[1] + d[1] * u[2];

__blockSum:
	if (warpId >= 4) {
		for (int i = 0; i < 6; i++) {
			EPS[i][warpId - 4][warpTid] = epsi[i];
		}
	}
	__syncthreads();

	if (warpId < 4) {
		for (int i = 0; i < 6; i++) {
			EPS[i][warpId][warpTid] += epsi[i];
		}
	}
	__syncthreads();
	
	if (warpId < 2) {
		for (int i = 0; i < 6; i++) {
			EPS[i][warpId][warpTid] += EPS[i][warpId + 2][warpTid];
		}
	}
	__syncthreads();

	if (warpId < 1 && eid != -1 && id < nv) {
		for (int i = 0; i < 6; i++) {
			epsi[i] = EPS[i][warpId][warpTid] + EPS[i][warpId + 1][warpTid];
		}
		// 1 - mu, mu, mu, 0, 0, 0,
		//	mu, 1 - mu, mu, 0, 0, 0,
		//	mu, mu, 1 - mu, 0, 0, 0,
		//	0, 0, 0, (1 - 2 * mu) / 2, 0, 0,
		//	0, 0, 0, 0, (1 - 2 * mu) / 2, 0,
		//	0, 0, 0, 0, 0, (1 - 2 * mu) / 2;
		//elastic_matrix /= (1 + poisson_ratio)*(1 - 2 * poisson_ratio);

		float s = E * powf(rho, power_penal) / ((1 + mv)*(1 - 2 * mv));
		float sigma[6];
		sigma[0] = (1 - mv)*epsi[0] + mv * epsi[1] + mv * epsi[2];
		sigma[1] = mv * epsi[0] + (1 - mv) * epsi[1] + mv * epsi[2];
		sigma[2] = mv * epsi[0] + mv * epsi[1] + (1 - mv) * epsi[2];
		float m = (1 - 2 * mv) / 2;
		sigma[3] = m * epsi[3];
		sigma[4] = m * epsi[4];
		sigma[5] = m * epsi[5];
		for (int i = 0; i < 6; i++) {
			sigma[i] *= s;
		}

		double c = 0, von = 0;
		for (int i = 0; i < 6; i++) {
			c += sigma[i] * epsi[i];
		}
		von = (double)powf(sigma[0] - sigma[1], 2) + powf(sigma[1] - sigma[2], 2) + powf(sigma[0] - sigma[2], 2) +
			6 * (sigma[3] * sigma[3] + sigma[4] * sigma[4] + sigma[5] * sigma[5]);
		von /= 2;
		von = sqrt(von);
		// DEBUG
		//printf("[%d] pc = %6.4e, pv = %6.4e\n", id, c, von);
		pc[id] = c;
		pv[id] = von;
	}

}

void stressAndComplianceOnVertex_impl(double elen, const std::vector<int>& vlexid, const std::vector<glm::vec4>& inclusionPos, std::vector<double>& clist, std::vector<double>& vonlist)
{
	// allocate buffer for inclusion element ids
	int* pvid;
	cudaMalloc(&pvid, sizeof(int) * vlexid.size());
	cudaMemcpy(pvid, vlexid.data(), sizeof(int) * vlexid.size(), cudaMemcpyHostToDevice);
	grids[0]->lexico2gsorder_g(nullptr, vlexid.size(), pvid, vlexid.size(), pvid, grids[0]->getVidmap());
	cuda_error_check;

	// allocate buffer inclusion offset in element
	glm::vec4* poffset;
	cudaMalloc(&poffset, sizeof(glm::vec4) * inclusionPos.size());
	cudaMemcpy(poffset, inclusionPos.data(), sizeof(glm::vec4) * inclusionPos.size(), cudaMemcpyHostToDevice);
	cuda_error_check;

	// allocate buffer to store stress and compliance
	double* pc, *pv;
	cudaMalloc(&pc, sizeof(double)* vlexid.size());
	cudaMalloc(&pv, sizeof(double) * vlexid.size());
	init_array(pc, 0., vlexid.size());
	init_array(pv, 0., vlexid.size());

	// get rholist
	float* rholist = grids[0]->getRho();

	float E = params.youngs_modulu;
	float mv = params.poisson_ratio;
	float p = params.power_penalty;
	// launch kernel to compute compliance and stress
	grids[0]->use_grid();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, vlexid.size() * 8, 32 * 8);
	stressAndComplianceOnVertex_kernel << <grid_size, block_size >> > ( vlexid.size(), pvid,  rholist, elen,
		poffset, pc, pv, E, mv, p);
	cudaDeviceSynchronize();
	cuda_error_check;

	clist.resize(vlexid.size());
	vonlist.resize(vlexid.size());

	cudaMemcpy(clist.data(), pc, sizeof(double) * vlexid.size(), cudaMemcpyDeviceToHost);
	cudaMemcpy(vonlist.data(), pv, sizeof(double) * vlexid.size(), cudaMemcpyDeviceToHost);

	cudaFree(pc);
	cudaFree(pv);
}



