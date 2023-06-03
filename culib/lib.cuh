#pragma once
#ifndef __LIB_CUH_H
#define __LIB_CUH_H

//#define USE_CUDA_CUB

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include<nvfunctional>
//#include "cuda/std/cstddef"
//#include "math.h"
#include"array"
#include<iostream>
//#include"Eigen/core"
//#include"tpmsTopopter_t.h"
//#include "gpu_deploy.cuh"
//#include "mytimer.h"
//#include "mycommon.h"
#include"curand.h"
#include "cudaCommon.cuh"

#ifdef USE_CUDA_CUB
#include "cub/cub.cuh"
#endif

typedef double Scaler;

//constexpr size_t node_num = 213 * 213 * 213;
//constexpr size_t node_task_num = 204;

extern void lib_test(void);
extern int get_device_info(void);
extern void use4Bytesbank(void);
extern void use8Bytesbank(void);
extern void init_cuda(void);
extern void* reserve_buf(size_t require);
extern void* get_libbuf(void);
extern size_t get_libbuf_size(void);

template <typename T,unsigned int blockSize>
__device__ void warpReduce(volatile T *sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <typename T, unsigned int blockSize>
__global__ void reduce(T *g_idata, T *g_odata, unsigned int n) {
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<typename Tin, int blockSize = 512, typename Tout = Tin>
__global__ void block_sum_kernel(const Tin* pdata, Tout* odata, size_t n) {
	__shared__ Tout sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockDim.x*blockIdx.x;
	Tout s = 0;
	// load data to block
	if (element_id < n) {
		s = pdata[element_id];
	}
	sdata[tid] = s;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	// use warpReduce to sum last 64 component 
	if (tid < 32) { warpReduce<Tout, blockSize>(sdata, tid); }
	if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<typename DType, int N>
struct devArray_t {
	DType _data[N];
	__host__ __device__ const DType& operator[](int k) const {
		return _data[k];
	}

	__host__ __device__  DType& operator[](int k) {
		return _data[k];
	}

	__host__ __device__ devArray_t<DType, N> operator-(const devArray_t<DType, N>& arr2) const {
		devArray_t<DType, N> darr;
		for (int i = 0; i < N; i++) {
			darr[i] = _data[i] - arr2[i];
		}
		return darr;
	}

	__host__ __device__ devArray_t<DType, N> operator+(const devArray_t<DType, N>& arr2) const {
		devArray_t<DType, N> arr;
		for (int i = 0; i < N; i++) {
			arr[i] = _data[i] + arr2[i];
		}
		return arr;
	}

	__host__ void destroy(void) {
		if (std::is_pointer<DType>::value) {
			for (int i = 0; i < N; i++) {
				cudaFree(_data[i]);
			}
		}
	}

	__host__ void create(int nelement) {
		if (std::is_pointer<DType>::value) {
			for (int i = 0; i < N; i++) {
				cudaMalloc(&_data[i], sizeof(DType) * nelement);
			}
		}
	}
};

template<typename DType, int N>
__host__ __device__ devArray_t<DType, N> operator*(DType s, const devArray_t<DType, N>& arr2) {
	devArray_t<DType, N> arr;
	for (int i = 0; i < N; i++) {
		arr[i] = s * arr2[i];
	}
	return arr;
}


template <int... Ns> struct nArgs { static constexpr int value = nArgs<Ns...>::value + 1; };

template <int N1, int... Ns> struct nArgs<N1, Ns...> { static constexpr int value = nArgs<Ns...>::value + 1; };

template<> struct nArgs<> { static constexpr int value = 0; };

template<int N1, int... Ns> struct MassProduct { static constexpr int value = N1 * MassProduct<Ns...>::value; };
template <int N1> struct MassProduct<N1> { static constexpr int value = N1; };

template<int N, int... Ns> struct FirstArg { static constexpr int value = N; };

template <int N1, int... Ns> struct LastArg { static constexpr int value = LastArg<Ns...>::value; };

template<int N> struct LastArg<N> { static constexpr int value = N; };


template<typename T, int... Ns> struct GraftArray;

template <typename T, int N1, int... Ns>
struct GraftArray<T, N1, Ns...>
{
	int _ldd;
	T* _ptr;
	__host__ __device__ GraftArray(T* pdata, int ldd) :_ptr(pdata), _ldd(ldd) {}
	//template<bool> inline GraftArray<T, Ns...> operator[](int i);

	static constexpr bool value = nArgs<Ns...>::value >= 1;
	template <bool Q = value, typename std::enable_if<Q, GraftArray<T, Ns...>>::type * = nullptr>
	__host__ __device__ inline GraftArray<T, Ns...> operator[](int i)
	{
		//typedef typename std::enable_if<Q::value, GraftArray<T, Ns...>>::type retType;
		return GraftArray<T, Ns...>(_ptr + i * _ldd * MassProduct<Ns...>::value, _ldd);
	}

	template <bool Q = value, typename std::enable_if<!Q, GraftArray<T>>::type * = nullptr>
	__host__ __device__ inline GraftArray<T> operator[](int i)
	{
		//typedef typename std::enable_if<Q::value, GraftArray<T>>::type retType;
		return GraftArray<T>(_ptr + i * _ldd);
	}

};

template <typename T>
struct GraftArray<T>
{
	T* _ptr;
	__host__ __device__ GraftArray(T *pdata) : _ptr(pdata) {}
	__host__ __device__ inline T& operator[](int i) {
		return _ptr[i];
	}
};

Scaler array_norm2(Scaler* dev_data/*, Scaler* host_data*/, int n, bool root = true);
__host__ void make_kernel_param(size_t* block_num, size_t* block_size, size_t num_tasks, size_t prefer_block_size);
__host__ void make_kernel_param(dim3& grid_dim, dim3& block_dim, const dim3& num_tasks, int prefer_block_size);
__host__ void show_cuSolver_version(void);

// by default weight_period is not too large and the block size is greater than weight period
//template<typename DType, int weight_period, int array_num, DType c>
//__global__ void add_weighted_array(DType* src, DType* addt, size_t n, DType* weight) {
//	int element_id = blockIdx.x*weight_period + threadIdx.x;
//	int tid = threadIdx.x;
//	if (tid < weight_period) {
//		DType ad = 0;
//#pragma unroll
//		for (int i = 0; i < array_num; i++)
//			ad += addt[element_id] * weight[tid];
//		src[element_id] += ad + c;
//	}
//}
//
//template<typename DType, int array_num>
//__global__ void add_weighted_array<DType, 0, array_num> (DType* src, DType* addt, size_t n, DType* weight) {
//	int id = blockIdx.x*blockDim.x + threadIdx.x;
//	if (id < n) {
//		DType ad = 0;
//#pragma unroll
//		for (int i = 0; i < array_num; i++) {
//			int array_id = id * array_num + i;
//			ad += addt[array_id] * weight[array_id];
//		}
//		src[id] += ad;
//	}
//}


//__global__ void kernel1(float* src, size_t n) {
//	size_t id = blockIdx.x*blockDim.x + threadIdx.x;
//	if (id < n) {
//		src[id] = 1;
//	}
//}
//
//__global__ void check1(float*src, size_t n) {
//	size_t id = blockIdx.x*blockDim.x + threadIdx.x;
//	if (id < n) {
//		if (src[id] != 1) {
//			printf("element %i is not 1!", id);
//		}
//	}
//}



/*
	this kernel computes, per-block, the sum of a block-sized portion of the input using a block-wide reduction
*/
template<class DType>
__global__ void block_sum(const DType *input,
	DType *per_block_results,
	const size_t n, Scaler* weight = nullptr)/* block sum should not bring a weight, since it is reduction sum, and weight will not reduce,do NOT use it in iterative call */
{
	extern __shared__ DType sdata[];

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// load input into __shared__ memory 
	DType x = 0;
	if (i < n)
	{
		if (weight == nullptr) {
			x = input[i];
		}
		else {
			x = input[i] * weight[i];
		}
	}
	sdata[threadIdx.x] = x;
	__syncthreads();// wait for sync

	// contiguous range pattern	
	for (int offset = blockDim.x / 2;
		offset > 0;
		offset >>= 1)
	{
		if (threadIdx.x < offset)//
		{
			// add a partial sum upstream to our own
			sdata[threadIdx.x] += sdata[threadIdx.x + offset];
		}
		/* wait until all threads in the block have  updated their partial sums */
		__syncthreads();
	}

	/* thread 0 writes the final result */
	if (threadIdx.x == 0)
	{
		per_block_results[blockIdx.x] = sdata[0];
	}
}

template<class DType>
DType array_sum_gpu(DType *dev_array, size_t  array_size, DType *dev_result, Scaler* weight = nullptr)
{
	//const size_t max_block_size  = 512;//	
	const size_t block_size = 512;//

	size_t rest_num = array_size;

	DType result;
	if (array_size == 0) return 0;
	if (array_size == 1) {
		cudaMemcpy(&result, dev_array, sizeof(DType), cudaMemcpyDeviceToHost);
		return result;
	}

	DType* tmp_buf;
	cudaMalloc(&tmp_buf, sizeof(DType)*((array_size + block_size - 1) / block_size + block_size - 1) / block_size);

	while (rest_num > 1) {
		size_t block_num = (rest_num + block_size - 1) / block_size;
		block_sum_kernel << <block_num, block_size >> > (dev_array, dev_result, rest_num);
		cudaDeviceSynchronize();
		if (rest_num == array_size) {
			dev_array = dev_result;
			dev_result = tmp_buf;
		}
		else {
			auto t = dev_array;
			dev_array = dev_result;
			dev_result = t;
		}
		rest_num = block_num;
	}

	cudaMemcpy(&result, dev_array, sizeof(DType), cudaMemcpyDeviceToHost);
	cudaFree(tmp_buf);
	return result;
}


template<typename T, unsigned int blockSize>
T reduce_sum(T *g_idata, T* g_odata, unsigned int n) {
	int rest = n;
	constexpr int num_multi_add = 1024;// large number cost less time and inaccuracy(float and double number)
	while (n > 1) {
		size_t block_need = (n + blockSize - 1) / blockSize;
		size_t grid_dim = (block_need + num_multi_add - 1) / num_multi_add;
		reduce<T,blockSize> <<<grid_dim, blockSize,sizeof(T)*blockSize >>> (g_idata, g_odata, n);
		cudaDeviceSynchronize();
		cuda_error_check;
		T* t = g_idata;
		g_idata = g_odata;
		g_odata = t;
		n = grid_dim;
	} 
	T sum;
	cudaMemcpy(&sum, g_idata, sizeof(T), cudaMemcpyDeviceToHost);
	return sum;
}

template<typename T>
__global__ void init_array_kernel(T* array, T value, int array_size) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < array_size) {
		array[tid] = value;
	}
}

template<typename T>
void init_array(T* dev_array, T value, int array_size) {
	size_t grid_dim;
	size_t block_dim;
	make_kernel_param(&grid_dim, &block_dim, array_size, 512);
	init_array_kernel<<<grid_dim, block_dim >>> (dev_array, value, array_size);
	cudaDeviceSynchronize();
	cuda_error_check;
}


template<typename T, typename Lam>
__global__ void map(T* g_data, Scaler* dst, int n, Lam func) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		dst[tid] = func(g_data[tid]);
	}
}

template<typename T, typename Lam>
__global__ void map(T* dst, int n, Lam func) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		dst[tid] = func(tid);
	}
}

template<typename Lam>
__global__ void map(int n,Lam func){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
	    func(tid);
	}
}

template<typename T>
__global__ void array_min(T* in1, T* in2, T* out, size_t n) {
	size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		T s1 = in1[tid];
		T s2 = in2[tid];
		out[tid] = s1 < s2 ? s1 : s2;
	}
}

template<typename T>
__global__ void array_max(T* in1, T* in2, T* out, size_t n) {
	size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n) {
		T s1 = in1[tid];
		T s2 = in2[tid];
		out[tid] = s1 > s2 ? s1 : s2;
	}
}


template<typename Lambda>
__global__ void traverse(Scaler* dst, int num, Lambda func) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num) {
		dst[tid] = func(tid);
	}
}

template<typename Lambda>
__global__ void traverse_noret(int num, Lambda func) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num) {
		func(tid);
	}
}

template<typename Lambda>
__global__ void traverse(Scaler* dst[], int num_array, int array_size, Lambda func) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int array_id = tid / array_size;
	int entry_id = tid % array_size;
	if (tid < array_size*num_array) {
		dst[array_id][entry_id] = func(array_id, entry_id);
	}
}

template<typename Lambda>
__global__ void traverse_noret(int num_array, int array_size, Lambda func) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int array_id = tid / array_size;
	int entry_id = tid % array_size;
	if (tid < array_size*num_array) {
		func(array_id, entry_id);
	}
}

template<typename T, int blockSize = 512>
__global__ void block_norm_kernel(const T* pdatax,const T* pdatay,const T* pdataz, T* odata, size_t n) {
	__shared__ T sdatax[blockSize];
	__shared__ T sdatay[blockSize];
	__shared__ T sdataz[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockDim.x*blockIdx.x;
	T sx = 0.f, sy = 0, sz = 0;
	// load data to block
	if (element_id < n) {
		sx = pdatax[element_id];
		sx = sx * sx;
		sy = pdatay[element_id];
		sy = sy * sy;
		sz = pdataz[element_id];
		sz = sz * sz;
	}
	sdatax[tid] = sx; sdatay[tid] = sy; sdataz[tid] = sz;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdatax[tid] += sdatax[tid + 256]; sdatay[tid] += sdatay[tid + 256]; sdataz[tid] += sdataz[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdatax[tid] += sdatax[tid + 128]; sdatay[tid] += sdatay[tid + 128]; sdataz[tid] += sdataz[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdatax[tid] += sdatax[tid + 64]; sdatay[tid] += sdatay[tid + 64]; sdataz[tid] += sdataz[tid + 64]; } __syncthreads(); }
	// use warpReduce to sum last 64 component 
	if (tid < 32) { warpReduce<T, blockSize>(sdatax, tid); warpReduce<T, blockSize>(sdatay, tid); warpReduce<T, blockSize>(sdataz, tid); }
	if (tid == 0) odata[blockIdx.x] = sdatax[0] + sdatay[0] + sdataz[0];
}

template<typename T, int blockSize = 512>
__global__ void block_dot_kernel(const T* pdatax,const T* pdatay,const T* pdataz,const T* qdatax,const T* qdatay,const T* qdataz, T* odata, size_t n) {
	__shared__ T sdatax[blockSize];
	__shared__ T sdatay[blockSize];
	__shared__ T sdataz[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T sx = 0, sy = 0, sz = 0.f;
	// load data to block
	if (element_id < n) {
		sx = pdatax[element_id] * qdatax[element_id];
		sy = pdatay[element_id] * qdatay[element_id];
		sz = pdataz[element_id] * qdataz[element_id];
	}
	sdatax[tid] = sx; sdatay[tid] = sy; sdataz[tid] = sz;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdatax[tid] += sdatax[tid + 256]; sdatay[tid] += sdatay[tid + 256]; sdataz[tid] += sdataz[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdatax[tid] += sdatax[tid + 128]; sdatay[tid] += sdatay[tid + 128]; sdataz[tid] += sdataz[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdatax[tid] += sdatax[tid + 64]; sdatay[tid] += sdatay[tid + 64]; sdataz[tid] += sdataz[tid + 64]; } __syncthreads(); }
	// use warpReduce to sum last 64 component 
	if (tid < 32) { warpReduce<T, blockSize>(sdatax, tid); warpReduce<T, blockSize>(sdatay, tid); warpReduce<T, blockSize>(sdataz, tid); }
	if (tid == 0) odata[blockIdx.x] = sdatax[0] + sdatay[0] + sdataz[0];
}

template<typename T, int blockSize = 512>
__global__ void block_dot_kernel_1(const T* pdatax, const T* pdatay, const T* pdataz, const T* qdatax, const T* qdatay, const T* qdataz, T* odata, size_t n) {
	__shared__ T sdata[blockSize];
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = 0;
	// load data to block
	if (element_id < n) {
		s = pdatax[element_id] * qdatax[element_id] +
			pdatay[element_id] * qdatay[element_id] +
			pdataz[element_id] * qdataz[element_id];
	}
	sdata[tid] = s;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	// use warpReduce to sum last 64 component 
	if (tid < 32) { warpReduce<T, blockSize>(sdata, tid); }
	if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<typename T, int blockSize = 512>
__global__ void block_norm_kernel_1(const T* pdatax, const T* pdatay, const T* pdataz, T* odata, size_t n) {
	__shared__ T sdata[blockSize];
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = 0;
	// load data to block
	if (element_id < n) {
		Scaler sx = pdatax[element_id];
		Scaler sy = pdatay[element_id];
		Scaler sz = pdataz[element_id];
		s = sx * sx + sy * sy + sz * sz;
	}
	sdata[tid] = s;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	// use warpReduce to sum last 64 component 
	if (tid < 32) { warpReduce<T, blockSize>(sdata, tid); }
	if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<typename T, int blockSize = 512>
__global__ void block_dot_kernel(const T* v1p, const T* v2p, T* odata, size_t n) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = 0.f;
	// load data to block
	if (element_id < n) {
		s = v1p[element_id] * v2p[element_id];
	}
	sdata[tid] = s;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	// use warpReduce to sum last 64 component 
	if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
	if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<typename T, typename func_t, int blockSize = 512>
__global__ void block_x_kernel(T* odata, size_t n, func_t func) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = 0.f;
	// load data to block
	if (element_id < n) {
		s = func(element_id);
	}
	sdata[tid] = s;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	// use warpReduce to sum last 64 component 
	if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
	if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<typename T, typename func_t, int blockSize = 512>
__global__ void block_x_kernel(T* src, T* odata, size_t n, func_t func) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = 0.f;
	// load data to block
	if (element_id < n) {
		s = func(src[element_id]);
	}
	sdata[tid] = s;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	// use warpReduce to sum last 64 component 
	if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
	if (tid == 0) odata[blockIdx.x] = sdata[0];

}

template <typename T, unsigned int blockSize>
__device__ void warpMax(volatile T *sdata, unsigned int tid) {
	if (blockSize >= 64) { T s = sdata[tid + 32]; if (sdata[tid] < s) sdata[tid] = s; };
	if (blockSize >= 32) { T s = sdata[tid + 16]; if (sdata[tid] < s) sdata[tid] = s; };
	if (blockSize >= 16) { T s = sdata[tid + 8]; if (sdata[tid] < s) sdata[tid] = s; };
	if (blockSize >= 8) { T s = sdata[tid + 4]; if (sdata[tid] < s) sdata[tid] = s; };
	if (blockSize >= 4) { T s = sdata[tid + 2]; if (sdata[tid] < s) sdata[tid] = s; };
	if (blockSize >= 2) { T s = sdata[tid + 1]; if (sdata[tid] < s) sdata[tid] = s; };
}

template <typename T, unsigned int blockSize>
__device__ void warpMin(volatile T *sdata, unsigned int tid) {
	if (blockSize >= 64) { T s = sdata[tid + 32]; if (sdata[tid] > s) sdata[tid] = s; };
	if (blockSize >= 32) { T s = sdata[tid + 16]; if (sdata[tid] > s) sdata[tid] = s; };
	if (blockSize >= 16) { T s = sdata[tid + 8]; if (sdata[tid] > s) sdata[tid] = s; };
	if (blockSize >= 8) { T s = sdata[tid + 4]; if (sdata[tid] > s) sdata[tid] = s; };
	if (blockSize >= 4) { T s = sdata[tid + 2]; if (sdata[tid] > s) sdata[tid] = s; };
	if (blockSize >= 2) { T s = sdata[tid + 1]; if (sdata[tid] > s) sdata[tid] = s; };
}

template<typename T,int blockSize=512>
__global__ void block_max_kernel(const T* indata, T* odata, size_t n) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = -1e30;
	if (element_id < n) {
		s = indata[element_id];
	}
	sdata[tid] = s;
	__syncthreads();

	// block max 
	if (blockSize >= 512) { if (tid < 256) { T v = sdata[tid + 256]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { T v = sdata[tid + 128]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { T v = sdata[tid + 64]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }

	// use warpReduce to sum last 64 component 
	if (tid < 32) warpMax<T, blockSize>(sdata, tid);
	if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<typename T, int blockSize = 512>
__global__ void block_maxabs_kernel(const T* indata, T* odata, size_t n) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = -1e30;
	if (element_id < n) {
		s = abs(indata[element_id]);
	}
	sdata[tid] = s;
	__syncthreads();

	// block max 
	if (blockSize >= 512) { if (tid < 256) { T v = sdata[tid + 256]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { T v = sdata[tid + 128]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { T v = sdata[tid + 64]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }

	// use warpReduce to sum last 64 component 
	if (tid < 32) warpMax<T, blockSize>(sdata, tid);
	if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template<typename T,int blockSize=512>
__global__ void block_min_kernel(const T* indata, T* odata, size_t n) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = 1e30;
	if (element_id < n) {
		s = indata[element_id];
	}
	sdata[tid] = s;
	__syncthreads();

	// block max 
	if (blockSize >= 512) { if (tid < 256) { T v = sdata[tid + 256]; if (sdata[tid] > v) sdata[tid] = v; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { T v = sdata[tid + 128]; if (sdata[tid] > v) sdata[tid] = v; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { T v = sdata[tid + 64]; if (sdata[tid] > v) sdata[tid] = v; } __syncthreads(); }

	// use warpReduce to sum last 64 component 
	if (tid < 32) warpMin<T, blockSize>(sdata, tid);
	if (tid == 0) odata[blockIdx.x] = sdata[0];
}

// dump_array_sum makes original array dirty, make sure dump is large enough
template<typename T, int blockSize = 512>
T dump_array_sum(T* dump, size_t n) {
#ifndef USE_CUDA_CUB
	T sum;
	if (n <= 1) {
		cudaMemcpy(&sum, dump, sizeof(T), cudaMemcpyDeviceToHost);
		return sum;
	}
	size_t grid_dim, block_dim;
	T* block_dump2 = dump;
	T* block_dump1 = dump + ((n + 63) / 2 / 32) * 32;
	do {
#if 0
		// input : dump1  output : dump2 
		std::swap(block_dump1, block_dump2);
		make_kernel_param(&grid_dim, &block_dim, n, blockSize);
		block_sum_kernel<T, blockSize> << <grid_dim, block_dim >> > (block_dump1, block_dump2, n);
		// error may occurred because of the inideal parallel block, the block result will overwrite latter data
#else
		make_kernel_param(&grid_dim, &block_dim, n, blockSize);
		block_sum_kernel<T, blockSize> << <grid_dim, block_dim >> > (block_dump2, block_dump2, n);
		// if the early block is excuted first, latter data will not be overwritten
#endif
	} while ((n = (n + blockSize - 1) / blockSize) > 1);
	cudaMemcpy(&sum, block_dump2, sizeof(T), cudaMemcpyDeviceToHost);
	return sum;
#else
	void* d_temp_buf = nullptr;
	size_t temp_size = 0;
	T* out;
	cub::DeviceReduce::Sum(d_temp_buf, temp_size, dump, out, n);
	d_temp_buf = (T*)reserve_buf(temp_size + sizeof(T) * 10);
	out = ((T*)d_temp_buf) + (temp_size + sizeof(T) - 1) / sizeof(T);
	cub::DeviceReduce::Sum(d_temp_buf, temp_size, dump, out, n);
	T sum;
	cudaMemcpy(&sum, out, sizeof(T), cudaMemcpyDeviceToHost);
	return sum;
#endif
}


extern double dump_array_sum(float* dump, size_t n);

template<typename T, typename Lambda, int blockSize = 512>
T dump_map_sum(T* dump, Lambda func, size_t n) {
	T sum;
	if (n <= 1) {
		cudaMemcpy(&sum, dump, sizeof(T), cudaMemcpyDeviceToHost);
		return sum;
	}
	size_t grid_size, block_size;
	T* block_dump = dump;
	int itn = 0;
	do {
		make_kernel_param(&grid_size, &block_size, n, blockSize);
		if (itn == 0) {
			block_x_kernel << <grid_size, block_size >> > (block_dump, block_dump, n, func);
		}
		else {
			block_sum_kernel<T, blockSize> << <grid_size, block_size >> > (block_dump, block_dump, n);
		}
		itn++;
	} while ((n = (n + blockSize - 1) / blockSize) > 1);
	cudaMemcpy(&sum, block_dump, sizeof(T), cudaMemcpyDeviceToHost);
	return sum;
}

template<typename T>
__device__ T clamp(T value, T low, T upp) {
	if (value < low) return low;
	if (value > upp) return upp;
	return value;
}

template<typename T,int blockSize=512>
T dump_max(T* dump, size_t n) {
	T max_num = -1e30;
	if (n <= 1) {
		cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
		return max_num;
	}
	size_t grid_dim, block_dim;
	do {
		make_kernel_param(&grid_dim, &block_dim, n, blockSize);
		block_max_kernel<T, blockSize> << <grid_dim, block_dim >> > (dump, dump, n);
		//cudaDeviceSynchronize();
		//cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
		//std::cout << "current max num " << max_num << std::endl;
	} while ((n = (n + blockSize - 1) / blockSize) > 1);
	cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
	return max_num;
}

template<typename T,int blockSize=512>
T dump_min(T* dump, size_t n) {
	T min_num = 1e30;
	if (n <= 1) {
		cudaMemcpy(&min_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
		return min_num;
	}
	size_t grid_dim, block_dim;
	do {
		make_kernel_param(&grid_dim, &block_dim, n, blockSize);
		block_min_kernel<T, blockSize> << <grid_dim, block_dim >> > (dump, dump, n);
		//cudaDeviceSynchronize();
		//cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
		//std::cout << "current max num " << max_num << std::endl;
	} while ((n = (n + blockSize - 1) / blockSize) > 1);
	cudaMemcpy(&min_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
	return min_num;
}

template<typename T, int blockSize = 512>
T dump_array_sum_v0(T* dump, size_t n) {
	T sum;
	if (n <= 1) {
		cudaMemcpy(&sum, dump, sizeof(T), cudaMemcpyDeviceToHost);
		return sum;
	}
	size_t grid_dim, block_dim;
	T* block_dump2 = dump;
	T* block_dump1 = dump + ((n + 63) / 2 / 32) * 32;
	do {
		// input : dump1  output : dump2 
		std::swap(block_dump1, block_dump2);
		make_kernel_param(&grid_dim, &block_dim, n, blockSize);
		block_sum_kernel<T, blockSize> << <grid_dim, block_dim >> > (block_dump1, block_dump2, n);
		// error may occurred because of the inideal parallel block, the block result will overwrite latter data
	} while ((n = (n + blockSize - 1) / blockSize) > 1);
	cudaMemcpy(&sum, block_dump2, sizeof(T), cudaMemcpyDeviceToHost);
	return sum;
}

template<typename T>
T norm(T* in_datax, T* in_datay, T* in_dataz, T* block_dump, size_t n, T* sum_dst = nullptr) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, n, blockSize);
	block_norm_kernel_1<T, blockSize> << <grid_dim, block_dim >> > (in_datax, in_datay, in_dataz, block_dump, n);
	if (n <= blockSize) {
		T sum;
		cudaMemcpy(&sum, block_dump, sizeof(T), cudaMemcpyDeviceToHost);
		sum = sqrt(sum);
		if (sum_dst != nullptr)
			cudaMemcpy(sum_dst, &sum, sizeof(T), cudaMemcpyHostToDevice);
		return sum;
	}
	else {
		T sum = dump_array_sum(block_dump, (n + blockSize - 1) / blockSize);
		sum = sqrt(sum);
		if (sum_dst != nullptr)
			cudaMemcpy(sum_dst, &sum, sizeof(T), cudaMemcpyHostToDevice);
		return sum;
	}
}

template<typename T>
T dot(T* pdatax, T* pdatay, T* pdataz, T* qdatax, T* qdatay, T* qdataz, T* odata, size_t n, T* sum_dst = nullptr) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, n, blockSize);
	block_dot_kernel_1<T, blockSize> << <grid_dim, block_dim >> > (pdatax, pdatay, pdataz, qdatax, qdatay, qdataz, odata, n);
	
	if (n <= blockSize) {
		T sum;
		cudaMemcpy(&sum, odata, sizeof(T), cudaMemcpyDeviceToHost);
		if (sum_dst != nullptr)
			cudaMemcpy(sum_dst, &sum, sizeof(T), cudaMemcpyHostToDevice);
		return sum;
	}
	else {
		T sum = dump_array_sum(odata, (n + blockSize - 1) / blockSize);
		if (sum_dst != nullptr)
			cudaMemcpy(sum_dst, &sum, sizeof(T), cudaMemcpyHostToDevice);
		return sum;
	}
}

template<typename T>
T dot(const T* indata1, const T* indata2, T* dump_buf, size_t n, T* dot_dst = nullptr) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, n, blockSize);
	block_dot_kernel << <grid_dim, block_dim >> > (indata1, indata2, dump_buf, n);

	if (n <= blockSize) {
		T sum;
		cudaMemcpy(&sum, dump_buf, sizeof(T), cudaMemcpyDeviceToHost);
		if (dot_dst != nullptr)
			cudaMemcpy(dot_dst, &sum, sizeof(T), cudaMemcpyHostToDevice);
		//cuda_error_check;
		return sum;
	}
	else {
		T sum = dump_array_sum(dump_buf, (n + blockSize - 1) / blockSize);
		if (dot_dst != nullptr)
			cudaMemcpy(dot_dst, &sum, sizeof(T), cudaMemcpyHostToDevice);
		//cuda_error_check;
		return sum;
	}
}

template<typename T>
T parallel_max(const T* indata, T* dump, size_t array_size, T* max_dst = nullptr) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
	block_max_kernel << <grid_dim, block_dim >> > (indata, dump, array_size);

	if (array_size <= blockSize) {
		T max_num;
		cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
		if (max_dst != nullptr)
			cudaMemcpy(max_dst, &max_num, sizeof(T), cudaMemcpyHostToDevice);
		return max_num;
	}
	else {
		T max_num = dump_max(dump, (array_size + blockSize - 1) / blockSize);
		if (max_dst != nullptr)
			cudaMemcpy(max_dst, &max_num, sizeof(T), cudaMemcpyHostToDevice);
		return max_num;
	}
}

template<typename T>
T parallel_maxabs(const T* indata, T* dump, size_t array_size, T* max_dst = nullptr) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
	block_maxabs_kernel << <grid_dim, block_dim >> > (indata, dump, array_size);

	if (array_size <= blockSize) {
		T max_num;
		cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
		if (max_dst != nullptr)
			cudaMemcpy(max_dst, &max_num, sizeof(T), cudaMemcpyHostToDevice);
		return max_num;
	}
	else {
		T max_num = dump_max(dump, (array_size + blockSize - 1) / blockSize);
		if (max_dst != nullptr)
			cudaMemcpy(max_dst, &max_num, sizeof(T), cudaMemcpyHostToDevice);
		return max_num;
	}
}

template<typename T>
T parallel_min(const T* indata, T* dump, size_t array_size, T* min_dst = nullptr) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
	block_min_kernel << <grid_dim, block_dim >> > (indata, dump, array_size);

	if (array_size <= blockSize) {
		T min_num;
		cudaMemcpy(&min_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
		if (min_dst != nullptr)
			cudaMemcpy(min_dst, &min_num, sizeof(T), cudaMemcpyHostToDevice);
		return min_num;
	}
	else {
		T min_num = dump_min(dump, (array_size + blockSize - 1) / blockSize);
		if (min_dst != nullptr)
			cudaMemcpy(min_dst, &min_num, sizeof(T), cudaMemcpyHostToDevice);
		return min_num;
	}
}

template<typename T>
T parallel_sum(const T* indata, T* dump, size_t array_size, T* sum_dst = nullptr) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
	block_sum_kernel << <grid_dim, block_dim >> > (indata, dump, array_size);

	if (array_size <= blockSize) {
		T array_sum;
		cudaMemcpy(&array_sum, dump, sizeof(T), cudaMemcpyDeviceToHost);
		if (sum_dst != nullptr) {
			cudaMemcpy(sum_dst, &array_sum, sizeof(T), cudaMemcpyHostToDevice);
		}
		return array_sum;
	}
	else {
		T array_sum = dump_array_sum(dump, (array_size + blockSize - 1) / blockSize);
		if (sum_dst != nullptr) {
			cudaMemcpy(sum_dst, &array_sum, sizeof(T), cudaMemcpyHostToDevice);
		}
		return array_sum;
	}
}

template<typename T>
double parallel_sum_d(const T* indata, double* dump, size_t array_size, T* sum_dst = nullptr) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
	block_sum_kernel<T, 512, double> << <grid_dim, block_dim >> > (indata, dump, array_size);

	if (array_size <= blockSize) {
		double array_sum;
		cudaMemcpy(&array_sum, dump, sizeof(double), cudaMemcpyDeviceToHost);
		if (sum_dst != nullptr) {
			cudaMemcpy(sum_dst, &array_sum, sizeof(double), cudaMemcpyHostToDevice);
		}
		return array_sum;
	}
	else {
		double array_sum = dump_array_sum(dump, (array_size + blockSize - 1) / blockSize);
		if (sum_dst != nullptr) {
			cudaMemcpy(sum_dst, &array_sum, sizeof(double), cudaMemcpyHostToDevice);
		}
		return array_sum;
	}
}

template<typename T, typename Lambda>
T parallel_map_sum(const T* indata, T* dump, size_t array_size, Lambda func, T* sum_dst = nullptr) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
	block_x_kernel << <grid_dim, block_dim >> > (indata, dump, array_size, func);

	if (array_size <= blockSize) {
		T array_sum;
		cudaMemcpy(&array_sum, dump, sizeof(T), cudaMemcpyDeviceToHost);
		if (sum_dst != nullptr) {
			cudaMemcpy(sum_dst, &array_sum, sizeof(T), cudaMemcpyHostToDevice);
		}
		return array_sum;
	}
	else {
		T array_sum = dump_array_sum(dump, (array_size + blockSize - 1) / blockSize);
		if (sum_dst != nullptr) {
			cudaMemcpy(sum_dst, &array_sum, sizeof(T), cudaMemcpyHostToDevice);
		}
		return array_sum;
	}
}

template<typename T, typename Lambda>
T parallel_map_sum(T* dump, size_t array_size, Lambda func, T* sum_dst = nullptr) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
	block_x_kernel << <grid_dim, block_dim >> > (dump, array_size, func);

	if (array_size <= blockSize) {
		T array_sum;
		cudaMemcpy(&array_sum, dump, sizeof(T), cudaMemcpyDeviceToHost);
		if (sum_dst != nullptr) {
			cudaMemcpy(sum_dst, &array_sum, sizeof(T), cudaMemcpyHostToDevice);
		}
		return array_sum;
	}
	else {
		T array_sum = dump_array_sum(dump, (array_size + blockSize - 1) / blockSize);
		if (sum_dst != nullptr) {
			cudaMemcpy(sum_dst, &array_sum, sizeof(T), cudaMemcpyHostToDevice);
		}
		return array_sum;
	}
}

template<typename T>
T parallel_diffdot(int v3size, T* v1[3], T* v2[3], T* v3[3], T* v4[3], T* dump) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, v3size, blockSize);
	devArray_t<const Scaler*, 3> _v1, _v2, _v3, _v4;
	for (int i = 0; i < 3; i++) {
		_v1[i] = v1[i];
		_v2[i] = v2[i];
		_v3[i] = v3[i];
		_v4[i] = v4[i];
	}
	auto v3diffdot = [=] __device__(int tid) {
		return (_v2[0][tid] - _v1[0][tid])*(_v4[0][tid] - _v3[0][tid]) 
			+ (_v2[1][tid] - _v1[1][tid])*(_v4[1][tid] - _v3[1][tid])
			+ (_v2[2][tid] - _v1[2][tid])*(_v4[2][tid] - _v3[2][tid]);
	};
	block_x_kernel << <grid_dim, block_dim >> > (dump, v3size, v3diffdot);

	if (v3size <= blockSize) {
		T array_sum;
		cudaMemcpy(&array_sum, dump, sizeof(T), cudaMemcpyDeviceToHost);
		return array_sum;
	}
	else {
		cudaDeviceSynchronize();
		T array_sum = dump_array_sum(dump, (v3size + blockSize - 1) / blockSize);
		return array_sum;
	}
}

template<typename T>
T parallel_diffdiffdot(int v3size, T* v1[3], T* v2[3], T* v3[3], T* v4[3],
	T* u1[3], T* u2[3], T* u3[3], T* u4[3],
	T* dump) {
	constexpr int blockSize = 512;
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, v3size, blockSize);
	cuda_error_check;
	devArray_t<const Scaler*, 3> _v1, _v2, _v3, _v4, _u1, _u2, _u3, _u4;
	for (int i = 0; i < 3; i++) {
		_v1[i] = v1[i];
		_v2[i] = v2[i];
		_v3[i] = v3[i];
		_v4[i] = v4[i];
		_u1[i] = u1[i];
		_u2[i] = u2[i];
		_u3[i] = u3[i];
		_u4[i] = u4[i];
	}
	auto v3diffdot = [=] __device__(int tid) {
		//if (tid == 0) {
		//	for (int i = 0; i < 3; i++) {
		//		printf("_v1[%d] = %p\n", i, _v1[i]);
		//		printf("_v2[%d] = %p\n", i, _v2[i]);
		//		printf("_v3[%d] = %p\n", i, _v3[i]);
		//		printf("_v4[%d] = %p\n", i, _v4[i]);
		//		printf("_u1[%d] = %p\n", i, _u1[i]);
		//		printf("_u2[%d] = %p\n", i, _u2[i]);
		//		printf("_u3[%d] = %p\n", i, _u3[i]);
		//		printf("_u4[%d] = %p\n", i, _u4[i]);
		//	}
		//}
		Scaler s = (_v2[0][tid] - _v1[0][tid] - _v4[0][tid] + _v3[0][tid]) *(_u2[0][tid] - _u1[0][tid] - _u4[0][tid] + _u3[0][tid]) +
			(_v2[1][tid] - _v1[1][tid] - _v4[1][tid] + _v3[1][tid]) *(_u2[1][tid] - _u1[1][tid] - _u4[1][tid] + _u3[1][tid]) +
			(_v2[2][tid] - _v1[2][tid] - _v4[2][tid] + _v3[2][tid]) *(_u2[2][tid] - _u1[2][tid] - _u4[2][tid] + _u3[2][tid]);
		return s;
	};
	block_x_kernel << <grid_dim, block_dim >> > (dump, v3size, v3diffdot);

	if (v3size <= blockSize) {
		T array_sum;
		cudaMemcpy(&array_sum, dump, sizeof(T), cudaMemcpyDeviceToHost);
		return array_sum;
	}
	else {
		cudaDeviceSynchronize();
		cuda_error_check;
		T array_sum = dump_array_sum(dump, (v3size + blockSize - 1) / blockSize);
		return array_sum;
	}
}

template<typename T>
__device__ T vsum_wise(int k, T* p) {
	return p[k];
}

template<typename T, typename... Args>
__device__ T vsum_wise(int k, T* v1, Args*... v) {
	return v1[k] + vsum_wise(k, v...);
}

template<typename _TOut,typename T, typename... Args>
__global__ void vsum_kernel(int n, _TOut* out, T* v1, Args*... args) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= n) return;
	out[tid] = vsum_wise(tid, v1, args...);
}


template<typename _Tout, typename T, typename... Args>
void vsum(int n, _Tout* out, T* v1, Args*... args) {
	size_t grid_dim, block_dim;
	make_kernel_param(&grid_dim, &block_dim, n, 512);
	vsum_kernel << <grid_dim, block_dim >> > (n, out, v1, args...);
	cudaDeviceSynchronize();
	cuda_error_check;
}


template<typename T>
struct array_t {
	T* _ptr;
	size_t _len;
	__host__ __device__ array_t(T* ptr, size_t len) : _ptr(ptr), _len(len) { }
	__host__  array_t(size_t len, T val) {
		cudaMalloc(&_ptr, sizeof(T)*len);
		init_array(_ptr, val, len);
	}
	__host__ void set(size_t offs, T val) {
		cudaMemcpy(_ptr + offs, &val, sizeof(T), cudaMemcpyHostToDevice);
	}

	__host__ void free(void) {
		cudaFree(_ptr);
	}

	__host__ T* data(void) {
		return _ptr;
	}

	template<typename T1>
	__host__ void operator=(const array_t<T1>& ar1) const {
		T* dst = _ptr;
		T1* src = ar1._ptr;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _len, 512);
		map<<<grid_size,block_size>>>(_len, [=] __device__(int eid) {
			dst[eid] = src[eid];
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	template<typename F>
	__host__ array_t& operator/=(F f) {
		T* src = _ptr;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _len, 512);
		map<<<grid_size,block_size>>>(_len, [=] __device__(int eid) {
			src[eid] /= f;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return (*this);
	}

	template<typename F>
	__host__ array_t& operator*=(F f) {
		T* src = _ptr;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _len, 512);
		map<<<grid_size,block_size>>>(_len, [=] __device__(int eid) {
			src[eid] *= f;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return (*this);
	}

	template<typename F>
	__host__ array_t& operator+=(F f) {
		T* src = _ptr;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _len, 512);
		map<<<grid_size,block_size>>>(_len, [=] __device__(int eid) {
			src[eid] += f;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return (*this);
	}

	template<typename F>
	__host__ array_t& operator-=(F f) {
		T* src = _ptr;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _len, 512);
		map<<<grid_size,block_size>>>(_len, [=] __device__(int eid) {
			src[eid] -= f;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return (*this);
	}
};

template<typename T>
struct _randArrayGen {
	static void gen(curandGenerator_t& gen, double** dst, int nArray, size_t len) {
		printf("\033[31mUnsupported scalar type!\033[0m\n");
	}
};

template<>
struct _randArrayGen<double> {
	static void gen(curandGenerator_t& gen, double** dst, int nArray, size_t len) {
		for (int i = 0; i < nArray; i++) {
			curandGenerateUniformDouble(gen, dst[i], len);
		}
	}
};

template<>
struct _randArrayGen<float> {
	static void gen(curandGenerator_t& gen, float** dst, int nArray, size_t len) {
		for (int i = 0; i < nArray; i++) {
			curandGenerateUniform(gen, dst[i], len);
		}
	}
};

template<typename T>
void randArray(T** dst, int nArray, size_t len, T low = T{ 0 }, T upp = T{ 0 }) {
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, (int)time(nullptr));
	_randArrayGen<T>::gen(generator, dst, nArray, len);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, len, 512);
	for (int i = 0; i < nArray; i++) {
		T* pdata = dst[i];
		traverse << <grid_size, block_size >> > (pdata, len, [=] __device__(int tid) {
			T value = pdata[tid];
			value = low + value * (upp - low);
			return  value;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}

template<typename T>
void check_array_len(const T* _pdata, size_t len) {
	//auto kernel = [=] __device__(int tid) {
	//	int a = _pdata[tid];
	//};
	//size_t grid_size, block_size;
	//make_kernel_param(&grid_size, &block_size, len, 512);
	//traverse_noret << <grid_size, block_size >> > (len, kernel);
	//cudaDeviceSynchronize();
	//auto err = cudaGetLastError();
	//if (err != CUDA_SUCCESS) {
	//	printf("\033[31m-- Check length Failed! \033[0m\n");
	//}

	CUdeviceptr pbase, pdata;
	pdata = reinterpret_cast<CUdeviceptr>(_pdata);
	size_t arrlen;
	cuMemGetAddressRange(&pbase, &arrlen, pdata);
	if (len * sizeof(T) > arrlen) {
		printf("\033[31m-- Check length Failed! \033[0m\n");
	}
	cuda_error_check;
}

template<typename T>
__device__ bool read_gbit(const T* bits, size_t id) {
	return bits[id / (sizeof(T) * 8)] & (T{ 1 } << (id % (sizeof(T) * 8)));
}

template<typename T, typename std::enable_if<!std::is_pointer<T>::value, int>::type = 0>
__device__ bool read_gbit(T word, int id) {
	return word & (T{ 1 } << id);
}

template<typename T>
__device__ void set_gbit(T* bits, size_t id) {
	bits[id / (sizeof(T) * 8)] |= (T{ 1 } << (id % (sizeof(T) * 8)));
}

template<typename T, typename std::enable_if<!std::is_pointer<T>::value, int>::type = 0>
__device__ void set_gbit(T word, int id) {
	word |= (T{ 1 } << id);
}

template<typename T>
__device__ void reset_gbit(T* bits, size_t id) {
	bits[id / (sizeof(T) * 8)] &= ~(T{ 1 } << (id % (sizeof(T) * 8)));
}

template<typename T, typename std::enable_if<!std::is_pointer<T>::value, int>::type = 0>
__device__ void reset_gbit(T word, int id) {
	word &= ~(T{ 1 } << id);
}

template<typename T>
__device__ void atomic_set_gbit(T* bits, size_t id) {
	atomicOr(
		((int*)(void*)bits) + id / (sizeof(int) * 8),
		int{ 1 } << (id % (sizeof(int) * 8))
	);
}

template<typename T>
__device__ void atomic_reset_gbit(T* bits, size_t id) {
	atomicAnd(
		((int*)(void*)bits) + id / (sizeof(int) * 8),
		~(int{ 1 } << (id % (sizeof(int) * 8)))
	);
}

template<typename T>
__device__ void initSharedMem(volatile T* pshared, int len, T value = T{ 0 }) {
	int nfil = 0;
	int tid = threadIdx.x;
	if (tid < len) {
		pshared[tid] = value;
	}
	nfil = blockDim.x;
	while (nfil < len) {
		if (tid + nfil < len) {
			pshared[tid + nfil] = value;
		}
		nfil += blockDim.x;
	}
	__syncthreads();
}

template<typename T/*, bool SelfAllocate = false */>
struct gBitSAT {
	static constexpr size_t size_mask = sizeof(T) * 8 - 1;
	const T* _bitarray;
	const int* _chunksat;
	template<int N, bool stop = (N == 0)>
	struct firstOne {
		static constexpr int value = 1 + firstOne< (N >> 1), ((N >> 1) == 0)>::value;
	};

	template<int N>
	struct firstOne<N, true> {
		static constexpr int value = -1;
	};

	template<typename Dt>
	__host__ __device__ inline int countOne(Dt num) const {
#if 0
		int n = 0;
		while (num) {
			num &= (num - 1);
			n++;
		}
		return n;
#else
		return __popc(num);
#endif
	}

	//__host__ ~gBitSAT() {
	//	if (SelfAllocate) {
	//		cudaFree(_bitarray);
	//		cudaFree(_chunksat);
	//	}
	//}

	__host__ void destroy(void) {
		cudaFree(const_cast<T*>(_bitarray));
		cudaFree(const_cast<int*>(_chunksat));
	}

	//template<bool Allocate = SelfAllocate, std::enable_if<!Allocate, void>::type *= nullptr>
	__host__ __device__  gBitSAT(const T* bitarray, const int* chunksat)
		:_bitarray(bitarray), _chunksat(chunksat)
	{ }

	//template<bool Allocate = SelfAllocate, std::enable_if<Allocate, void>::type *= nullptr>
	__host__  gBitSAT(const std::vector<T>& hostbits, const std::vector<int>& hostsat) {
		cudaMalloc(const_cast<T**>(&_bitarray), hostbits.size() * sizeof(T));
		cudaMemcpy(const_cast<T*>(_bitarray), hostbits.data(), sizeof(T) * hostbits.size(), cudaMemcpyHostToDevice);

		cudaMalloc(const_cast<int**>(&_chunksat), hostsat.size() * sizeof(int));
		cudaMemcpy(const_cast<int*>(_chunksat), hostsat.data(), sizeof(int)* hostsat.size(), cudaMemcpyHostToDevice);
	}

	__host__ __device__ gBitSAT(void) = default;

	__host__ __device__ int operator[](size_t id) const {
		int ent = id >> firstOne<sizeof(T) * 8>::value;
		int mod = id & size_mask;
		return _chunksat[ent] + countOne(_bitarray[ent] & ((T{ 1 } << mod) - 1));
	}

	__host__ __device__ int operator()(size_t id) const {
		int ent = id >> firstOne<sizeof(T) * 8>::value;
		int mod = id & size_mask;
		T resword = _bitarray[ent];
		if ((resword & (T{ 1 } << mod)) == 0) {
			return -1;
		}
		else {
			return _chunksat[ent] + countOne(resword & ((T{ 1 } << mod) - 1));
		}
	}
};

__global__ void remap(int n, Scaler* p, Scaler l_, Scaler h_, Scaler L_, Scaler H_);



#endif

