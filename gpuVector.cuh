#ifndef __GPU_VECTOR_CUH
#define __GPU_VECTOR_CUH

#include "gpuVector.h"

#ifdef __USE_GVECTOR_LAZY_EVALUATION

#include"cuda_runtime.h"
#include"iostream"
#include"lib.cuh"
#include"type_traits"

namespace gv {


template<typename graph_t>
__global__ void compute_graph_kernel(Scalar* dst, int array_size, graph_t graph) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= array_size) return;
	//if (tid == 0) {
	//	printf("graph exp ptr1 =  %p, dim1 = %d\n", graph.exp1.ptr, graph.exp1.vec_dim);
	//	printf("                    size() = %d\n", graph.size());
	//}
	dst[tid] = graph.eval(tid);
}

template<typename T, typename graph_t1, typename graph_t2, int blockSize = 512>
__global__ void dot_graph_kernel(T* dump, int array_size, graph_t1 g1, graph_t2 g2) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = 0.f;
	// load data to block
	if (element_id < array_size) {
		s = g1.eval(element_id) * g2.eval(element_id);
	}
	sdata[tid] = s;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	// use warpReduce to sum last 64 component 
	if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
	if (tid == 0) dump[blockIdx.x] = sdata[0];
}

template<typename T, typename graph_t, int blockSize = 512>
__global__ void sum_graph_kernel(T* dump, int array_size, graph_t graph) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = 0.f;
	// load data to block
	if (element_id < array_size) {
		s = graph.eval(element_id);
	}
	sdata[tid] = s;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	// use warpReduce to sum last 64 component 
	if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
	if (tid == 0) dump[blockIdx.x] = sdata[0];
}

template<typename T, typename graph_t, int blockSize = 512>
__global__ void sqrnorm_graph_kernel(T* dump, int array_size, graph_t graph) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = 0.f;
	// load data to block
	if (element_id < array_size) {
		T val = graph.eval(element_id);
		s = val * val;
	}
	sdata[tid] = s;
	__syncthreads();

	// block reduce sum
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	// use warpReduce to sum last 64 component 
	if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
	if (tid == 0) dump[blockIdx.x] = sdata[0];
}

template<typename T, typename graph_t, int blockSize = 512>
__global__ void max_graph_kernel(T* odata, size_t n, graph_t graph) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = -1e30;
	if (element_id < n) {
		s = graph.eval(element_id);
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

template<typename T, typename graph_t, int blockSize = 512>
__global__ void min_graph_kernel(T* odata, size_t n, graph_t graph) {
	__shared__ T sdata[blockSize];
	if (blockDim.x != blockSize) {
		printf("error block size does not match at line %d ! \n", __LINE__);
	}
	int tid = threadIdx.x;
	size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
	T s = 1e30;
	if (element_id < n) {
		s = graph.eval(element_id);
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

//struct exp_base_t {
//	//int a = 1;
//	//__host__ __device__ exp_base_t(void) {}
//};

//template<typename...> struct is_expression;
//template<typename, typename> struct min_exp_t;
//template<typename, typename> struct max_exp_t;
//template<typename T = Scalar, typename std::enable_if<std::is_scalar<T>::value, int >::type = 0> struct scalar_t;
//template<typename T = gVector, typename std::enable_if<std::is_same<T, gVector>::value, int>::type = 0> struct var_t;

template<typename T>
struct scalar_type {
	typedef typename T::value_type type;
};

template<typename subExp_t>
struct exp_t 
	//:public exp_base_t
{
	static constexpr bool is_exp = true;
	typedef Scalar value_type;
	void launch(Scalar* dst, int n) const {
		const subExp_t* p_graph = static_cast<const subExp_t*>(this);
		subExp_t graph = *p_graph;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n, 512);
		//std::cout << "launcing with size " << n << std::endl;
		//std::cout << "result at " << dst << std::endl;
		cuda_error_check;
		//std::cout << typeid(graph).name() << std::endl;
		compute_graph_kernel<<<grid_size,block_size>>>(dst, n, graph);
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0>
	min_exp_t<subExp_t, opExp_t> min(const opExp_t& op2) const{
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		return min_exp_t<subExp_t, opExp_t>(*p_ex, op2);
	}

	template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	min_exp_t<subExp_t, scalar_t<T>> min(T s) const {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		return min_exp_t<subExp_t, scalar_t<T>>(*p_ex, scalar_t<T>(s));
	}

#if 1
	template<typename T, typename std::enable_if<std::is_same<T, gVector>::value, int>::type = 0>
	min_exp_t<subExp_t, var_t<T>> min(const T& s) const {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		return min_exp_t<subExp_t, var_t<T>>(*p_ex, var_t<T>(s));
	}
#else
	template<typename T, typename std::enable_if<std::is_same<T, gVector>::value, int>::type = 0>
	min_exp_t<var_t,var_t> min(const T& s) const {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		return min_exp_t<var_t, var_t>(*p_ex, var_t(s));
	}
#endif

	template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0>
	max_exp_t<subExp_t, opExp_t> max(const opExp_t& op2) const{
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		return max_exp_t<subExp_t, opExp_t>(*p_ex, op2);
	}

	template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	max_exp_t<subExp_t, scalar_t<T>> max(T s) const {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		return max_exp_t<subExp_t, scalar_t<T>>(*p_ex, scalar_t<T>(s));
	}

	template<typename T, typename std::enable_if<std::is_same<T, gVector>::value, int>::type = 0>
	max_exp_t<subExp_t, var_t<T>> max(const T& s) const {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		return max_exp_t<subExp_t, var_t<T>>(*p_ex, var_t<T>(s));
	}

	template<typename Lambda>
	map_exp_t<subExp_t, Lambda> map(Lambda func) const {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		return map_exp_t<subExp_t, Lambda>(*p_ex, func);
	}

	template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0>
	Scalar dot(const opExp_t& op2) const {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		subExp_t graph1 = *p_ex;
		opExp_t graph2 = op2;
		Scalar* pbuf = gVector::get_dump_buf();
		//printf("pbuf = %p\n", pbuf);
		int n = op2.size();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n, 512);
		cuda_error_check;
		dot_graph_kernel << <grid_size, block_size >> > (pbuf, n, graph1, graph2);
		cudaDeviceSynchronize();
		cuda_error_check;
		n = (n + 511) / 512;
		return dump_array_sum(pbuf, n);
	}

	Scalar sum(void)const {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		subExp_t graph = *p_ex;
		Scalar* pbuf = gVector::get_dump_buf();
		int n = graph.size();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n, 512);
		cuda_error_check;
		sum_graph_kernel << <grid_size, block_size >> > (pbuf, n, graph);
		cudaDeviceSynchronize();
		cuda_error_check;
		n = (n + 511) / 512;
		return dump_array_sum(pbuf, n);
	}

	Scalar sqrnorm(void) {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		subExp_t graph = *p_ex;
		Scalar* pbuf = gVector::get_dump_buf();
		int n = graph.size();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n, 512);
		cuda_error_check;
		sqrnorm_graph_kernel << <grid_size, block_size >> > (pbuf, n, graph);
		cudaDeviceSynchronize();
		cuda_error_check;
		n = (n + 511) / 512;
		return dump_array_sum(pbuf, n);
	}

	Scalar max(void) {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		subExp_t graph = *p_ex;
		Scalar* pbuf = gVector::get_dump_buf();
		int n = graph.size();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n, 512);
		cuda_error_check;
		max_graph_kernel << <grid_size, block_size >> > (pbuf, n, graph);
		cudaDeviceSynchronize();
		cuda_error_check;
		n = (n + 511) / 512;
		return dump_max(pbuf, n);
	}

	Scalar min(void) {
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		subExp_t graph = *p_ex;
		Scalar* pbuf = gVector::get_dump_buf();
		int n = graph.size();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n, 512);
		cuda_error_check;
		min_graph_kernel << <grid_size, block_size >> > (pbuf, n, graph);
		cudaDeviceSynchronize();
		cuda_error_check;
		n = (n + 511) / 512;
		return dump_min(pbuf, n);
	}

	Scalar norm(void) {
		return sqrt(sqrnorm());
	}

	void toMatlab(const char* name) {
#if defined(__GVECTOR_WITH_MATLAB)  
		const subExp_t* p_ex = static_cast<const subExp_t*>(this);
		subExp_t graph = *p_ex;
		gVector vec = graph;
		vec.toMatlab(name);
#endif
	}
};

template<typename T /*= gVector*/, typename std::enable_if<std::is_same<T, gVector>::value, int>::type /*= 0*/>
struct var_t 
	:public exp_t<var_t<T>>
{
	const Scalar* ptr;
	int vec_dim;
	__host__ __device__ var_t(const Scalar* ptr_) :ptr(ptr_) {}
	__host__ __device__ var_t(const gVector& var) : ptr(var.data()), vec_dim(var.size()) {}
	__device__ Scalar eval(int eid)const {
		return ptr[eid];
	}
	__host__ __device__ int size(void)const {
		return vec_dim;
	}
};

template<typename T /*= Scalar*/, typename std::enable_if<std::is_scalar<T>::value, int >::type /*= 0*/> 
struct scalar_t
	:public exp_t<scalar_t<T>>
{
	T scalar;
	__host__ __device__ scalar_t(T s) :scalar(s) {}
	__device__ T eval(int eid) const {
		return scalar;
	}
	__host__ __device__ int size(void) const {
		return 0;
	}
};

template<typename subExp_t, typename opExp_t>
struct unary_exp_t 
	:public exp_t<subExp_t>
{
	opExp_t exp;
	__host__ __device__ unary_exp_t(const opExp_t& opexp) :exp(opexp) {}
	__host__ __device__ int size(void) const {
		return exp.size();
	}
};

template<typename opExp_t>
struct negat_exp_t
	:public unary_exp_t<negat_exp_t<opExp_t>, opExp_t>
{
	__host__ __device__ negat_exp_t(const opExp_t& ex) : unary_exp_t<negat_exp_t<opExp_t>, opExp_t>(ex) {}
	__device__ Scalar eval(int eid) const{
		return -unary_exp_t<negat_exp_t<opExp_t>, opExp_t>::exp.eval(eid);
	}
};

template<typename opExp_t>
struct sqrt_exp_t
	:public unary_exp_t<sqrt_exp_t<opExp_t>, opExp_t> 
{
	__host__ __device__ sqrt_exp_t(const opExp_t& ex) :unary_exp_t<sqrt_exp_t<opExp_t>, opExp_t>(ex) {}
	__device__ Scalar eval(int eid) const {
		return sqrt(unary_exp_t<sqrt_exp_t<opExp_t>, opExp_t>::exp.eval(eid));
	}
};

template<typename opExp_t, typename Lambda>
struct map_exp_t 
	:public unary_exp_t<map_exp_t<opExp_t, Lambda>, opExp_t>
{
	Lambda _map;
	__host__ __device__ map_exp_t(const opExp_t& ex, Lambda map) 
		: unary_exp_t<map_exp_t<opExp_t, Lambda>, opExp_t>(ex), _map(map)
	{ }
	__device__ Scalar eval(int eid) const {
		return _map(unary_exp_t<map_exp_t<opExp_t, Lambda>, opExp_t>::exp.eval(eid));
	}
};

template<typename subExp_t, typename opExp1_t, typename opExp2_t>
struct binary_exp_t 
	:public exp_t<subExp_t>
{
	opExp1_t exp1;
	opExp2_t exp2;
	__host__ __device__ binary_exp_t(const opExp1_t& op1, const opExp2_t& op2) : exp1(op1), exp2(op2) {}
	__host__ __device__ int size(void) const {
		return (std::max)(exp1.size(), exp2.size());
	}
};

template<typename opExp1_t, typename opExp2_t>
struct add_exp_t
	:public binary_exp_t<add_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	typedef binary_exp_t<add_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ add_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<add_exp_t, opExp1_t, opExp2_t>(op1, op2) {}
	// add_exp_t(const add_exp_t<opExp1_t,opExp2_t>& ex): binary_exp_t<add_exp_t,opExp1_t,opExp2_t>(ex.exp1,ex.exp2){}
	__device__ Scalar eval(int eid) const {
		return baseType::exp1.eval(eid) + baseType::exp2.eval(eid);
	}
};

template<typename opExp1_t,typename opExp2_t>
struct minus_exp_t 
	:public binary_exp_t<minus_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	typedef binary_exp_t<minus_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ minus_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<minus_exp_t, opExp1_t, opExp2_t >(op1, op2) {}

	__device__ Scalar eval(int eid) const {
		return baseType::exp1.eval(eid) - baseType::exp2.eval(eid);
	}
};

template<typename opExp1_t,typename opExp2_t>
struct div_exp_t 
	:public binary_exp_t<div_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	typedef binary_exp_t<div_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ div_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<div_exp_t, opExp1_t, opExp2_t >(op1, op2) {}

	__device__ Scalar eval(int eid)const {
		return baseType::exp1.eval(eid) / baseType::exp2.eval(eid);
	}
};

template<typename opExp1_t, typename opExp2_t>
struct multiply_exp_t
	:public binary_exp_t<multiply_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	typedef  binary_exp_t<multiply_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ multiply_exp_t(const opExp1_t& op1, const opExp2_t& op2) : binary_exp_t<multiply_exp_t/*<opExp1_t, opExp2_t>*/, opExp1_t, opExp2_t>(op1, op2) {}
	// multiply_exp_t(const multiply_exp_t& ex) :baseType(ex.exp1, ex.exp2) {}
	__device__ Scalar eval(int eid) const {
		return baseType::exp1.eval(eid)*baseType::exp2.eval(eid);
	}
};

template<typename opExp1_t,typename opExp2_t>
struct pow_exp_t 
	:public binary_exp_t<pow_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	typedef binary_exp_t<pow_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ pow_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<pow_exp_t, opExp1_t, opExp2_t >(op1, op2) {}

	__device__ Scalar eval(int eid) const {
		return std::is_same<Scalar, float>::value ? powf(baseType::exp1.eval(eid), baseType::exp2.eval(eid)) : pow(baseType::exp1.eval(eid), baseType::exp2.eval(eid));
	}
};

template<typename opExp1_t,typename opExp2_t>
struct min_exp_t
	:public binary_exp_t<min_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	typedef binary_exp_t<min_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ min_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<min_exp_t, opExp1_t, opExp2_t >(op1, op2) {}
	__device__ Scalar eval(int eid) const {
		Scalar val1 = baseType::exp1.eval(eid);
		Scalar val2 = baseType::exp2.eval(eid);
		return val1 < val2 ? val1 : val2;
	}
};

//template<typename opExp1_t>
//struct min_exp_t<>
//	:public binary_exp_t<min_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
//{
//	typedef binary_exp_t<min_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
//	__host__ __device__ min_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<min_exp_t, opExp1_t, opExp2_t >(op1, op2) {}
//	__device__ Scalar eval(int eid) {
//		Scalar val1 = baseType::exp1.eval(eid);
//		Scalar val2 = baseType::exp2.eval(eid);
//		return val1 < val2 ? val1 : val2;
//	}
//};

template<typename opExp1_t,typename opExp2_t>
struct max_exp_t
	:public binary_exp_t<max_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	typedef binary_exp_t<max_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ max_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<max_exp_t, opExp1_t, opExp2_t >(op1, op2) {}
	__device__ Scalar eval(int eid) const {
		Scalar val1 = baseType::exp1.eval(eid);
		Scalar val2 = baseType::exp2.eval(eid);
		return val1 > val2 ? val1 : val2;
	}
};

//=============================================================================================================================

template<typename _T,typename _T2>
struct is_expression_impl {
	static constexpr bool value = false;
};

template<typename... _T>
struct is_expression<add_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<minus_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<multiply_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<div_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<pow_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<negat_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<sqrt_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<map_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<max_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<min_exp_t<_T...>> {
	static constexpr bool value = true;
};
//=============================================================================================================================


template<typename opExp_t>
negat_exp_t<opExp_t> operator-(
	const typename std::enable_if<opExp_t::is_exp, opExp_t>::type&
	exp) {
	return negat_exp_t<opExp_t>(exp);
}

extern negat_exp_t<var_t<>> operator-(const gVector& gvec);

/****************************************************************************
		minus expression
****************************************************************************/

template<typename opExp1_t, typename opExp2_t, 
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
minus_exp_t<opExp1_t, opExp2_t> operator-(
	const opExp1_t& op1,
	const opExp2_t& op2) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return minus_exp_t<opExp1_t, opExp2_t>(op1, op2);
}

template<typename opExp1_t,
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0 >
minus_exp_t<opExp1_t, var_t<>> operator-(
	const opExp1_t& op1, const gVector& v2) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	return minus_exp_t<opExp1_t, var_t<>>(op1, var_t<>(v2));
}

template<typename opExp1_t,
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0 >
minus_exp_t<opExp1_t, scalar_t<>> operator-(
	const opExp1_t& op1, 
	Scalar s2
	) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	return minus_exp_t<opExp1_t, scalar_t<>>(op1, scalar_t<>(s2));
}

template<typename opExp2_t,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
minus_exp_t<var_t<>, opExp2_t> operator-(
	const gVector& v1,
	const opExp2_t& op2) {
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return minus_exp_t<var_t<>, opExp2_t>(var_t<>(v1), op2);
}

template<typename opExp2_t,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
minus_exp_t<scalar_t<>, opExp2_t > operator-(
	Scalar s1,
	const opExp2_t& op2
	) {
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return minus_exp_t<scalar_t<>, opExp2_t>(scalar_t<>(s1), op2);
}

extern minus_exp_t<var_t<>, var_t<>> operator-(
	const gVector& v1,
	const gVector& v2
	);

extern minus_exp_t<scalar_t<>, var_t<>> operator-(
	Scalar s,
	const gVector& v2
	);

extern minus_exp_t<var_t<>, scalar_t<>> operator-(
	const gVector& v1,
	Scalar s
	);

/*****************************************************************************
	add expression
****************************************************************************/

template<typename opExp1_t, typename opExp2_t, 
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
add_exp_t<opExp1_t, opExp2_t> operator+(
	const opExp1_t& op1,
	const opExp2_t& op2) {
	return add_exp_t<opExp1_t, opExp2_t>(op1, op2);
}

template<typename opExp1_t,
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0>
add_exp_t<opExp1_t, var_t<>> operator+(
	const opExp1_t& op1, const gVector& v2) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	return add_exp_t<opExp1_t, var_t<>>(op1, var_t<>(v2));
}

template<typename opExp1_t,
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0>
add_exp_t<opExp1_t, scalar_t<>> operator+(
	const opExp1_t& op1, 
	Scalar s2
	) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	return add_exp_t<opExp1_t, scalar_t<>>(op1, scalar_t<>(s2));
}

template<typename opExp2_t,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0>
add_exp_t<var_t<>, opExp2_t> operator+(
	const gVector& v1,
	const opExp2_t& op2) {
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return add_exp_t<var_t<>, opExp2_t>(var_t<>(v1), op2);
}

template<typename opExp2_t,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
add_exp_t<scalar_t<>, opExp2_t > operator+(
	Scalar s1,
	const opExp2_t& op2
	) {
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return add_exp_t<scalar_t<>, opExp2_t>(scalar_t<>(s1), op2);
}

extern add_exp_t<var_t<>, var_t<>> operator+(
	const gVector& v1,
	const gVector& v2
	);

extern add_exp_t<scalar_t<>, var_t<>> operator+(
	Scalar s,
	const gVector& v2
	);

extern add_exp_t<var_t<>, scalar_t<>> operator+(
	const gVector& v1,
	Scalar s
	);

/*****************************************************************************
	multiply expression
****************************************************************************/

template<typename opExp1_t, typename opExp2_t, 
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
multiply_exp_t<opExp1_t, opExp2_t> operator*(
	const opExp1_t& op1,
	const opExp2_t& op2) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return multiply_exp_t<opExp1_t, opExp2_t>(op1, op2);
}

template<typename opExp1_t,
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0 >
multiply_exp_t<opExp1_t, var_t<>> operator*(
	const opExp1_t& op1, const gVector& v2) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	return multiply_exp_t<opExp1_t, var_t<>>(op1, var_t<>(v2));
}

template<typename opExp1_t,
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0 >
multiply_exp_t<opExp1_t, scalar_t<>> operator*(
	const opExp1_t& op1, 
	Scalar s2
	) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	return multiply_exp_t<opExp1_t, scalar_t<>>(op1, scalar_t<>(s2));
}

template<typename opExp2_t,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
multiply_exp_t<var_t<>, opExp2_t> operator*(
	const gVector& v1,
	const opExp2_t& op2) {
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return multiply_exp_t<var_t<>, opExp2_t >(var_t<>(v1), op2);
}

template<typename opExp2_t,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
multiply_exp_t<scalar_t<>, opExp2_t > operator*(
	Scalar s1,
	const opExp2_t& op2
	) {
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return multiply_exp_t<scalar_t<>, opExp2_t>(scalar_t<>(s1), op2);
}

extern multiply_exp_t<var_t<>, var_t<>> operator*(
	const gVector& v1,
	const gVector& v2
	);

extern multiply_exp_t<scalar_t<>, var_t<>> operator*(
	Scalar s,
	const gVector& v2
	);

extern multiply_exp_t<var_t<>, scalar_t<>> operator*(
	const gVector& v1,
	Scalar s
	);
/*****************************************************************************
	div expression
****************************************************************************/

template<typename opExp1_t, typename opExp2_t, 
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
div_exp_t<opExp1_t, opExp2_t> operator/(
	const opExp1_t& op1,
	const opExp2_t& op2) {
	static_assert(opExp1_t::is_exp, "Not a vector expression !");
	static_assert(opExp2_t::is_exp, "Not a vector expression !");
	return div_exp_t<opExp1_t, opExp2_t>(op1, op2);
}

template<typename opExp1_t,
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0 >
div_exp_t<opExp1_t, var_t<>> operator/(
	const opExp1_t& op1, const gVector& v2) {
	static_assert(opExp1_t::is_exp, "Not a vector expression !");
	return div_exp_t<opExp1_t, var_t<>>(op1, var_t<>(v2));
}

template<typename opExp1_t,
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0 >
div_exp_t<opExp1_t, scalar_t<>> operator/(
	const opExp1_t& op1,
	Scalar s2
	) {
	static_assert(opExp1_t::is_exp, "Not a vector expression !");
	return div_exp_t<opExp1_t, scalar_t<>>(op1, scalar_t<>(s2));
}

template<typename opExp2_t,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
div_exp_t<var_t<>, opExp2_t> operator/(
	const gVector& v1,
	const opExp2_t& op2) {
	static_assert(opExp2_t::is_exp, "Not a vector expression !");
	return div_exp_t<var_t<>, opExp2_t>(var_t<>(v1), op2);
}

template<typename opExp2_t,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
div_exp_t<scalar_t<>, opExp2_t > operator/(
	Scalar s1,
	const opExp2_t& op2
	) {
	static_assert(opExp2_t::is_exp, "Not a vector expression !");
	return div_exp_t<scalar_t<>, opExp2_t>(scalar_t<>(s1), op2);
}

extern div_exp_t<var_t<>, var_t<>> operator/(
	const gVector& v1,
	const gVector& v2
	);

extern div_exp_t<scalar_t<>, var_t<>> operator/(
	Scalar s,
	const gVector& v2
	);

extern div_exp_t<var_t<>, scalar_t<>> operator/(
	const gVector& v1,
	Scalar s
	);
/*****************************************************************************
	pow expression
****************************************************************************/

template<typename opExp1_t, typename opExp2_t, 
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
pow_exp_t<opExp1_t, opExp2_t> operator^(
	const opExp1_t& op1,
	const opExp2_t& op2) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return pow_exp_t<opExp1_t, opExp2_t>(op1, op2);
}

template<typename opExp1_t,
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0 >
pow_exp_t<opExp1_t, var_t<>> operator^(
	const opExp1_t& op1, const gVector& v2) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	return pow_exp_t<opExp1_t, var_t<>>(op1, var_t<>(v2));
}

template<typename opExp1_t,
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0 >
pow_exp_t<opExp1_t, scalar_t<>> operator^(
	const opExp1_t& op1, 
	Scalar s2
	) {
	static_assert(opExp1_t::is_exp, "Not a vector expression");
	return pow_exp_t<opExp1_t, scalar_t<>>(op1, scalar_t<>(s2));
}

template<typename opExp2_t,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
pow_exp_t<var_t<>, opExp2_t> operator^(
	const gVector& v1,
	const opExp2_t& op2) {
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return pow_exp_t<var_t<>, opExp2_t>(var_t<>(v1), op2);
}

template<typename opExp2_t,
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >
pow_exp_t<scalar_t<>, opExp2_t > operator^(
	Scalar s1,
	const opExp2_t& op2
	) {
	static_assert(opExp2_t::is_exp, "Not a vector expression");
	return pow_exp_t<scalar_t<>, opExp2_t>(scalar_t<>(s1), op2);
}

extern pow_exp_t<var_t<>, var_t<>> operator^(const gVector& v1, const gVector& v2);

extern pow_exp_t<scalar_t<>, var_t<>> operator^(Scalar s, const gVector& v2);

extern pow_exp_t<var_t<>, scalar_t<>> operator^(const gVector& v1, Scalar s);

};

#endif

#endif


