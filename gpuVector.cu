#include"gpuVector.h"
//#include"cublas.h"
#include"cuda_runtime.h"
#include"lib.cuh"
#include"vector"
#include"gpuVector.cuh"

//#define __DEBUG_GVECTOR


namespace gv {
	gVector buf_vector;

	void gVector::build(size_t dim) {
		if (_size != dim) {
			clear();
			cudaMalloc(&_data, dim * sizeof(Scalar));
#ifdef __DEBUG_GVECTOR
			std::cout << "vector " << _data << " calling build with size " << dim << std::endl;
#endif
			_size = dim;
			cuda_error_check;
		}
		else {
			return;
		}
	}

	void gv::gVector::Init(size_t max_vec_size)
	{
		if (std::is_same<Scalar, double>::value) {
			init_cuda();
		}
		if (max_vec_size > buf_vector.size()) {
			buf_vector.resize(max_vec_size);
		}
	}

	void gv::gVector::resize(size_t dim, int)
	{
		cudaMalloc(&_data, dim * sizeof(Scalar));
		_size = dim;
	}

	template<typename Lambda>
	void apply_vector(gVector& v1, const gVector& v2, Lambda func) {
		if (v2.size() != v1.size()) printf("warning : using two vectors with unmatched size !");
		Scalar* v1data = v1.data();
		const Scalar* v2data = v2.data();
		auto merge = [=] __device__(int eid) {
			v1data[eid] = func(v1data[eid], v2data[eid]);
		};
		size_t gridSize, blockSize;
		make_kernel_param(&gridSize, &blockSize, v1.size(), 512);
		traverse_noret << <gridSize, blockSize >> > (v1.size(), merge);
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	void gVector::clear(void) {
		if (_data == nullptr) {
			if (_size == 0) { return; }
		}
		cudaFree(_data);
#ifdef __DEBUG_GVECTOR
		std::cout << "vector " << _data << " calling clear " << std::endl;
#endif
		_size = 0;
		_data = nullptr;
	}

	gVector::gVector(size_t dim, Scalar default_value) {
		_size = dim;
		cudaMalloc(&_data, _size * sizeof(Scalar));
#ifdef __DEBUG_GVECTOR
		std::cout << "vector " << _data << " constructing with size " << dim << std::endl;
#endif
		init_array(_data, default_value, _size);
		cuda_error_check;
	}

	gVector::~gVector(void) {
		if (_data != nullptr) {
			cudaFree(_data);
		}
#ifdef __DEBUG_GVECTOR
		std::cout << "vector " << _data << " deconstructing with size " << _size << std::endl;
#endif
		cuda_error_check;
	}

	//gVector::gVector(Scalar* host_ptr, size_t size) {
	//	_size = size;
	//	cudaMalloc(&_data, size * sizeof(Scalar));
	//	cudaMemcpy(_data, host_ptr, sizeof(Scalar)*size, cudaMemcpyHostToDevice);
	//	cuda_error_check;
	//}

	gVector::gVector(const gVector& v) {
		cudaMalloc(&_data, v.size() * sizeof(Scalar));
#ifdef __DEBUG_GVECTOR
		std::cout << "vector " << _data << " copy constructing with size " << v.size() << " from vector " << v.data() << std::endl;
#endif
		_size = v.size();
		cudaMemcpy(_data, v.data(), _size * sizeof(Scalar), cudaMemcpyDeviceToDevice);
		cuda_error_check;
	}

	//gv::gVector::gVector(gVector&& v) noexcept {
	//	Scalar* pt = _data;
	//	int ps = _size;
	//	_data = v.data();
	//	_size = v._size;
	//	v._data = pt;
	//	v._size = ps;
	//}

	const gVector& gVector::operator=(const gVector& v2) {
		if (size() != v2.size()) {
			clear();
			build(v2.size());
		}
		cudaMemcpy(data(), v2.data(), sizeof(Scalar)*v2.size(), cudaMemcpyDeviceToDevice);
#ifdef __DEBUG_GVECTOR
		std::cout << "vector " << _data << " copying from vector " << v2.data() << std::endl;
#endif
		cuda_error_check;
		return (*this);
	}

	const gv::gVectorMap& gv::gVectorMap::operator=(const gVector& v2) const
	{
		cudaMemcpy(_data, v2.data(), v2.size(), cudaMemcpyDeviceToDevice);
		cuda_error_check;
		return *this;
	}

	void gVector::download(Scalar* host_ptr) const {
		cudaMemcpy(host_ptr, data(), sizeof(Scalar)*size(), cudaMemcpyDeviceToHost);
		cuda_error_check;
	}

	void gVector::set(const Scalar* host_ptr) {
		cudaMemcpy(data(), host_ptr, sizeof(Scalar)*size(), cudaMemcpyHostToDevice);
		cuda_error_check;
	}

	const gVector& gVector::operator+=(const gVector& v2) {
		auto add = [=] __device__(Scalar v1, Scalar v2) {
			return v1 + v2;
		};
		apply_vector(*this, v2, add);
		return *this;
	}

	const gVector& gVector::operator-=(const gVector& v2) {
		auto minus = [=] __device__(Scalar v1, Scalar v2) {
			return v1 - v2;
		};
		apply_vector(*this, v2, minus);
		return *this;
	}

	const gVector& gVector::operator*=(const gVector& v2) {
		auto multiply = [=] __device__(Scalar v1, Scalar v2) {
			return v1 * v2;
		};
		apply_vector(*this, v2, multiply);
		return *this;
	}

	const gVector& gVector::operator/=(const gVector& v2) {
		auto divide = [=]__device__(Scalar v1, Scalar v2) {
			return v1 / v2;
		};
		apply_vector(*this, v2, divide);
		return *this;
	}

#ifndef __USE_GVECTOR_LAZY_EVALUATION
	gv::gVector gVector::operator*(const gVector& v2) const
	{
		gVector v(*this);
		v *= v2;
		return v;
	}

	gVector gVector::operator/(const gVector& v2) const {
		gVector v(*this);
		v /= v2;
		return v;
	}

	gv::gVector gv::gVector::operator/(Scalar s) const
	{
		gVector v(size());
		Scalar* src = _data;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, size(), 512);
		map << <grid_size, block_size >> > (v.data(), size(), [=] __device__(int eid) {
			return src[eid] / s;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return v;
	}

	gv::gVector gVector::operator+(const gVector& v2) const
	{
		gVector sum(*this);
		sum += v2;
		return sum;
	}

	gv::gVector gVector::operator-(const gVector& v2) const
	{
		gVector diff(*this);
		diff -= v2;
		return diff;
	}

	gv::gVector gVector::operator-(Scalar val) const
	{
		gVector diff(size());
		Scalar* ptr = diff.data();
		const Scalar* src = data();
		auto sub_kernel = [=] __device__(int eid) {
			return src[eid] - val;
		};
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(ptr, size(), sub_kernel);
		cudaDeviceSynchronize();
		cuda_error_check;
		return diff;
	}

	gv::gVector gv::gVector::operator-(void) const
	{
		gVector res(size());
		Scalar* src = _data;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map << <grid_size, block_size >> > (res.data(), size(), [=] __device__(int eid) {
			return -src[eid];
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return res;
	}

	gv::gVector gv::gVector::operator*(Scalar s) const
	{
		gVector res(size());
		Scalar* src = _data;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map << <grid_size, block_size >> > (res.data(), size(), [=] __device__(int eid) {
			return src[eid] * s;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return res;
	}
#endif

	const gVector& gVector::operator/=(Scalar s) {
		Scalar* ptr = _data;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map << <grid_size, block_size >> > (_data, size(), [=] __device__(int tid) { return ptr[tid] / s; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	void gVector::invInPlace(void)
	{
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map << <grid_size, block_size >> > (ptr, size(), [=] __device__(int tid) { return 1 / ptr[tid]; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return;
	}

	void gVector::swap(gVector& v2)
	{
		if (_size != v2.size()) {
			throw std::string("size does not match !");
		}
		auto ptr = _data;
		_data = v2._data;
		v2._data = ptr;
	}

	const gVector& gVector::operator*=(Scalar s) {
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(data(), size(), [=] __device__(int tid) { return ptr[tid] * s; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}


	void gVector::set(Scalar val) {
		init_array(data(), val, size());
	}

	__device__ bool read_bit(int* flag, int offset) {
		int bit32 = flag[offset / 32];
		return bit32 & (offset % 32);
	}

	void gv::gVector::set(int* filter, Scalar val)
	{
		int len = size();
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, len, 512);
		map << <grid_size, block_size >> > (size(), [=] __device__(int tid) {
			if (read_bit(filter, tid)) {
				ptr[tid] = val;
			}
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}


	void gVector::maximize(Scalar s)
	{
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(ptr, size(), [=]  __device__(int tid) {
			Scalar v = ptr[tid];
			return v > s ? v : s;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	void gVector::maximize(const gVector& v2)
	{
		Scalar* v1data = data();
		const Scalar* v2data = v2.data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(v1data, size(), [=] __device__(int tid) {
			Scalar val1 = v1data[tid];
			Scalar val2 = v2data[tid];
			return val1 > val2 ? val1 : val2;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	void gVector::minimize(Scalar s)
	{
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(ptr, size(), [=] __device__(int tid) {
			Scalar v = ptr[tid];
			return v < s ? v : s;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	void gVector::minimize(const gVector& v2)
	{
		Scalar* v1data = data();
		const Scalar* v2data = v2.data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(v1data, size(), [=] __device__(int tid) {
			Scalar val1 = v1data[tid];
			Scalar val2 = v2data[tid];
			return val1 < val2 ? val1 : val2;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

#ifndef __USE_GVECTOR_LAZY_EVALUATION
	gv::gVector gVector::max(const gVector& v2) const
	{
		gVector res(*this);
		res.maximize(v2);
		return res;
	}

	gv::gVector gVector::min(const gVector& v2) const
	{
		gVector res(*this);
		res.minimize(v2);
		return res;
	}

	gv::gVector gVector::max(Scalar v2) const
	{
		gVector res(*this);
		res.maximize(v2);
		return res;
	}

	gv::gVector gVector::min(Scalar v2) const
	{
		gVector res(*this);
		res.minimize(v2);
		return res;
	}
#else

#endif

	gVector::Scalar gVector::sum(void) const
	{
		//gVector tmp((size() + 511) / 512);
		Scalar res = parallel_sum(data(), buf_vector.data(), size());
		return res;
	}

	gVector::Scalar* gVector::begin(void) {
		return _data;
	}

	gVector::Scalar* gVector::end(void) {
		return _data + _size;
	}

	void gVector::clamp(Scalar lower, Scalar upper)
	{
		Scalar* ptr = data();
		auto clamp_kernel = [=] __device__(int eid) {
			Scalar val = ptr[eid];
			if (lower > val) return lower;
			if (upper < val) return upper;
			return  val;
		};
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(ptr, size(), clamp_kernel);
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	void gVector::clamp(Scalar* lower, Scalar* upper)
	{
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(ptr, size(), [=] __device__(int eid) {
			Scalar val = ptr[eid];
			Scalar low = lower[eid], up = upper[eid];
			if (low > val) return low;
			if (up < val) return up;
			return val;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	//gVector::Scalar gVector::operator[](int eid) const
	//{
	//	gVector::Scalar s;
	//	cudaMemcpy(&s, data() + eid, sizeof(Scalar), cudaMemcpyDeviceToHost);
	//	cuda_error_check;
	//	return s;
	//}

	void gv::gVector::clamp(gVector& vl, gVector& vu)
	{
		clamp(vl.data(), vu.data());
	}

	const gv::gElementProxy& gElementProxy::operator=(gVector::Scalar val)
	{
#ifdef __DEBUG_GVECTOR
		std::cout << "proxy assignment is called, val = " << val << std::endl;
#endif
		/// DEBUG
		//cuda_error_check;
		cudaMemcpy(address, &val, sizeof(gVector::Scalar), cudaMemcpyHostToDevice);
		/// DEBUG
		//cuda_error_check;
		//printf("-- addr = %p,  val = %6.2e\n", address, val);
		return (*this);
	}

	gv::gElementProxy::operator gv::gVector::Scalar(void) const {
		gVector::Scalar val;
#ifdef __DEBUG_GVECTOR
		std::cout << "type conversion is called, address = " << address << std::endl;
#endif
		cudaMemcpy(&val, address, sizeof(gVector::Scalar), cudaMemcpyDeviceToHost);
		return val;
	}

#ifndef __USE_GVECTOR_LAZY_EVALUATION
	gv::gVector operator*(gv::gVector::Scalar scale, const gVector& v)
	{
		gVector res(v);
		res *= scale;
		return res;
	}

	gv::gVector operator/(gv::gVector::Scalar nom, const gVector& v)
	{
		gVector res(v);
		Scalar* ptr = res.data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, res.size(), 512);
		map<<<grid_size,block_size>>>(ptr, v.size(), [=] __device__(int eid) {
			return nom / ptr[eid];
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return res;
	}

	gv::gVector gVector::operator^(Scalar exp_pow) const
	{
		gVector res(*this);
		Scalar* ptr = res.data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(ptr, res.size(), [=] __device__(int eid) {
			if (std::is_same<Scalar, float>::value) {
				return powf(ptr[eid], exp_pow);
			}
			else {
				return pow(ptr[eid], exp_pow);
			}
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return res;
	}

#endif

#ifdef __USE_GVECTOR_LAZY_EVALUATION

/*******************************************************************************************
*                         negate      expression                                           *
********************************************************************************************/
	negat_exp_t<var_t<>> operator-(const gVector& gvec) {
		return negat_exp_t<var_t<>>(var_t<>(gvec));
	}

/*******************************************************************************************
*                         minus      expression                                            *
********************************************************************************************/
	minus_exp_t<var_t<>, var_t<>> operator-(
		const gVector& v1,
		const gVector& v2
		) {
		return minus_exp_t<var_t<>, var_t<>>(var_t<>(v1), var_t<>(v2));
	}

	minus_exp_t<scalar_t<>, var_t<>> operator-(
		Scalar s,
		const gVector& v2
		) {
		return minus_exp_t<scalar_t<>, var_t<>>(scalar_t<>(s), var_t<>(v2));
	}

	minus_exp_t<var_t<>, scalar_t<>> operator-(
		const gVector& v1,
		Scalar s
		) {
		return minus_exp_t<var_t<>, scalar_t<>>(var_t<>(v1), scalar_t<>(s));
	}

/*******************************************************************************************
*                         add      expression                                              *
********************************************************************************************/
	add_exp_t<var_t<>, var_t<>> operator+(
		const gVector& v1,
		const gVector& v2
		) {
		return add_exp_t<var_t<>, var_t<>>(var_t<>(v1), var_t<>(v2));
	}

	add_exp_t<scalar_t<>, var_t<>> operator+(
		Scalar s,
		const gVector& v2
		) {
		return add_exp_t<scalar_t<>, var_t<>>(scalar_t<>(s), var_t<>(v2));
	}

	add_exp_t<var_t<>, scalar_t<>> operator+(
		const gVector& v1,
		Scalar s
		) {
		return add_exp_t<var_t<>, scalar_t<>>(var_t<>(v1), scalar_t<>(s));
	}

/*******************************************************************************************
*                         multiply      expression                                         *
********************************************************************************************/
	multiply_exp_t<var_t<>, var_t<>> operator*(
		const gVector& v1,
		const gVector& v2
		) {
		return multiply_exp_t<var_t<>, var_t<>>(var_t<>(v1), var_t<>(v2));
	}

	multiply_exp_t<scalar_t<>, var_t<>> operator*(
		Scalar s,
		const gVector& v2
		) {
		return multiply_exp_t<scalar_t<>, var_t<>>(scalar_t<>(s), var_t<>(v2));
	}

	multiply_exp_t<var_t<>, scalar_t<>> operator*(
		const gVector& v1,
		Scalar s
		) {
		return multiply_exp_t<var_t<>, scalar_t<>>(var_t<>(v1), scalar_t<>(s));
	}

/*******************************************************************************************
*                         divide      expression                                           *
********************************************************************************************/
	div_exp_t<var_t<>, var_t<>> operator/(
		const gVector& v1,
		const gVector& v2
		) {
		return div_exp_t<var_t<>, var_t<>>(var_t<>(v1), var_t<>(v2));
	}

	div_exp_t<scalar_t<>, var_t<>> operator/(
		Scalar s,
		const gVector& v2
		) {
		return div_exp_t<scalar_t<>, var_t<>>(scalar_t<>(s), var_t<>(v2));
	}

	div_exp_t<var_t<>, scalar_t<>> operator/(
		const gVector& v1,
		Scalar s
		) {
		return div_exp_t<var_t<>, scalar_t<>>(var_t<>(v1), scalar_t<>(s));
	}

/*******************************************************************************************
*                         pow    expression                                                *
********************************************************************************************/
	pow_exp_t<var_t<>, var_t<>> operator^(
		const gVector& v1,
		const gVector& v2
		) {
		return pow_exp_t<var_t<>, var_t<>>(var_t<>(v1), var_t<>(v2));
	}

	pow_exp_t<scalar_t<>, var_t<>> operator^(
		Scalar s,
		const gVector& v2
		) {
		return pow_exp_t<scalar_t<>, var_t<>>(scalar_t<>(s), var_t<>(v2));
	}

	pow_exp_t<var_t<>, scalar_t<>> operator^(
		const gVector& v1,
		Scalar s
		) {
		return pow_exp_t<var_t<>, scalar_t<>>(var_t<>(v1), scalar_t<>(s));
	}


#endif

	gv::gVector::Scalar* gv::gVector::get_dump_buf(void)
	{
		return buf_vector.data();
	}

	gv::gVector gVector::slice(int start, int end) const
	{
		gVector res(end - start);
		const Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, res.size(), 512);
		map<<<grid_size,block_size>>>(res.data(), res.size(), [=]  __device__(int eid) {
			return ptr[eid + start];
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return res;
	}


	std::vector<gVector::Scalar> gVector::slice2host(int start, int end) const
	{
		if (end < start || start < 0 || end >= size()) {
			throw std::string("invalid indices !");
		}
		std::vector<Scalar> res(end - start);
		cudaMemcpy(res.data(), data(), sizeof(Scalar)*size(), cudaMemcpyDeviceToHost);
		cuda_error_check;
		return res;
	}

	gVector::Scalar gVector::dot(const gVector& v2) const
	{
		//gVector tmp((size() + 511) / 512);
		return ::dot(data(), v2.data(), buf_vector.data(), size());
	}

	Scalar gVector::max(void) const
	{
		//gVector tmp((size() + 511) / 512);
		return parallel_max(data(), buf_vector.data(), size());
	}

	Scalar gVector::min(void) const
	{
		//gVector tmp((size() + 511) / 512);
		return parallel_min(data(), buf_vector.data(), size());
	}

	Scalar gv::gVector::min_positive(void) const
	{
		gVector tmp(size());
		Scalar* src = _data;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, size(), 512);
		map << <grid_size, block_size >> > (tmp.data(), size(), [=] __device__(int eid) {
			Scalar val = src[eid];
			if (val < 0) {
				val = 1e30;
			}
			return val;
		});
		cudaDeviceSynchronize();
		cuda_error_check;

		return tmp.min();
	}

	Scalar gVector::norm(void) const
	{
		return sqrt(dot(*this));
	}

	Scalar gv::gVector::infnorm(void) const
	{
		//gv::gVector tmp((size() + 511) / 512);
		return parallel_maxabs(_data, buf_vector.data(), size());
	}

	Scalar gVector::sqrnorm(void) const 
	{
		return dot(*this);
	}

	void gv::gVector::Sqrt(void)
	{
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, size(), 512);
		Scalar* src = _data;
		map << <grid_size, block_size >> > (_data, size(), [=] __device__(int tid) {
			return sqrt(src[tid]);
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	gVectorMap gVector::Map(Scalar* ptr, size_t size) {
		return gVectorMap(ptr, size);
	}


	gv::gVector gv::gVector::concated_one(const gVector& v2) const
	{
		gVector result(size() + v2.size());
		cudaMemcpy(result.data(), data(), sizeof(Scalar)*size(), cudaMemcpyDeviceToDevice);
		cudaMemcpy(result.data() + size(), v2.data(), sizeof(Scalar)*v2.size(), cudaMemcpyDeviceToDevice);
		cuda_error_check;
		return result;
	}

	gv::gVector gv::gVector::concated_one(Scalar val) const
	{
		gVector result(size() + 1);
		cudaMemcpy(result.data(), data(), size() * sizeof(Scalar), cudaMemcpyDeviceToDevice);
		result[size()] = val;
		cuda_error_check;
		return result;
	}

	void gv::gVector::concate_one(const gVector& v2)
	{
		gVector old_vec = *this;
		size_t new_size = v2.size() + size();
		clear();
		build(new_size);
		cudaMemcpy(data(), old_vec.data(), sizeof(Scalar)*old_vec.size(), cudaMemcpyDeviceToDevice);
		cudaMemcpy(data() + old_vec.size(), v2.data(), sizeof(Scalar)*v2.size(), cudaMemcpyDeviceToDevice);
		cuda_error_check;
	}

	void gv::gVector::concate_one(Scalar val)
	{
		gVector old_vec = *this;
		size_t new_size = size() + 1;
		clear();
		build(new_size);
		cudaMemcpy(data(), old_vec.data(), old_vec.size() * sizeof(Scalar), cudaMemcpyDeviceToDevice);
		cudaMemcpy(data() + old_vec.size(), &val, sizeof(Scalar), cudaMemcpyHostToDevice);
		cuda_error_check;
	}

	void test_lazy_eval(void) {
		gVector v1(1000), v2(1000), v3(1000);
		v1.set(1);
		v2.set(2);
		v3.set(3);

		auto ex = (v1 + v2)*v3;
		std::cout << "expression size " << ex.size() << std::endl;
		gVector v4 = ((v1 + v2)*v3 + (v3 + v3)*(v3 - v1)).min(v1);
		gVector v5 = (v1 - v2).max(v3) + v1;
		Scalar v45 = ((v1 + v2)*v3 + (v3 + v3)*(v3 - v1)).min(v1).dot((v1 - v2).max(v3) + v1);
		v4.toMatlab("v4");
		v5.toMatlab("v5");
		std::cout << "v45 = " << v45 << std::endl;
		//constexpr bool ic = is_expression<multiply_exp_t<add_exp_t<var_t, var_t>, var_t>>::value;
		//constexpr bool ib = std::is_base_of<exp_base_t, add_exp_t<var_t, var_t>>::value;
		gVector v6 = (v1 + (v2 ^ 3.f) - v3).min(v4)*(v2 ^ 2) / v3 - v5;
		v6.toMatlab("v6");
	}

	void test_nested_lambda(void) {
		//printf("========================================================================\n");
		//printf("starting nested lambda test ...\n");
		//Scalar* buf;
		//cudaMalloc(&buf, 1000 * sizeof(Scalar));
		//int a = 1;
		//int b = 2;
		//auto ker1 = [=]__device__(void) {
		//	return a;
		//};
		//auto ker2 = [=] __device__(Scalar* ptr) {
		//	*ptr = ker1() - b;
		//};
		//Foo foo;
		////auto ker3 = gen_lam();
		//size_t grid_size, block_size;
		//make_kernel_param(&grid_size, &block_size, 1, 64);
		//traverse_noret << <grid_size, block_size >> > (1, [=] __device__(int eid){
		//	ker2(buf);
		//	buf[0] = foo.a;
		//});
		//cudaDeviceSynchronize();
		//cuda_error_check;

		//Scalar result;
		//cudaMemcpy(&result, buf, sizeof(Scalar), cudaMemcpyDeviceToHost);

		//printf("result is %f\n", result);
		//return;
	}



	void test_gVector(void) {
		gVector::Init(100000);
		{
			gVector v1(10000), v2(10000), v3(10000);
			v1.set(1);
			v2.set(1.5);
			v3.set(3);
			//constexpr bool val = decltype(v1 - v2)::is_exp;
			gVector v4 = (v1 - v2) / v3 + v1 * v1 - 3;

			std::cout << "max of v4 " << v4.max() << std::endl;
			std::cout << "min of v4 " << v4.min() << std::endl;
			std::cout << "sum of v4 " << v4.sum() << std::endl;
			std::cout << "norm of v4 " << v4.norm() << std::endl;
			std::cout << "norm of v4 / 2 " << (v4 / 2).norm() << std::endl;


			// test dot
			std::cout << "v2'*(v1+v3) = " << v2.dot(v1 + v3) << std::endl;

			// test map
			gVector v5 = gVector::Map(v4.data(), 1000);
			std::cout << "max of v5 " << v5.max() << std::endl;
			std::cout << "min of v5 " << v5.min() << std::endl;
			std::cout << "norm of v5 " << v5.norm() << std::endl;
			std::cout << "norm of v5 * 2 " << (v5 * 2).norm() << std::endl;

			// test index access
			gVector v6 = gVector::Map(v5.data(), 10);
			std::cout << "old value : " << std::endl;
			for (int i = 0; i < 10; i++) {
				Scalar val = v6[i];
				printf("v6[%d] = %f\n", i, val);
				v6[i] = val - 1;
			}

			std::cout << "new value : " << std::endl;
			for (int i = 0; i < 10; i++) {
				Scalar val = v6[i];
				printf("v6[%d] = %f\n", i, val);
			}

			// test max/minimize 
			v5.maximize(1);
			printf("maximized v5 max = %f\n", v5.max());
			printf("maximized v5 min = %f\n", v5.min());

			v5.minimize(-1);
			printf("minimized v5 max = %f\n", v5.max());
			printf("minimized v5 min = %f\n", v5.min());

			printf("max(v5,2) norm = %f\n", v5.max(2).norm());

			// test concate
			gVector v7 = v1.concated(v2, v3);
			printf("size of v7 = %d\n", v7.size());
			printf("max of v7 = %f \n", v7.max());
			printf("min of v7 = %f \n", v7.min());

			gVector v8;
			v8.concate(gVector(2, 1), gVector(2, 2), gVector(2, 3), 4, 4);
			printf("v8 = \n");
			for (int i = 0; i < v8.size(); i++) {
				printf("%f\n", Scalar(v8[i]));
			}

		}

		// test nested lambda
		test_nested_lambda();

		// test lazy evaluation
		test_lazy_eval();
	}

#if 0
	gv::gMat::gMat(int rows, int cols, gHostPtr p)
		:_rows(rows), _cols(cols)
	{
		cudaMalloc(&_ptr, sizeof(Scalar)*rows*cols);
		cudaMemcpy(_ptr, p.get(), sizeof(Scalar) * rows * cols, cudaMemcpyHostToDevice);
	}

	gv::gMat::gMat(int rows, int cols, gDevicePtr p)
		:_rows(rows), _cols(cols)
	{
		cudaMalloc(&_ptr, sizeof(Scalar) * rows * cols);
		cudaMemcpy(_ptr, p.get(), sizeof(Scalar) * rows * cols, cudaMemcpyDeviceToDevice);
	}

	gv::gMat::gMat(gMat&& gm)
	{
		if (_rows != gm._rows || _cols != gm._cols) {
			_devIpiv = nullptr;
		}
		_rows = gm._rows;
		_cols = gm._cols;
		std::swap(_ptr, gm._ptr);
	}

	gv::gMat::~gMat(void)
	{
		cudaFree(_ptr);
		if (_devIpiv != nullptr) {
			cudaFree(_devIpiv);
		}
		if (_workspace != nullptr) {
			cudaFree(_workspace);
		}
	}

	void gv::gMat::compute(void)
	{
		cusolverDnCreate(&_handle);
		int newBufLen;
		cusolverDnDgetrf_bufferSize(_handle, _rows, _cols, _ptr, _rows, &newBufLen);
		if (newBufLen > _buflen[0]) {
			if (_workspace != nullptr) {
				cudaFree(_workspace);
			}
			_workspace = (Scalar*)cudaMalloc(&_workspace, sizeof(Scalar) * newBufLen);
		}
		_buflen[0] = newBufLen;
		if (_devIpiv == nullptr) {
			cudaMalloc(&_devIpiv, sizeof(double)*(_rows > _cols ? _cols : _rows));
		}
		cusolverDnDgetrf(_handle, _rows, _cols, _ptr, _rows, _workspace, _devIpiv, (int*)(void*)(buf_vector.data() + _rows));
		if (!std::is_same<Scalar, double>::value) {
			std::cout << "\033[31m" << "gMat solver Type error !" << "\033[0m" << std::endl;
		}
	}

	gv::gVector gv::gMat::solve(gVector& v)
	{
		if (!_computed) compute();
		gVector x(v);
		cusolverDnDgetrs(_handle, CUBLAS_OP_N, _rows, 1, _ptr, _rows, _devIpiv, x.data(), _rows, (int*)(void*)buf_vector.data());
		return x;
	}

	void gv::gMat::solveInplace(gVector& v)
	{
		if (!_computed) compute();
		cusolverDnDgetrs(_handle, CUBLAS_OP_N, _rows, 1, _ptr, _rows, _devIpiv, v.data(), _rows, (int*)(void*)buf_vector.data());
		return;
	}

	void gv::gMat::update(gDevicePtr p)
	{
		Scalar* ptr = p.get();
		cudaMemcpy(_ptr, ptr, sizeof(Scalar) * _rows * _cols, cudaMemcpyDeviceToDevice);
		_computed = false;
	}
#endif 
};




