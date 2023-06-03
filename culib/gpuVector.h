#pragma once

#ifndef __GPU_VECTOR_H
#define __GPU_VECTOR_H

//#define __GVECTOR_WITH_MATLAB
#define __USE_GVECTOR_LAZY_EVALUATION

//#define GVector_USE_DOUBLE

#include "vector"
#include "cusolverDn.h"
#include <string>


namespace gv {

class gVectorMap;
class gVector;
class gElementProxy;
	
#ifdef GVector_USE_DOUBLE
	typedef double Scalar;
#else
	typedef float Scalar;
#endif
#ifdef __USE_GVECTOR_LAZY_EVALUATION
	template<typename... > struct is_expression;
	template<typename, typename> struct min_exp_t;
	template<typename, typename> struct max_exp_t;
	template<typename> struct sqrt_exp_t;
	template<typename, typename > struct map_exp_t;
	template<typename T = Scalar, typename std::enable_if<std::is_scalar<T>::value, int >::type = 0> struct scalar_t;
	template<typename T = gVector, typename std::enable_if<std::is_same<T, gVector>::value, int>::type = 0> struct var_t;
	template<typename, typename >struct dot_exp_t;

#else
#endif

	struct gPtrBase {
		Scalar* _ptr = nullptr; 
		gPtrBase(Scalar* p) :_ptr(p) {}
	};

	struct gDevicePtr :gPtrBase {
		explicit gDevicePtr(Scalar* p) : gPtrBase(p) {}
		operator Scalar*() { return _ptr; }
		Scalar* get() { return _ptr; }
	};

	struct gHostPtr :gPtrBase {
		explicit gHostPtr(Scalar* p) : gPtrBase(p) {}
		operator Scalar*() { return _ptr; }
		Scalar* get() { return _ptr; }
	};

class gVector {
public:
	// typedef Scalar Scalar;
private:
	Scalar* _data;
	size_t _size;

	void build(size_t dim);

	friend class gVectorMap;

	template<typename Lambda>friend  void apply_vector(gVector& v1, const gVector& v2, Lambda func);

protected:
	gVector(Scalar* data_ptr, size_t size) :_data(data_ptr), _size(size) {}

protected:
	auto& _Get_data(void) { return _data; }
	auto& _Get_size(void) { return _size; }
public:
	Scalar*& data() { return _data; }

	const Scalar* data() const { return _data; }

	size_t size() const { return _size; }

	void clear(void);

	bool empty(void) const { return _size == 0; }

	void swap(gVector& v2);

	void resize(size_t dim);

	// for init, this version will not free buf
	void resize(size_t dim, int);

	void set(Scalar val);

	void set(int* filter, Scalar val);

	void set(const Scalar* host_ptr);

	explicit gVector(size_t dim, Scalar default_value = 0);

	gVector(void) :_size(0), _data(nullptr) {}

	virtual ~gVector(void);

	//gVector(Scalar* host_ptr, size_t size);

	gVector(const gVector& v);

	//gVector(gVector&& v) noexcept;

public:
	const gVector& operator=(const gVector& v2);

	void download(Scalar* host_ptr) const;

	const gVector& operator+=(const gVector& v2);

	const gVector& operator-=(const gVector& v2);

	const gVector& operator*=(const gVector& v2);

	const gVector& operator/=(const gVector& v2);

	const gVector& operator/=(Scalar s);

	const gVector& operator*=(Scalar s);

#ifdef __USE_GVECTOR_LAZY_EVALUATION

	template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0 >
	const gVector& operator+=(const expr_t& expr) {
		size_t expr_dim = expr.size();
		if (expr_dim != _size) {
			throw std::string("unmatched vector size !");
		}
		((*this) + expr).launch(_data, expr_dim);
		return *this;
	}

	template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0 >
	const gVector& operator-=(const expr_t& expr) {
		size_t expr_dim = expr.size();
		if (expr_dim != _size) {
			throw std::string("unmatched vector size !");
		}
		((*this) - expr).launch(_data, expr_dim);
		return *this;
	}

	template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0 >
	const gVector& operator/=(const expr_t& expr) {
		size_t expr_dim = expr.size();
		if (expr_dim != _size) {
			throw std::string("unmatched vector size !");
		}
		((*this) / expr).launch(_data, expr_dim);
		return *this;
	}

	template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0 >
	const gVector& operator*=(const expr_t& expr) {
		size_t expr_dim = expr.size();
		if (expr_dim != _size) {
			throw std::string("unmatched vector size !");
		}
		((*this) * expr).launch(_data, expr_dim);
		return *this;
	}

#endif

#ifndef __USE_GVECTOR_LAZY_EVALUATION

	gVector operator/(const gVector& v2) const;

	gVector operator/(Scalar s) const;

	gVector operator*(const gVector& v2) const;

	gVector operator*(Scalar s) const;

	gVector operator^(Scalar pow) const;

	gVector operator+(const gVector& v2) const;

	gVector operator-(const gVector& v2) const;

	gVector operator-(Scalar val) const;

	gVector operator-(void) const;

#endif

#ifdef __USE_GVECTOR_LAZY_EVALUATION


#if 1
	template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0 >
	const gVector& operator=(const expr_t& expr) {
		size_t expr_dim = expr.size();
		if (expr_dim != size()) {
			clear();
			build(expr_dim);
		}
		expr.launch(_data, expr_dim);
		return *this;
	}

	template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0>
	gVector(const expr_t& expr) {
		size_t expr_dim = expr.size();
		resize(expr_dim, 0);
		expr.launch(_data, expr_dim);
	}
#else
	template<typename expr_t>
	const gVector& operator=(const expr_t& expr) {
		static_assert(expr_t::is_exp, "Not a expression");
		size_t expr_dim = expr.size();
		if (expr_dim != size()) {
			clear();
			build(expr_dim);
		}
		expr.launch(_data, expr_dim);
	}

	template<typename expr_t>
	gVector(const expr_t& expr) {
		static_assert(expr_t::is_exp, "Not a expression");
		size_t expr_dim = expr.size();
		build(expr_dim);
		expr.launch(_data, expr_dim);
	}
#endif

	template<typename expr_t>
	const gVector& operator*=(const typename std::enable_if<expr_t::is_exp, expr_t>::type& expr) {
		auto new_expr = (*this)*expr;
		size_t expr_dim = expr.size();
		new_expr.launch(_data, expr_dim);
		return *this;
	}

	template<typename expr_t>
	const gVector& operator/=(const typename std::enable_if<expr_t::is_exp, expr_t>::type& expr) {
		auto new_expr = (*this)/expr;
		size_t expr_dim = expr.size();
		new_expr.launch(_data, expr_dim);
		return *this;
	}

	template<typename expr_t>
	const gVector& operator+=(const typename std::enable_if<expr_t::is_exp, expr_t>::type& expr) {
		auto new_expr = (*this) + expr;
		size_t expr_dim = expr.size();
		new_expr.launch(_data, expr_dim);
		return *this;
	}

	template<typename expr_t>
	const gVector& operator-=(const typename std::enable_if<expr_t::is_exp, expr_t>::type& expr) {
		auto new_expr = (*this) - expr;
		size_t expr_dim = expr.size();
		new_expr.launch(_data, expr_dim);
		return *this;
	}
#endif

	//Scalar operator[](int eid) const;

	gElementProxy operator[](int eid);

	void invInPlace(void);

	void maximize(Scalar s);

	void maximize(const gVector& v2);

	void minimize(Scalar s);

	void minimize(const gVector& v2);

#ifndef __USE_GVECTOR_LAZY_EVALUATION

	gVector max(const gVector& v2) const;

	gVector min(const gVector& v2) const;

	gVector max(Scalar v2) const;

	gVector min(Scalar v2) const;
#else
	template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0, typename vec_t = gVector>
	min_exp_t<var_t<vec_t>, opExp_t> min(const opExp_t& op2) const{
		return min_exp_t<var_t<vec_t>, opExp_t>(var_t<vec_t>(*this), op2);
	}

	template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0, typename vec_t = gVector>
	max_exp_t<var_t<vec_t>, opExp_t> max(const opExp_t& op2) const {
		return max_exp_t<var_t<vec_t>, opExp_t>(var_t<vec_t>(*this), op2);
	}

	template<typename vec_t = gVector, typename Scalar_type = Scalar,
		typename std::enable_if<std::is_scalar<Scalar_type>::value, int>::type = 0>
	min_exp_t<var_t<vec_t>, scalar_t<Scalar_type>> min(Scalar_type op2) const {
		return min_exp_t<var_t<vec_t>, scalar_t<Scalar_type>>(var_t<vec_t>(*this), op2);
	}

	template<typename vec_t = gVector, typename Scalar_type = Scalar,
		typename std::enable_if<std::is_scalar<Scalar_type>::value, int>::type = 0>
	max_exp_t<var_t<vec_t>, scalar_t<Scalar_type>> max(Scalar_type op2)const {
		return max_exp_t<var_t<vec_t>, scalar_t<Scalar_type>>(var_t<vec_t>(*this), op2);
	}

	template<typename Lambda, typename vec_t = gVector>
	map_exp_t<var_t<vec_t>, Lambda> fmap(Lambda func) {
		return map_exp_t<var_t<vec_t>, Lambda>(var_t<vec_t>(*this), func);
	}

#endif

	Scalar max(void) const;

	Scalar min(void) const;

	Scalar min_positive(void) const;

	Scalar norm(void) const;

	Scalar infnorm(void) const;

	Scalar sqrnorm(void) const;

	void Sqrt(void);

	template<typename Lambda, typename vec_t = gVector>
	void mapInplace(Lambda func) {
		auto expr = fmap<Lambda, vec_t>(func);
		expr.launch(_data, _size);
	}

	void clamp(Scalar lower, Scalar upper);

	void clamp(Scalar* lower, Scalar* upper);

	void clamp(gVector& vl, gVector& vu);

	gVector slice(int start, int end) const;

	gVector concated_one(const gVector& v2) const;

	gVector concated_one(Scalar val) const;

	void concate_one(const gVector& v2);

	void concate_one(Scalar val);

	template<typename Arg0, typename... Args>
	gVector concated(Arg0 arg0, Args... args) {
		return concated_one(arg0).concated(args...);
	}

	template<typename Arg0, typename... Args>
	void concate(const Arg0& arg0, Args... args) {
		concate_one(arg0);
		concate(args...);
	}

	void concate(void) { return; }

	gVector concated(void) const { return *this; }

	std::vector<Scalar> slice2host(int start, int end) const;

	static void Init(size_t max_vec_size);

	static Scalar* get_dump_buf(void);

	Scalar sum(void) const;

	Scalar dot(const gVector& v2) const;

#ifdef __USE_GVECTOR_LAZY_EVALUATION
	template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0, typename T = gVector>
	Scalar dot(const opExp_t& ex) {
		return var_t<T>(*this).dot(ex);
	}
#endif

	Scalar* begin(void);

	Scalar* end(void);

	static gVectorMap Map(Scalar* ptr, size_t size);

	void toMatlab(const char* vName);
	static void toMatlab(const char* vName, std::vector<gVector*>& vecs);
	static void toMatlab(const char* vName, std::vector<Scalar*>& vecs, int dim);
 };

#ifndef __USE_GVECTOR_LAZY_EVALUATION

gVector operator*(Scalar scale, const gVector& v);

gVector operator/(Scalar nom, const gVector& v);

#endif

class gVectorMap 
	:public gVector
{
public:
	gVectorMap(Scalar* data_ptr, size_t size) :gVector(data_ptr, size) {}
	~gVectorMap(void) override;
	const gVectorMap& operator=(const gVector& v2) const;

	template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0>
	const gVectorMap& operator=(const expr_t& expr) {
		size_t expr_dim = expr.size();
		if (expr_dim != _size) {
			throw std::string("unmatched vector size !");
		}
		expr.launch(_data, expr_dim);
		return *this;
	}
};

class gElementProxy {
	Scalar* address;
public:
	explicit gElementProxy(Scalar* ptr) : address(ptr) {}

	const gElementProxy& operator=(Scalar val);

	operator Scalar(void) const;
};

#if 0
class gMat {
private:
	typedef Scalar Scalar;
	Scalar* _ptr = nullptr;
	int _rows = 0, _cols = 0;

	// >cusolver part
	cusolverDnHandle_t _handle;
	bool _computed = false;
	int _buflen[1] = { 0 };
	int* _devIpiv = nullptr;
	Scalar *_workspace;
	void compute(void);
	// <cusolver_part

public:
	gMat(int rows, int cols, gHostPtr p);
	gMat(int rows, int cols, gDevicePtr p);
	gMat(gMat&& gm);
	~gMat(void);

	void update(gDevicePtr p);

	gVector solve(gVector& v);

	void solveInplace(gVector& v);
};
#endif

}

#endif

