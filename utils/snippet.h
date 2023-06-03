#pragma once

#ifndef __SNIPPET_H
#define __SNIPPET_H

#include "string"
#include "algorithm"
#include "iostream"
#include "functional"
#include "fstream"
#include "Eigen/Eigen"
#include "stack"
#include "string"
#include "numeric"

#include "array"

#ifdef __linux__
#ifndef sprintf_s
#define sprintf_s(buf, ...) snprintf((buf), sizeof(buf), __VA_ARGS__)
#endif
#endif

namespace snippet {
	void trim(std::string& str);

	template<typename Vec>
	void write_vector(const std::string& filename, const Vec& v) {
		std::ofstream ofs(filename);
		for (int i = 0; i < v.size(); i++) {
			ofs << v[i] << std::endl;
		}
	}

	// align v1 to v2
	template<typename R>
	Eigen::Matrix<R, 3, 3> rotateAlign(const Eigen::Matrix<R, 3, 1>& v1, const Eigen::Matrix<R, 3, 1>& v2)
	{
		typedef Eigen::Matrix<R, 3, 1> vec3;
		typedef Eigen::Matrix<R, 3, 3> mat3;
		// for closed v1 and v2, v1 x (v2 - v1) is more stable than v1 x v2
		vec3 axis = v1.cross(v2 - v1);

		const R cosA = v1.dot(v2);
		if (cosA < -0.99) {
			printf("\033[31mRotateAlign: Parallel!\033[0m\n");
			Eigen::Vector3d z = v1.cross(Eigen::Vector3d::UnitY());
			if (z.squaredNorm() < 1e-4) {
				z = v1.cross(Eigen::Vector3d::UnitZ());
			}
			z.normalize();
			Eigen::Matrix3d z_skew;
			z_skew << 0, -z[2], z[1],
					z[2], 0, -z[0],
					-z[1], z[0], 0;
			Eigen::Matrix3d R1 = z * z.transpose() + z_skew;

			//if ((rotateAlign<double>(R1*v1, v2)*R1*v1 - v2).norm() > 1e-3) {
			//	throw - 1;
			//}

			return rotateAlign<double>(R1*v1, v2) * R1;
		}
		const R k = 1.0f / (1.0f + cosA);

		mat3 result;
		/*
		result ( (axis.x() * axis.x() * k) + cosA,
			(axis.y() * axis.x() * k) - axis.z(),
			(axis.z() * axis.x() * k) + axis.y(),
			(axis.x() * axis.y() * k) + axis.z(),
			(axis.y() * axis.y() * k) + cosA,
			(axis.z() * axis.y() * k) - axis.x(),
			(axis.x() * axis.z() * k) - axis.y(),
			(axis.y() * axis.z() * k) + axis.x(),
			(axis.z() * axis.z() * k) + cosA
			);
		*/
		result << (axis.x() * axis.x() * k) + cosA, (axis.x() * axis.y() * k) - axis.z(), (axis.x() * axis.z() * k) + axis.y(),
			(axis.y() * axis.x() * k) + axis.z(), (axis.y() * axis.y() * k) + cosA, (axis.y() * axis.z() * k) - axis.x(),
			(axis.z() * axis.x() * k) - axis.y(), (axis.z() * axis.y() * k) + axis.x(), (axis.z() * axis.z() * k) + cosA;
		return result;
	}

	template<typename T>
	bool is_include(const std::vector<T>& list, const T& element) {
		return std::find(list.begin(), list.end(), element) != list.end();
	}

	template<typename T>
	std::pair<T, T> sorted(const std::pair<T, T>& oldpair) {
		std::pair<T, T> newpair;
		if (oldpair.first < oldpair.second) {
			newpair.first = oldpair.first;
			newpair.second = oldpair.second;
		}
		else {
			newpair.second = oldpair.first;
			newpair.first = oldpair.second;
		}
		return newpair;
	}

	template<typename T, int N>
	std::array<T, N> sorted(const std::array<T, N>& oldarray) {
		std::array<T, N> newArray(oldarray);
		std::sort(newArray.begin(), newArray.end());
		return newArray;
	}

	template<typename T>
	bool intersected(const std::pair<T, T>& p1, const std::pair<T, T>& p2) {
		return p1.first == p2.first || p1.first == p2.second || p1.second == p2.first || p1.second == p2.second;
	}

	template<typename... Args>
	std::string formated(const char* format_str, Args... args) {
		char buf[1000];
		sprintf_s(buf, format_str, args...);
		return std::string(buf);
	}

	template<typename Iter>
	void format_log(std::ostream& os, Iter iter, Iter end, std::function<std::string(Iter)> formater) {
		for (; iter != end; iter++) {
			os << formater(iter);
		}
	}

	template<typename T>
	void remove_dup(std::vector<T>& dupvector) {
		std::sort(dupvector.begin(), dupvector.end());
		auto iter = std::unique(dupvector.begin(), dupvector.end());
		dupvector.erase(iter, dupvector.end());
	}

	template<typename T, int N>
	class circle_array_t :private std::array<T, N> {
		typedef std::array<T, N> BaseArray;
	public:
		T& operator[](size_t k) {
			return BaseArray::operator [](((k % N) + N) % N);
		}

		void set(T val) {
			for (int i = 0; i < N; i++) {
				(*this)[i] = val;
			}
		}

		template<typename... Args>
		circle_array_t(Args... args) :std::array<T, N>(args...) { }

	};

	class converge_criteria {
	private:
		int _maxcounter = 0;
		int _stopcounter = 0;
		int _itn = 0;
		double _thres = 1e-3;
		circle_array_t<double, 20> _oldvalue;
		int _nConstrain;
		std::vector<circle_array_t<double, 20>> _oldconstrain;
	public:
		converge_criteria(int nConstrain, int max_counter, double stop_thres) :
			_nConstrain(nConstrain),
			_oldconstrain(nConstrain),
			_maxcounter(max_counter),
			_thres(stop_thres)
		{
			_oldvalue.set(std::numeric_limits<double>::quiet_NaN());
			for (int i = 0; i < _oldconstrain.size(); i++) {
				_oldconstrain[i].set(std::numeric_limits<double>::quiet_NaN());
			}
		}

		bool update(double newvalue, double* newConstrain) {
			_oldvalue[_itn] = newvalue;
			for (int i = 0; i < _nConstrain; i++) {
				_oldconstrain[i][_itn] = newConstrain[i];
			}
			double fch = abs(_oldvalue[_itn] + _oldvalue[_itn - 1] - _oldvalue[_itn - 2] - _oldvalue[_itn - 3])
				/ (_oldvalue[_itn - 2] + _oldvalue[_itn - 3]);
			bool converged = true;
			if (!(fch < _thres)) {
				converged = false;
			}
			double violate_gch = -1;
			for (int i = 0; i < _nConstrain; i++) {
				double gch = abs(
					_oldconstrain[i][_itn] + _oldconstrain[i][_itn - 1]
					- _oldconstrain[i][_itn - 2] - _oldconstrain[i][_itn - 3])
					/ (_oldconstrain[i][_itn - 2] + _oldconstrain[i][_itn - 3]);
				violate_gch = (std::max)(violate_gch, gch);
				if (gch > 1e-3) {
					converged = false;
					break;
				}
			}

			if (converged) {
				_stopcounter++;
			}
			else {
				_stopcounter--;
				_stopcounter = (std::max)(_stopcounter, 0);
			}
			printf("[StopCheck] : fch %6.4lf%%(%6.4lf%%)  |   gch %6.4lf%%(%6.4lf%%)\n", fch * 100, _thres * 100, violate_gch * 100, _thres * 100);
			_itn++;
			return  _stopcounter > _maxcounter;
		}
	};

	template<int modulu = 0>
	struct Zp {
		int operator[](int n) {
			n -= modulu * (n / modulu);
			n += modulu;
			return n % modulu;
		}
	};

	template<>
	struct Zp<0> {
		int _modulu;
		Zp(int m) :_modulu(m) {}
		int operator[](int n) {
			n -= _modulu * (n / _modulu);
			n += _modulu;
			return n % _modulu;
		}
	};

	template<int N>
	inline size_t Round(size_t n) {
		size_t rmod = n % N;
		if (!rmod)
			return n;
		else {
			return (n / N + 1)*N;
		}
	}

	template <typename T, typename Compare>
	std::vector<std::size_t> sort_permutation(
		const std::vector<T>& vec,
		Compare compare = std::less<T>())
	{
		std::vector<std::size_t> p(vec.size());
		std::iota(p.begin(), p.end(), 0);
		std::sort(p.begin(), p.end(),
			[&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
		return p;
	};

	template <typename T>
	std::vector<T> apply_permutation(
		const std::vector<T>& vec,
		const std::vector<std::size_t>& p)
	{
		std::vector<T> sorted_vec(vec.size());
		std::transform(p.begin(), p.end(), sorted_vec.begin(),
			[&](std::size_t i) { return vec[i]; });
		return sorted_vec;
	};

	template<typename vec>
	Eigen::Matrix<double, -1, 1> vecJoin(const Eigen::MatrixBase<vec>& v1, const Eigen::MatrixBase<vec>& v2) {
		if (v1.cols() != 1 || v2.cols() != 1) {
			printf("\033[31mcannot join to matrix\033[0m\n");
			throw - 1;
		}
		Eigen::Matrix<double, -1, 1> v(v1.rows() + v2.rows());
		v << v1, v2;
		return v;
	}

	class Loger {
	private:
		std::stack<std::string>& _logstack;
		std::string _lastlog;
	public:
		Loger(std::stack<std::string>& logs, const std::string& logfile)
			: _logstack(logs), _lastlog(logfile)
		{
			_logstack.push(logfile);
		}

		~Loger() {
			if (_logstack.top() != _lastlog) {
				printf("\033[31mLoger> log stack top does not match last log file\033[0m ");
			}
			else {
				_logstack.pop();
			}
		}
	};

	template<typename T>
	struct SerialChar {
		circle_array_t<T, 100> _serial;
		int _counter = 0;
		void add(T num) {
			_serial[_counter++] = num;
		}
		bool arising(void) {
			bool isArising = true;
			for (int i = _counter - 10; i < _counter; i++) {
				if (_serial[i - 1] > _serial[i]) {
					isArising = false;
					break;
				}
			}
			return isArising;
		}
	};

	void stop_ms(int millisecond);
};


#endif

