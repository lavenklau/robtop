#pragma once

#ifndef _BINARY_IO_H
#define _BINARY_IO_H

#include "vector"
#include "set"
#include "list"
#include "type_traits"

namespace bio {

	//<

	//template<typename T>
	//struct has_const_iterator
	//{
	//private:
	//	typedef char                      yes;
	//	typedef struct { char array[2]; } no;

	//	template<typename C> static yes test(typename C::const_iterator*);
	//	template<typename C> static no  test(...);
	//public:
	//	static const bool value = sizeof(test<T>(0)) == sizeof(yes);
	//	typedef T type;
	//};

	//template <typename T>
	//struct has_begin_end
	//{
	//	template<typename C> static char(&f(typename std::enable_if<
	//		std::is_same<decltype(static_cast<typename C::const_iterator(C::*)() const>(&C::begin)),
	//		typename C::const_iterator(C::*)() const>::value, void>::type*))[1];

	//	template<typename C> static char(&f(...))[2];

	//	template<typename C> static char(&g(typename std::enable_if<
	//		std::is_same<decltype(static_cast<typename C::const_iterator(C::*)() const>(&C::end)),
	//		typename C::const_iterator(C::*)() const>::value, void>::type*))[1];

	//	template<typename C> static char(&g(...))[2];

	//	static bool const beg_value = sizeof(f<T>(0)) == 1;
	//	static bool const end_value = sizeof(g<T>(0)) == 1;
	//};

	//template<typename T>
	//struct is_container : std::integral_constant<bool, has_const_iterator<T>::value && has_begin_end<T>::beg_value && has_begin_end<T>::end_value>
	//{ };

	// refer to https://stackoverflow.com/questions/9407367/determine-if-a-type-is-an-stl-container-at-compile-time >


	//template<typename T, typename std::enable_if<std::is_array<T>::value, int>::type = 0>
	//size_t write_branch(std::ostream& os, std::vector<T>& treedata) {
	//	size_t size_account = 0;
	//	if (std::is_class<T>::value) {
	//		for (auto iter = treedata.begin(); iter != treedata.end(); iter++) {
	//			size_account += write_branch(os, *iter);
	//		}
	//	}
	//	else {
	//		os.write(&treedata[i], sizeof(treedata[i]));
	//	}
	//	return size_account;
	//}
	
	template<typename T, typename std::enable_if<std::is_scalar<T>::value, void*>::type = nullptr>
	void write_vector(const std::string& filename, const std::vector<T>& datavector) {
		std::ofstream ofs(filename, std::ios::binary);
		if (!ofs.is_open()) {
			std::cout << "\033[31m" << "Cannot open file " << filename << "\033[0m" << std::endl;
			return;
		}
		ofs.write((char*)&datavector[0], sizeof(T)*datavector.size());
		ofs.close();
	}

	// write [v[0][0] v[1][0] v[2][0] v[0][1] v[1][1] v[2][1] v[0][2] v[1][2] v[2][2] ... ]
	template<typename T, int N, typename std::enable_if<std::is_scalar<T>::value, void*>::type = nullptr>
	void write_vectors(const std::string& filename, const std::vector<T>(&datavectors)[N], bool transpose = false) {
		std::ofstream ofs(filename, std::ios::binary);
		int vecsize = datavectors->size();
		for (int i = 0; i < datavectors->size(); i++) {
			for (int j = 0; j < N; j++) {
				ofs.write((char*)&datavectors[j][i], sizeof(T));
			}
		}
		ofs.close();
	}

	template<typename T, typename std::enable_if<std::is_scalar<T>::value, void*>::type = nullptr>
	bool read_vector(const std::string& filename, std::vector<T>& datavector) {
		std::ifstream ifs(filename, std::ios::binary);
		if (!ifs.is_open()) return false;
		ifs.seekg(0, std::ios::end);
		size_t filelen = ifs.tellg();
		ifs.seekg(0, std::ios::beg);
		datavector.resize(filelen / sizeof(T));
		for (int i = 0; i < datavector.size(); i++) {
			T value = 0;
			ifs.read((char*)&value, sizeof(T));
			datavector[i] = value;
		}
		ifs.close();
		return true;
	}
};

#endif
