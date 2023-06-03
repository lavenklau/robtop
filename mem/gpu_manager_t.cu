#include "cuda_runtime.h"
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
//#include "lib.cuh"
#include "gpu_manager_t.h"
#include "cudaCommon.cuh"
//#include "matlab_utils.h"


typedef double Scaler;


void* mallocDeviceMemory(size_t size) {
	void* ptr;
	cudaMalloc(&ptr, size);
	cuda_error_check;
	return ptr;
}

void deleteDeviceMemory(void* ptr) {
	cudaFree(ptr);
	cuda_error_check;
}

gpu_manager_t::gpu_buf_t::gpu_buf_t(const std::string& name, size_t size)
	:std::unique_ptr<void, std::function<void(void*)>>(
		mallocDeviceMemory(size),
		deleteDeviceMemory
		)
{
	cuda_error_check;
	_desc = name;
	_size = size;
}

gpu_manager_t::gpu_buf_t::gpu_buf_t(gpu_buf_t&& tmp_buf) noexcept
	:unique_void_ptr(std::move(tmp_buf))
{
	_desc = tmp_buf._desc;
	_size = tmp_buf._size;
	cuda_error_check;
}

gpu_manager_t::gpu_buf_t& gpu_manager_t::gpu_buf_t::operator=(gpu_buf_t&& tmp_buf)
{
	unique_void_ptr::operator=(std::move(tmp_buf));
	_desc = tmp_buf._desc;
	_size = tmp_buf._size;
	cuda_error_check;
	return *this;
}

void gpu_manager_t::upload_buf(void* dst, const void* src, size_t size)
{
	if (size == 0 || dst == nullptr)  return; 
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	cuda_error_check;
}

void gpu_manager_t::download_buf(void* host_dst, const void* dev_src, size_t n)
{
	if (host_dst == nullptr || n == 0) return;
	cudaMemcpy(host_dst, dev_src, n, cudaMemcpyDeviceToHost);
	cuda_error_check;
}

void* gpu_manager_t::add_buf(const std::string& name, size_t size, const void* src, size_t size_copy)
{
	void* ptr_buf;
	auto buf = get_buf(name);

	if (size == 0) {
		//std::cout << "requested buf << " << name << " >> has size of 0 and will not be allocated, it may be allocated latter" << std::endl;
		return nullptr;
	}

	if (buf.has_value()) {
		ptr_buf = (*buf).first;
		size_t buf_size = (*buf).second;
		//std::cout << "buf << " << name << " >> has already been allocated\n" << std::endl;
		if (size != buf_size) {
			std::cout << "\033[31m" << "warning : adding duplicated gpu buf " << name << " !" << "\033[0m" << std::endl;
			//cudaFree(ptr_buf);
			delete_buf(name);
			ptr_buf = add_buf(name, size);
		}
		if (src != nullptr) {

			upload_buf(ptr_buf, src, size_copy);
		}
		return ptr_buf;
	}

	gpu_buf.emplace_back(name, size);
	ptr_buf = gpu_buf.rbegin()->get_buf();
	if (ptr_buf == nullptr) {
		printf("\033[31m-- unexcepted error at file %s, line %d\n\033[0m", __FILE__, __LINE__);
	}
	if (src != nullptr) {
		upload_buf(ptr_buf, src, size_copy);
	}
	return ptr_buf;

}

void* gpu_manager_t::add_buf(const std::string& name, size_t size, const void* src /*= nullptr*/)
{
	return add_buf(name, size, src, size);
}

std::optional<std::pair<void*, size_t >> gpu_manager_t::get_buf(const std::string& name)
{
	for (auto&& b : gpu_buf) {
		if (b._desc == name) {
			return std::pair(b.get(), b._size);
		}
		cuda_error_check;
	}
	cuda_error_check;
	return std::nullopt;
}

std::vector<gpu_manager_t::gpu_buf_t>::iterator gpu_manager_t::find_buf(const std::string& name)
{
	for (auto i = gpu_buf.begin(); i != gpu_buf.end(); i++) {
		if (i->_desc == name) {
			return i;
		}
	}
	cuda_error_check;
	return gpu_buf.end();
}

std::vector<gpu_manager_t::gpu_buf_t>::iterator gpu_manager_t::find_buf(const void* p)
{
	for (auto i = gpu_buf.begin(); i != gpu_buf.end(); i++) {
		if (i->get() == p) {
			return i;
		}
	}
	cuda_error_check;
	return gpu_buf.end();
}

void gpu_manager_t::delete_buf(const std::string& name)
{
	auto k = find_buf(name);
	if (k == gpu_buf.end()) {
		return;
	}
	else {
		gpu_buf.erase(k);
	}
}

void gpu_manager_t::delete_buf(void * pbuf)
{
	auto k = find_buf(pbuf);
	if (k == gpu_buf.end()) {
		printf("\033[31mfind no buf for this pointer, Line %d, File %s\n\033[0m", __LINE__, __FILE__);
		return;
	}
	else {
		gpu_buf.erase(k);
	}
}

void gpu_manager_t::initMem(void* pdata, size_t len, char value)
{
	cudaMemset(pdata, value, len);
	cuda_error_check;
}


