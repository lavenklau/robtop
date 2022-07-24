#pragma once

#include<string>
#include "memory"
#include "functional"
#include "optional"
//#include "mycommon.h"


class gpu_manager_t {
	typedef double Scaler;
private:

	typedef std::unique_ptr<void, std::function<void(void*)>> unique_void_ptr;
	/* GPU buf type */
	class gpu_buf_t : public std::unique_ptr<void, std::function<void(void*)>> {
		std::string _desc;
		//void* _buf;
		size_t _size;
		friend class gpu_manager_t;
	public:
		gpu_buf_t(const std::string& name, size_t size);

		gpu_buf_t(gpu_buf_t&& tmp_buf)noexcept;

		gpu_buf_t& operator=(gpu_buf_t&& tmp_buf);

		//~gpu_buf_t() {};
		void* get_buf(void) {
			return this->get();
		}
	};

	int anonymous_counter = 0;
	std::string make_anonymous_name(void);

	/* GPU buf array */
	std::vector<gpu_buf_t> gpu_buf;

public:
	/* upload data from host to GPU buf allocated */
	static void upload_buf(void* dst, const void* src, size_t size);

	static void download_buf(void* host_dst, const void* dev_src, size_t n);

	static void initMem(void* pdata, size_t len, char value = 0);

	/* add a GPU buf with specified name and size */
	void* add_buf(const std::string& name, size_t size, const void* src, size_t size_copy);

	void* add_buf(const std::string& name, size_t size, const void* src = nullptr);

	void* add_buf(size_t size, const void* src = nullptr);

	/* find specific GPU buf from those allocated */
	std::optional<std::pair<void*, size_t>> get_buf(const std::string& name);

	/* locate GPU buf in the buf array */
	std::vector<gpu_buf_t>::iterator find_buf(const std::string& name);
	std::vector<gpu_buf_t>::iterator find_buf(const void* p);

	/* delete specific GPU buf */
	void delete_buf(const std::string& name);
	void delete_buf(void * pbuf);

	size_t size(void);

	static void pass_dev_buf_to_matlab(const char*name, float* dev_ptr, size_t n);

	static void pass_dev_buf_to_matlab(const char* name, const int* dev_ptr, size_t n);

	static void pass_dev_buf_to_matlab(const char* name, double* dev_ptr, size_t n);

	static void pass_dev_buf_to_matlab(const char* name, Scaler* dev_ptr, int ldd, size_t n);

	static void pass_buf_to_matlab(const char* name, Scaler* host_ptr, size_t n);

	static void pass_buf_to_matlab(const char* name, int* host_ptr, size_t n);

	class symbol_uploader_t {
	public:
		//void upload_template_stiffness_matrix(Scaler* src);
	}symbol_uploader;
};

class gpu_manager_member_wrapper_t {
protected:
	gpu_manager_t* gpu_manager;
public:
	gpu_manager_member_wrapper_t(gpu_manager_t& gm) : gpu_manager(&gm) {}
	gpu_manager_member_wrapper_t() {}
};

