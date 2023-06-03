#include "gpu_manager_t.h"
#include "matlab_utils.h"
std::string gpu_manager_t::make_anonymous_name(void)
{
	char buf[512];
#ifdef _WIN32
	sprintf_s(buf, "this is an anonymous buf of order %d", anonymous_counter++);
#else
	snprintf(buf, sizeof(buf), "this is an anonymous buf of order %d", anonymous_counter++);
#endif
	return std::string(buf);
}

void* gpu_manager_t::add_buf(size_t size, const void* src /*= nullptr*/)
{
	auto name_ = make_anonymous_name();
	return add_buf(name_, size, src);
}



size_t gpu_manager_t::size(void)
{
	size_t total_size = 0;
	for (int i = 0; i < gpu_buf.size(); i++) {
		total_size += gpu_buf[i]._size;
	}
	return total_size;
}

void gpu_manager_t::pass_dev_buf_to_matlab(const char*name, float* dev_ptr, size_t n)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<float, -1, 1> mat_buf;
	mat_buf.resize(n, 1);
	download_buf(mat_buf.data(), dev_ptr, n * sizeof(float));
	eigen2ConnectedMatlab(name, mat_buf);
#endif
}

void gpu_manager_t::pass_dev_buf_to_matlab(const char* name, Scaler* dev_ptr, int ldd, size_t n)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<Scaler, -1, -1> mat_buf;
	if (n%ldd != 0) {
		printf("\033[31mWarning : leading dimension or array size may be wrong !");
	}
	mat_buf.resize(ldd, n / ldd);
	download_buf(mat_buf.data(), dev_ptr, n * sizeof(Scaler));
	eigen2ConnectedMatlab(name, mat_buf);
#endif
}

void gpu_manager_t::pass_dev_buf_to_matlab(const char* name, double* dev_ptr, size_t n)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<double, -1, 1> mat_buf;
	mat_buf.resize(n, 1);
	download_buf(mat_buf.data(), dev_ptr, n * sizeof(double));
	eigen2ConnectedMatlab(name, mat_buf);
#endif
}

void gpu_manager_t::pass_dev_buf_to_matlab(const char* name, const int* dev_ptr, size_t n)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<double, -1, 1> mat_buf;
	mat_buf.resize(n, 1);
	std::vector<int> vechost(n);
	download_buf(vechost.data(), dev_ptr, n * sizeof(int));
	for (int i = 0; i < n; i++) mat_buf[i] = vechost[i];
	eigen2ConnectedMatlab(name, mat_buf);
#endif
}

void gpu_manager_t::pass_buf_to_matlab(const char* name, Scaler* host_ptr, size_t n)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<Scaler, -1, 1> mat_buf;
	mat_buf.resize(n, 1);
	std::copy(host_ptr, host_ptr + n, mat_buf.begin());
	eigen2ConnectedMatlab(name, mat_buf);
#endif
}

void gpu_manager_t::pass_buf_to_matlab(const char* name, int* host_ptr, size_t n)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<int, -1, 1> mat_buf;
	mat_buf.resize(n, 1);
	std::copy(host_ptr, host_ptr + n, mat_buf.begin());
	eigen2ConnectedMatlab(name, mat_buf);
#endif
}
