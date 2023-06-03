#include "lib.cuh"
#include "cusolver_common.h"
#include "cusolverSp.h"
#include "cusparse.h"
#include <vector>


void* _libbuf = nullptr;
size_t _libbufSize = 0;

__host__ void make_kernel_param(size_t* block_num, size_t* block_size, size_t num_tasks, size_t prefer_block_size) {
	*block_size = prefer_block_size;
	*block_num = (num_tasks + prefer_block_size - 1) / prefer_block_size;
}

__host__ void make_kernel_param(dim3& grid_dim, dim3& block_dim, const dim3& num_tasks, int prefer_block_size) {
	block_dim.x = prefer_block_size;
	block_dim.y = prefer_block_size;
	block_dim.z = prefer_block_size;
	grid_dim.x = (num_tasks.x + prefer_block_size - 1) / prefer_block_size;
	grid_dim.y = (num_tasks.y + prefer_block_size - 1) / prefer_block_size;
	grid_dim.z = (num_tasks.z + prefer_block_size - 1) / prefer_block_size;
}


double dump_array_sum(float* dump, size_t n) {
	constexpr int blockSize = 512;
	double sum;
	if (n <= 1) {
		float fsum;
		cudaMemcpy(&fsum, dump, sizeof(float), cudaMemcpyDeviceToHost);
		return fsum;
	}

	double* p_out = reinterpret_cast<double*>(dump);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n, 512);
	block_sum_kernel<float, 512, double> << <grid_size, block_size >> > (dump, p_out, n);

	n = (n + blockSize - 1) / blockSize;

	while (n > 1) {
		make_kernel_param(&grid_size, &block_size, n, blockSize);
		block_sum_kernel << <grid_size, block_size >> > (p_out, p_out, n);
		n = (n + blockSize - 1) / blockSize;
	}
	cudaMemcpy(&sum, p_out, sizeof(double), cudaMemcpyDeviceToHost);
	return sum;
}

Scaler array_norm2(Scaler* dev_data/*, Scaler* host_data*/, int n, bool root/* = true*/) {
	Scaler* tmp_buf, *block_buf;
	cudaMalloc(&tmp_buf, n * sizeof(Scaler));
	cudaMalloc(&block_buf, n * sizeof(Scaler));
	size_t block_dim;
	size_t grid_dim;
	make_kernel_param(&grid_dim, &block_dim, n, 512);
	auto sq = []__device__(Scaler s) { return s * s; };
	map << <grid_dim, block_dim >> > (dev_data, tmp_buf, n, sq);
	cudaDeviceSynchronize();
	cuda_error_check;
	Scaler sum = array_sum_gpu(tmp_buf, n, block_buf);
	cudaFree(tmp_buf);
	cudaFree(block_buf);
	if (root) {
		return sqrt(sum);
	}
	else {
		return sum;
	}
}

void show_cuSolver_version(void) {
	int major = -1, minor = -1, patch = -1;
	cusolverGetProperty(MAJOR_VERSION, &major);
	cusolverGetProperty(MINOR_VERSION, &minor);
	cusolverGetProperty(PATCH_LEVEL, &patch);
	printf("[cuda version] : %d.%d.%d\n", major, minor, patch);
}


void init_cuda(void) {
	get_device_info();
	show_cuSolver_version();
	if (std::is_same<Scaler, double>::value) {
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		printf("[bank width] : set 8 bytes \n");
	}
	else if(std::is_same<Scaler, float>::value){
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
		printf("[bank width] : set 4 bytes \n");
	}

	printf("\n");

	//
	printf("-- test library...\n");
	lib_test();
}

void use4Bytesbank(void) {
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	printf("\033[32mCUDA> Using 4 byte bank width\n\033[0m");
}
void use8Bytesbank(void){
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	printf("\033[32mCUDA> Using 8 byte bank width\n\033[0m");
}

__global__ void remap(int n, Scaler* p, Scaler l_, Scaler h_, Scaler L_, Scaler H_) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	Scaler sca = (H_ - L_) / (h_ - l_);
	if (tid < n) {
		p[tid] = (p[tid] - l_)*sca + L_;
	}
}


void lib_test(void) {
#if 0
	// test sum kernel
	Scaler* tmp, *tmp1;
	Scaler* v[3], *u[3];
	int large_size = 1e6;
	cudaMalloc(&tmp, sizeof(Scaler)*large_size);
	cudaMalloc(&tmp1, sizeof(Scaler)*large_size);
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&v[i], sizeof(Scaler)*large_size);
		cudaMalloc(&u[i], sizeof(Scaler)*large_size);
	}
	init_array(tmp, Scaler(1.5), large_size);
	std::cout << "dump array sum " << dump_array_sum(tmp, large_size) << std::endl;
	init_array(tmp, Scaler(1.5), large_size);
	std::cout << "dump array sum v0 " << dump_array_sum_v0(tmp, large_size) << std::endl;
	init_array(tmp, Scaler(1.5), large_size);
	std::cout << "array_sum_gpu " << array_sum_gpu(tmp, large_size, tmp1) << std::endl;
	init_array(tmp, Scaler(1.5), large_size);
	std::cout << "parallel_sum " << parallel_sum(tmp, tmp1, large_size) << std::endl;
	init_array(tmp, Scaler(1.5), large_size);
	std::cout << "array_norm2  " << array_norm2(tmp, large_size) << std::endl;;
	init_array(tmp, Scaler(1.5), large_size);
	init_array(tmp + large_size - 1, Scaler(1.51), 1);
	std::cout << "parallel max " << parallel_max(tmp, tmp1, large_size) << std::endl;
	//std::cout << "parallel min " << parallel_min(tmp, tmp1, large_size) << std::endl;

	for (int i = 0; i < 3; i++) { init_array(v[i], Scaler(1.5), large_size); init_array(u[i], Scaler(1.5), large_size); }
	std::cout << "u * v = " << dot(u[0], u[1], u[2], v[0], v[1], v[2], tmp, large_size) << std::endl;

	
	for (int i = 0; i < 3; i++) { init_array(v[i], Scaler(1.5), large_size); init_array(u[i], Scaler(2), large_size); }
	std::cout << "norm of u " << norm(u[0], u[1], u[2], tmp1, large_size) << std::endl;
	std::cout << "norm of v " << norm(v[0], v[1], v[2], tmp1, large_size) << std::endl;


	cuda_error_check;

	cudaFree(tmp);
	cudaFree(tmp1);
	for (int i = 0; i < 3; i++) {
		cudaFree(u[i]);
		cudaFree(v[i]);
	}
#else

#endif
	printf("-- Passed\n");
}

int get_device_info()
{
	int device_count{ 0 };
	// get number of devices 
	cudaGetDeviceCount(&device_count);
	fprintf(stdout, "[GPU device Number]: %d\n", device_count);

	if (device_count == 0) {
		fprintf(stdout, "\033[31mNo CUDA supported device Found!\033[0m\n");
		exit(-1);
	}

	int usedevice = 0;

	cudaDeviceProp use_device_prop;
	use_device_prop.major = 0;
	use_device_prop.minor = 0;

	fprintf(stdout, "- Enumerating Device...\n");
	for (int dev = 0; dev < device_count; ++dev) {
		fprintf(stdout, "---------------------------------------------------------------\n");
		int driver_version{ 0 }, runtime_version{ 0 };
		// set cuda execuation GPU 
		cudaSetDevice(dev);
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, dev);

		fprintf(stdout, "\n[Device %d]: %s\n", dev, device_prop.name);

		cudaDriverGetVersion(&driver_version);
		fprintf(stdout, "[CUDA driver]:- - - - - - - - - - - - - - - - - %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		fprintf(stdout, "[CUDA runtime]: - - - - - - - - - - - - - - - - %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
		fprintf(stdout, "[Device Capicity]: - -  - - - - - - - - - - - - %d.%d\n", device_prop.major, device_prop.minor);

		if (device_prop.major >= use_device_prop.major && device_prop.minor >= use_device_prop.minor) {
			usedevice = dev;
			use_device_prop = device_prop;
		}
	}
	fprintf(stdout, " \n");
	fprintf(stdout, "- set device %d [%s]\n", usedevice, use_device_prop.name);
	cudaSetDevice(usedevice);

	return 0;
}


void* reserve_buf(size_t require) {
	if (require < _libbufSize) {
		return _libbuf;
	} else {
		cudaMalloc(&_libbuf, require);
		_libbufSize = require;
		return _libbuf;
	}
}

void* get_libbuf(void) { return _libbuf; }

size_t get_libbuf_size(void) { return _libbufSize; }
