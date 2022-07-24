#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN // Please, not too much windows shenanigans
#endif

// Standard libs
#include <string>
#include <cstdio>
// GLM for maths
// #define GLM_FORCE_PURE GLM_FORCE_PURE (not needed anymore with recent GLM versions)
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
// Trimesh for model importing
#include "TriMesh.h"
// Util
#include "util.h"
#include "util_io.h"
#include "util_cuda.h"
#include "timer.h"
// CPU voxelizer fallback
#include "cpu_voxelizer.h"
#include "voxelizer.h"
#include "array"

using namespace std;
string version_number = "v0.4.12";

// Forward declaration of CUDA functions
float* meshToGPU_thrust(const trimesh::TriMesh *mesh); // METHOD 3 to transfer triangles can be found in thrust_operations.cu(h)
void cleanup_thrust();
void voxelize(const voxinfo & v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);
void voxelize_solid(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);

// Output formats
enum class OutputFormat { output_binvox = 0, output_morton = 1, output_obj_points = 2, output_obj_cubes = 3};
const char *OutputFormats[] = { "binvox file", "morton encoded blob", "obj file (pointcloud)", "obj file (cubes)" };

// Default options
string filename = "";
string filename_base = "";
OutputFormat outputformat = OutputFormat::output_binvox;
unsigned int gridsize = 256;
bool useThrustPath = false;
bool forceCPU = false;
bool solidVoxelization = false;

void printHeader(){
	fprintf(stdout, "## CUDA VOXELIZER \n");
	cout << "CUDA Voxelizer " << version_number << " by Jeroen Baert" << endl; 
	cout << "github.com/Forceflow/cuda_voxelizer - mail@jeroen-baert.be" << endl;
}

void printExample() {
	cout << "Example: cuda_voxelizer -f /home/jeroen/bunny.ply -s 512" << endl;
}

void printHelp(){
	fprintf(stdout, "\n## HELP  \n");
	cout << "Program options: " << endl << endl;
	cout << " -f <path to model file: .ply, .obj, .3ds> (required)" << endl;
	cout << " -s <voxelization grid size, power of 2: 8 -> 512, 1024, ... (default: 256)>" << endl;
	cout << " -o <output format: binvox, obj, obj_points or morton (default: binvox)>" << endl;
	cout << " -thrust : Force using CUDA Thrust Library (possible speedup / throughput improvement)" << endl;
	cout << " -cpu : Force CPU-based voxelization (slow, but works if no compatible GPU can be found)" << endl;
	cout << " -solid : Force solid voxelization (experimental, needs watertight model)" << endl << endl;
	printExample();
	cout << endl;
}

// METHOD 1: Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
float* meshToGPU_managed(const trimesh::TriMesh *mesh) {
	Timer t; t.start();
	size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
	float* device_triangles;
	fprintf(stdout, "[Mesh] Allocating %s of CUDA-managed UNIFIED memory for triangle data \n", (readableSize(n_floats)).c_str());
	checkCudaErrors(cudaMallocManaged((void**) &device_triangles, n_floats)); // managed memory
	fprintf(stdout, "[Mesh] Copy %llu triangles to CUDA-managed UNIFIED memory \n", (size_t)(mesh->faces.size()));
	for (size_t i = 0; i < mesh->faces.size(); i++) {
		glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
		glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
		glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
		size_t j = i * 9;
		memcpy((device_triangles)+j, glm::value_ptr(v0), sizeof(glm::vec3));
		memcpy((device_triangles)+j+3, glm::value_ptr(v1), sizeof(glm::vec3));
		memcpy((device_triangles)+j+6, glm::value_ptr(v2), sizeof(glm::vec3));
	}
	t.stop();fprintf(stdout, "[Perf] Mesh transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds);
	return device_triangles;
}

// METHOD 2: Helper function to transfer triangles to old-style, self-managed CUDA memory ( < CUDA 7.x )
// Leaving this here for reference, the function above should be faster and better managed on all versions CUDA 7+
// 
//float* meshToGPU(const trimesh::TriMesh *mesh){
//	size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
//	float* pagelocktriangles;
//	fprintf(stdout, "Allocating %llu kb of page-locked HOST memory \n", (size_t)(n_floats / 1024.0f));
//	checkCudaErrors(cudaHostAlloc((void**)&pagelocktriangles, n_floats, cudaHostAllocDefault)); // pinned memory to easily copy from
//	fprintf(stdout, "Copy %llu triangles to page-locked HOST memory \n", (size_t)(mesh->faces.size()));
//	for (size_t i = 0; i < mesh->faces.size(); i++){
//		glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
//		glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
//		glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
//		size_t j = i * 9;
//		memcpy((pagelocktriangles)+j, glm::value_ptr(v0), sizeof(glm::vec3));
//		memcpy((pagelocktriangles)+j+3, glm::value_ptr(v1), sizeof(glm::vec3));
//		memcpy((pagelocktriangles)+j+6, glm::value_ptr(v2), sizeof(glm::vec3));
//	}
//	float* device_triangles;
//	fprintf(stdout, "Allocating %llu kb of DEVICE memory \n", (size_t)(n_floats / 1024.0f));
//	checkCudaErrors(cudaMalloc((void **) &device_triangles, n_floats));
//	fprintf(stdout, "Copy %llu triangles from page-locked HOST memory to DEVICE memory \n", (size_t)(mesh->faces.size()));
//	checkCudaErrors(cudaMemcpy((void *) device_triangles, (void*) pagelocktriangles, n_floats, cudaMemcpyDefault));
//	return device_triangles;
//}

// Parse the program parameters and set them as global variables
void parseProgramParameters(int argc, char* argv[]){
	if(argc<2){ // not enough arguments
		fprintf(stdout, "Not enough program parameters. \n \n");
		printHelp();
		exit(0);
	} 
	bool filegiven = false;
	for (int i = 1; i < argc; i++) {
		if (string(argv[i]) == "-f") {
			filename = argv[i + 1];
			filename_base = filename.substr(0, filename.find_last_of("."));
			filegiven = true;
			if (!file_exists(filename)) {
				fprintf(stdout, "[Err] File does not exist / cannot access: %s \n", filename.c_str());
				exit(1);
			}
			i++;
		}
		else if (string(argv[i]) == "-s") {
			gridsize = atoi(argv[i + 1]);
			i++;
		} else if (string(argv[i]) == "-h") {
			printHelp();
			exit(0);
		} else if (string(argv[i]) == "-o") {
			string output = (argv[i + 1]);
			transform(output.begin(), output.end(), output.begin(), ::tolower); // to lowercase
			if (output == "binvox"){outputformat = OutputFormat::output_binvox;}
			else if (output == "morton"){outputformat = OutputFormat::output_morton;}
			else if (output == "obj"){outputformat = OutputFormat::output_obj_cubes;}
			else if (output == "obj_points") { outputformat = OutputFormat::output_obj_points; }
			else {
				fprintf(stdout, "[Err] Unrecognized output format: %s, valid options are binvox (default), morton, obj or obj_points \n", output.c_str());
				exit(1);
			}
		}
		else if (string(argv[i]) == "-thrust") {
			useThrustPath = true;
		}
		else if (string(argv[i]) == "-cpu") {
			forceCPU = true;
		}
		else if (string(argv[i])=="-solid"){
			solidVoxelization = true;
		}
	}
	if (!filegiven) {
		fprintf(stdout, "[Err] You didn't specify a file using -f (path). This is required. Exiting. \n");
		printExample();
		exit(1);
	}
	fprintf(stdout, "[Info] Filename: %s \n", filename.c_str());
	fprintf(stdout, "[Info] Grid size: %i \n", gridsize);
	fprintf(stdout, "[Info] Output format: %s \n", OutputFormats[int(outputformat)]);
	fprintf(stdout, "[Info] Using CUDA Thrust: %s (default: No)\n", useThrustPath ? "Yes" : "No");
	fprintf(stdout, "[Info] Using CPU-based voxelization: %s (default: No)\n", forceCPU ? "Yes" : "No");
	fprintf(stdout, "[Info] Using Solid Voxelization: %s (default: No)\n", solidVoxelization ? "Yes" : "No");
}

#if 0
int main(int argc, char* argv[]) {
	// PRINT PROGRAM INFO
	Timer t; t.start();
	printHeader();

	// PARSE PROGRAM PARAMETERS
	fprintf(stdout, "\n## PROGRAM PARAMETERS \n");
	parseProgramParameters(argc, argv);
	fflush(stdout);
	trimesh::TriMesh::set_verbose(false);

	// READ THE MESH
	fprintf(stdout, "\n## READ MESH \n");
#ifdef _DEBUG
	trimesh::TriMesh::set_verbose(true);
#endif
	fprintf(stdout, "[I/O] Reading mesh from %s \n", filename.c_str());
	trimesh::TriMesh* themesh = trimesh::TriMesh::read(filename.c_str());
	themesh->need_faces(); // Trimesh: Unpack (possible) triangle strips so we have faces for sure
	fprintf(stdout, "[Mesh] Number of triangles: %zu \n", themesh->faces.size());
	fprintf(stdout, "[Mesh] Number of vertices: %zu \n", themesh->vertices.size());
	fprintf(stdout, "[Mesh] Computing bbox \n");
	themesh->need_bbox(); // Trimesh: Compute the bounding box (in model coordinates)

	// COMPUTE BOUNDING BOX AND VOXELISATION PARAMETERS
	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	// Initialize our own AABox, pad it so it's a cube
	AABox<glm::vec3> bbox_mesh_cubed = createMeshBBCube<glm::vec3>(AABox<glm::vec3>(trimesh_to_glm(themesh->bbox.min), trimesh_to_glm(themesh->bbox.max)));
	// Create voxinfo struct and print all info
	voxinfo voxelization_info(bbox_mesh_cubed, glm::uvec3(gridsize, gridsize, gridsize), themesh->faces.size());
	voxelization_info.print();
	// Compute space needed to hold voxel table (1 voxel / bit)
	unsigned int* vtable; // Both voxelization paths (GPU and CPU) need this
	size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x) * static_cast<size_t>(voxelization_info.gridsize.y) * static_cast<size_t>(voxelization_info.gridsize.z) / 32.0f) * 4);

	// CUDA initialization
	bool cuda_ok = false;
	if (!forceCPU)
	{
		// SECTION: Try to figure out if we have a CUDA-enabled GPU
		fprintf(stdout, "\n## CUDA INIT \n");
		cuda_ok = initCuda();
		if (! cuda_ok ) fprintf(stdout, "[Info] CUDA GPU not found\n");
	}

	// SECTION: The actual voxelization
	if (cuda_ok && !forceCPU) { 
		// GPU voxelization
		fprintf(stdout, "\n## TRIANGLES TO GPU TRANSFER \n");

		float* device_triangles;
		// Transfer triangles to GPU using either thrust or managed cuda memory
		if (useThrustPath) { device_triangles = meshToGPU_thrust(themesh); }
		else { device_triangles = meshToGPU_managed(themesh); }

		if (!useThrustPath) {
			fprintf(stdout, "[Voxel Grid] Allocating %s of CUDA-managed UNIFIED memory for Voxel Grid\n", readableSize(vtable_size).c_str());
			checkCudaErrors(cudaMallocManaged((void**)&vtable, vtable_size));
		}
		else {
			// ALLOCATE MEMORY ON HOST
			fprintf(stdout, "[Voxel Grid] Allocating %s kB of page-locked HOST memory for Voxel Grid\n", readableSize(vtable_size).c_str());
			checkCudaErrors(cudaHostAlloc((void**)&vtable, vtable_size, cudaHostAllocDefault));
		}
		fprintf(stdout, "\n## GPU VOXELISATION \n");
		if (solidVoxelization){
			voxelize_solid(voxelization_info, device_triangles, vtable, useThrustPath, (outputformat == OutputFormat::output_morton));
		}
		else{
			voxelize(voxelization_info, device_triangles, vtable, useThrustPath, (outputformat == OutputFormat::output_morton));
		}
	} else { 
		// CPU VOXELIZATION FALLBACK
		fprintf(stdout, "\n## CPU VOXELISATION \n");
		if (!forceCPU) { fprintf(stdout, "[Info] No suitable CUDA GPU was found: Falling back to CPU voxelization\n"); }
		else { fprintf(stdout, "[Info] Doing CPU voxelization (forced using command-line switch -cpu)\n"); }
		// allocate zero-filled array
		vtable = (unsigned int*) calloc(1, vtable_size);
		if (!solidVoxelization) {
			cpu_voxelizer::cpu_voxelize_mesh(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
		}
		else {
			cpu_voxelizer::cpu_voxelize_mesh_solid(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
		}
	}

	//// DEBUG: print vtable
	//for (int i = 0; i < vtable_size; i++) {
	//	char* vtable_p = (char*)vtable;
	//	cout << (int) vtable_p[i] << endl;
	//}

	fprintf(stdout, "\n## FILE OUTPUT \n");
	if (outputformat == OutputFormat::output_morton){
		write_binary(vtable, vtable_size, filename);
	} else if (outputformat == OutputFormat::output_binvox){
		write_binvox(vtable, voxelization_info, filename);
	}
	else if (outputformat == OutputFormat::output_obj_points) {
		write_obj_pointcloud(vtable, voxelization_info, filename);
	}
	else if (outputformat == OutputFormat::output_obj_cubes) {
		write_obj_cubes(vtable, voxelization_info, filename);
	}

	if (useThrustPath) {
		cleanup_thrust();
	}

	fprintf(stdout, "\n## STATS \n");
	t.stop(); fprintf(stdout, "[Perf] Total runtime: %.1f ms \n", t.elapsed_time_milliseconds);
}
#endif


std::pair<glm::vec3, glm::vec3> get_point_bb(const std::vector<float>& vcoords) {
	glm::vec3 xmin, xmax;
	for (int i = 0; i < 3; i++) xmin[i] = std::numeric_limits<float>::infinity();
	xmax = -xmin;

	printf("initial xmax = (%f, %f, %f)\n", xmax[0], xmax[1], xmax[2]);

	int npoints = vcoords.size();
	if (npoints % 3 != 0) {
		throw std::string("number of points coordinate array should be multiple of 3");
	}
	for (int i = 0; i < npoints; i += 3) {
		for (int j = 0; j < 3; j++) {
			xmin[j] = (std::min)(xmin[j], vcoords[i + j]);
			xmax[j] = (std::max)(xmax[j], vcoords[i + j]);
		}
	}
	return { xmin,xmax };
}

int padd2SatifyResolution(int prefer_reso, std::pair<glm::vec3, glm::vec3> tight_bb, std::pair<glm::vec3, glm::vec3>& out_bb) {
	unsigned int padd_res = 1;
	for (int i = 0; i < 32; i++) {
		if (prefer_reso < padd_res) {
			break;
		}
		padd_res <<= 1;
	}

	glm::vec3 len = tight_bb.second - tight_bb.first;

	float max_len = (std::max)((std::max)(len[0], len[1]), len[2]);

	glm::vec3 padd_len;
	for (int j = 0; j < 3; j++) padd_len[j] = max_len * padd_res / prefer_reso;

	printf("max axis len : %f\n", max_len);
	printf("padd_len     : (%f, %f, %f)\n", padd_len[0], padd_len[1], padd_len[2]);

	std::pair<glm::vec3, glm::vec3> new_bb;
	for (int j = 0; j < 3; j++) {
		new_bb.first[j] = tight_bb.first[j] - (padd_len[j] - len[j]) / 2;
		new_bb.second[j] = tight_bb.second[j] + (padd_len[j] - len[j]) / 2;
	}

	out_bb = new_bb;

	return padd_res;
}

float* uploadTriangles(const std::vector<float>& vertex_coords, const std::vector<int>& triface_ids);

voxinfo voxelize_mesh(
	const std::vector<float>& vertex_coords, const std::vector<int>& triface_ids,
	int prefered_resolution, std::vector<unsigned int>& solid_bits, int out_resolution[3], float out_box[2][3]
) {

	auto vbb = get_point_bb(vertex_coords);

	printf("-- vbb = (%f, %f, %f)->(%f, %f, %f)\n", vbb.first[0], vbb.first[1], vbb.first[2], vbb.second[0], vbb.second[1], vbb.second[2]);

	std::pair<glm::vec3, glm::vec3> out_bb;
	int reso = padd2SatifyResolution(prefered_resolution, vbb, out_bb);

	printf("-- resolution old : %d, new : %d\n", prefered_resolution, reso);
	printf("-- padd_bb        : (%f, %f, %f)->(%f, %f, %f)\n",
		out_bb.first[0], out_bb.first[1], out_bb.first[2],
		out_bb.second[0], out_bb.second[1], out_bb.second[2]
	);

	AABox<glm::vec3> bcube(out_bb.first, out_bb.second);
	voxinfo voxelization_info(bcube, glm::uvec3(reso, reso, reso), triface_ids.size() / 3);
	voxelization_info.print();

	for (int i = 0; i < 3; i++) {
		out_resolution[i] = reso;
		out_box[0][i] = bcube.min[i];
		out_box[1][i] = bcube.max[i];
	}

	size_t vtable_size = static_cast<size_t>(
		ceil(static_cast<size_t>(voxelization_info.gridsize.x) *
			static_cast<size_t>(voxelization_info.gridsize.y) *
			static_cast<size_t>(voxelization_info.gridsize.z) / 32.0f) * 4);

	std::vector<unsigned int>& vtable = solid_bits;
	vtable.clear();
	vtable.resize(vtable_size / 4, -1);

	float* faces = uploadTriangles(vertex_coords, triface_ids);

	bool cuda_ok = initCuda();

	if (cuda_ok) {
		std::cout << "[Vox] Found cuda, using GPU routine" << std::endl;
		voxelize_solid(voxelization_info, faces, vtable.data(), true, false);
	}
	else {
		std::cout << "[Vox] Miss suitable CUDA package, using CPU routine" << std::endl;
		trimesh::TriMesh mesh;
		for (int i = 0; i < vertex_coords.size(); i += 3) {
			mesh.vertices.emplace_back(vertex_coords[i + 0], vertex_coords[i + 1], vertex_coords[i + 2]);
		}
		for (int i = 0; i < triface_ids.size(); i += 3) {
			mesh.faces.emplace_back(triface_ids[i], triface_ids[i + 1], triface_ids[i + 2]);
		}
		cpu_voxelizer::cpu_voxelize_mesh_solid(voxelization_info, &mesh, vtable.data(), false);
	}

	cleanup_thrust();

	return voxelization_info;
}

void hierarchy_voxelize_mesh(
	const std::vector<float>& vertex_coords, const std::vector<int>& triface_ids,
	int nlayer, int prefered_resolution, std::vector<vector<unsigned int>>& solid_bits, std::vector<std::array<int, 3>> out_resolutions, std::vector<std::pair<std::array<float, 3>, std::array<float, 3> >> out_boxs
) {
	int fine_reso[3];
	float fine_box[2][3];
	std::vector<unsigned int> fine_bits;
	voxelize_mesh(vertex_coords, triface_ids, prefered_resolution, fine_bits, fine_reso, fine_box);
	solid_bits.emplace_back(fine_bits);
	out_resolutions.push_back({ fine_reso[0], fine_reso[1], fine_reso[2] });
	out_boxs.emplace_back(
		std::array<float, 3>{ fine_box[0][0], fine_box[0][1], fine_box[0][2] }, 
		std::array<float, 3>{ fine_box[1][0], fine_box[1][1], fine_box[1][2] }
	);

	bool cuda_ok = initCuda();
	float* faces;
	if (cuda_ok) {
		faces = uploadTriangles(vertex_coords, triface_ids);
	}

	AABox<glm::vec3> bcube(glm::vec3{ fine_box[0][0],fine_box[0][1],fine_box[0][2] }, glm::vec3{ fine_box[1][0],fine_box[1][1],fine_box[1][2] });
	glm::uvec3 grid_size{ fine_reso[0],fine_reso[1],fine_reso[2] };
	int n = 0;
	while (n++ < nlayer) {
		grid_size /= 2;
		voxinfo voxelize_info(bcube, grid_size, triface_ids.size() / 3);
		std::vector<unsigned int> coarse_bits;
		size_t vtable_size = static_cast<size_t>(
			ceil(static_cast<size_t>(voxelize_info.gridsize.x) * static_cast<size_t>(voxelize_info.gridsize.y) * static_cast<size_t>(voxelize_info.gridsize.z) / 32.0f) * 4
			);
		coarse_bits.resize(vtable_size);

		if (cuda_ok) {
			voxelize_solid(voxelize_info, faces, coarse_bits.data(), true, false);
		}
		else {
			trimesh::TriMesh mesh;
			for (int i = 0; i < vertex_coords.size(); i += 3) {
				mesh.vertices.emplace_back(vertex_coords[i + 0], vertex_coords[i + 1], vertex_coords[i + 2]);
			}
			for (int i = 0; i < triface_ids.size(); i += 3) {
				mesh.faces.emplace_back(triface_ids[i], triface_ids[i + 1], triface_ids[i + 2]);
			}
			cpu_voxelizer::cpu_voxelize_mesh_solid(voxelize_info, &mesh, coarse_bits.data(), false);
		}

		solid_bits.emplace_back(coarse_bits);
		out_resolutions.push_back({ (int)grid_size[0], (int)grid_size[1], (int)grid_size[2] });
		out_boxs.emplace_back(
			std::array<float, 3>{bcube.min[0], bcube.min[1], bcube.min[2]},
			std::array<float, 3>{ bcube.max[0], bcube.max[1], bcube.max[2] }
		);
	}

	cleanup_thrust();
}
