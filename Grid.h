#pragma once

#ifndef GRID_H
#define GRID_H

#include "voxelizer.h"

#include "string"
#include "map"
#include "type_traits"
#include "gpu_manager_t.h"
#include "snippet.h"
#include "set"
#include <memory>

namespace grid {

	class HierarchyGrid;

	template<typename T>
	struct BitCount {
		static constexpr int value = sizeof(T) * 8;
	};

	template<typename T, int N>
	struct LowerOnes {
		static constexpr T value = (T{ 1 } << (N )) - 1;
	};

	template<typename T>
	inline bool read_bit(T* _ptr, size_t id) {
		return _ptr[id / (sizeof(T) * 8)] & (T{ 1 } << (id % (sizeof(T) * 8)));
	}

	template<typename T, typename std::enable_if<!std::is_pointer<T>::value, int>::type = 0>
	inline bool read_bit(T word, int id) {
		return word & (T{ 1 } << id);
	}

	template<typename T>
	inline void set_bit(T* _ptr, size_t id) {
		_ptr[id / (sizeof(T) * 8)] |= (T{ 1 } << (id % (sizeof(T) * 8)));
	}

	template<typename T, typename std::enable_if<!std::is_pointer<T>::value, int>::type = 0>
	inline void set_bit(T& word, int id) {
		word |= T{ 1 } << id;
	}

	template<typename T>
	inline void clear_bit(T* _ptr, size_t id) {
		_ptr[id / (sizeof(T) * 8)] &= ~(T{ 1 } << id);
	}

	template<typename T>
	inline void clear_bit(T& word, int id) {
		word &= ~(T{ 1 } << id);
	}

	template<typename T>
	inline int countOne(T num) {
		int n = 0;
		while (num) {
			num &= (num - 1);
			n++;
		}
		return n;
	}

	template<int N, bool stop = (N == 0)>
	struct firstOne {
		static constexpr int value = 1 + firstOne< (N >> 1), ((N >> 1) == 0)>::value;
	};

	template<int N>
	struct firstOne<N, true> {
		static constexpr int value = -1;
	};

	struct DispatchCubeVertex {
		enum vertex_type{
			corner_vertex,
			face_center,
			edge_center,
			volume_center
		};
		static vertex_type vtype[27];
		DispatchCubeVertex(void);
		vertex_type operator()(int id);
		static vertex_type dispatch(int id);
	};

	template<typename T>
	class BitSAT {
	private:
		void buildChunkSat(void) {
			_chunkSat.resize(_bitArray.size() + 1, 0);
			int accu = 0;
			for (int i = 0; i < _bitArray.size(); i++) {
				_chunkSat[i] = accu;
				accu += countOne(_bitArray[i]);
			}
			*_chunkSat.rbegin() = accu;
		}
	public:
		static constexpr size_t size_mask = sizeof(T) * 8 - 1;
		std::vector<T> _bitArray;
		std::vector<int> _chunkSat;
		BitSAT(const std::vector<T>& bitArray) : _bitArray(bitArray) { buildChunkSat(); }

		BitSAT(std::vector<T>&& bitArray) noexcept : _bitArray(bitArray) { buildChunkSat(); }
		// the sat sum at id-th element in bit array
		int operator[](size_t id) {
			int ent = id >> firstOne<sizeof(T) * 8>::value;
			int mod = id & size_mask;
			return _chunkSat[ent] + countOne(_bitArray[ent] & ((T{ 1 } << mod) - 1));
		}
		
		size_t total(void) {
			return *_chunkSat.rbegin();
		}

		// the bit id of k-th 1
		int operator()(size_t id) const {
			int ent = id >> firstOne<sizeof(T) * 8>::value;
			int mod = id & size_mask;
			T resword = _bitArray[ent];
			if ((resword & (T{ 1 } << mod)) == 0) {
				return -1;
			}
			else {
				return _chunkSat[ent] + countOne(resword & ((T{ 1 } << mod) - 1));
			}
		}
	};
	void wordReverse_g(size_t nword, unsigned int* wordlist);

	void cubeGridSetSolidVertices(int reso, const std::vector<unsigned int>& solid_ebit, std::vector<unsigned int>& solid_vbit);

	void cubeGridSetSolidVertices_g(int reso, const std::vector<unsigned int>& solid_ebit, std::vector<unsigned int>& solid_vbit);

	void setSolidElementFromFineGrid_g(int finereso, const std::vector<unsigned int>& ebits_fine, std::vector<unsigned int>& ebits_coarse);

	//enum HierarchyGrid::Mode;

	enum Mode {
		no_support_constrain_force_direction,
		no_support_free_force,
		with_support_constrain_force_direction,
		with_support_free_force
	};

	template<typename dt = double, int N = 3>
	struct hostbufbackup_t {
		std::vector<dt> _hostbuf[N];
		double* _gbuf[N];
		hostbufbackup_t(dt** devBuf, size_t len) {
			for (int i = 0; i < N; i++) {
				_gbuf[i] = devBuf[i];
				_hostbuf[i].resize(len);
				gpu_manager_t::download_buf(_hostbuf[i].data(), devBuf[i], len * sizeof(dt));
			}
		}
		~hostbufbackup_t() {
			for (int i = 0; i < N; i++) {
				gpu_manager_t::upload_buf(_gbuf[i], _hostbuf[i].data(), _hostbuf[i].size() * sizeof(dt));
			}
		}
	};


	class Grid
	{
	public:
		static std::string _outdir;
		static void* _tmp_buf;
		static size_t _tmp_buf_size;
		static Mode _mode;
		static void setOutDir(const std::string& outdir);
		static const std::string& getOutDir(void);
		static void* getTempBuf(size_t requre);
		template<typename T>
		static void getTempBufArray(T** pbufs, int n_buf, size_t requre4each) {
			size_t alignedsize = snippet::Round<512 / sizeof(T)>(requre4each);
			T* ptotal = (T*)getTempBuf(alignedsize * n_buf * sizeof(T));
			for (int i = 0; i < n_buf; i++) {
				pbufs[i] = ptotal + i * alignedsize;
			}
		}
		static void clearBuf(void);

	public:
		friend class HierarchyGrid;
		std::string _name;
		struct {
			float* rho_e;
			int * v2e[8];
			int* v2vfine[27];
			int* v2vcoarse[8];
			int* v2vfinecenter[64];
			//int* v2vcoarsecoarse[8];
			int* v2v[27];
			//double* rxStenil[27][9];
			// stencil[27][9][vertex]
			double* rxStencil;

			unsigned int* eActiveBits;
			int* eActiveChunkSum;
			int nword_ebits;

			float* g_sens;

			/*
			  |_*_|_*_|_*_| * | * | * | * | * |
			  |___________|___________|
			  |   mod 8   |   mod 5   |
			*/
			int* vBitflag;
			int* eBitflag;
			int* vidmap;
			int* eidmap;
			double* Uworst[3];
			double* Fworst[3];
			double* U[3];
			double* F[3];
			double* R[3];
			double* Fsupport[3];
		} _gbuf;

		int n_vertices = 0;
		int n_elements = 0;
		int n_gsvertices = 0;
		int n_gselements = 0;

		std::map<std::string, double> _keyvalues;

		std::vector<int> _v2v[27];

		int gs_num[8];

		int _ereso = 0;

		Grid* fineGrid = nullptr;
		Grid* coarseGrid = nullptr;

		enum Bitmask {
			mask_xmod7 = 0b111,
			offset_xmod7 = 0,

			mask_ymod7 = 0b111000,
			offset_ymod7 = 3,

			mask_zmod7 = 0b111000000,
			offset_zmod7 = 6,

			mask_gscolor = 0b111000000000,
			offset_gscolor = 9,

			mask_shellelement = 0b1000000000000,
			offset_shellelement = 12,

			mask_surfacenodes = 0b10000000000000,
			offset_surfacenodes = 13,

			mask_loadnodes = 0b100000000000000,
			offset_loadnodes = 14,

			mask_layerid = 0b111000000000000000,
			offset_layerid = 15,

			mask_invalid = 0b1000000000000000000,
			offset_invalid = 18,

			mask_supportnodes = 0b10000000000000000000,
			offset_supportnodes = 19,

			mask_surfaceelements = 0b100000000000000000000,
			offset_surfaceelements = 20
		};

		int _layer = 0;
		bool _finest = false;
		bool _coarsest = false;
		bool _dummy = false;

		bool _skiplayer = false;

		float _box[2][3];
		std::function<bool(double[3])> _inLoadArea;
		std::function<bool(double[3])> _inFixedArea;
		std::function<Eigen::Matrix<double, 3, 1>(double[3])> _loadField;

		std::vector<int> _gsLoadNodes;

		std::stack<std::string> _logStack;

		std::set<std::string> _logHistory;

		snippet::Loger subLog(const std::string& sublog);

		void set_skip(void) { _skiplayer = true; }
		bool is_skip(void) { return _skiplayer; }
		void set_dummy(void) { _dummy = true; }
		bool is_dummy(void) { return _dummy; }

		void clearMsglog(void);

		std::ostream& msg(void);

		std::ofstream msglog(void);

		double elementLength(void);

		void lexico2gsorder(int* idmap, int n_id, int* ids, int n_mapid, int* mapped_ids, int* valuemap = nullptr);

		void lexico2gsorder_g(int* idmap, int n_id, int* ids, int n_mapid, int* mapped_ids, int* valuemap = nullptr);

		void mark_surface_nodes_g(void);

		void mark_surface_nodes_g(int nv, int* v2e[8], int* vflag);

		void mark_surface_elements_g(int nv, int ne, int* v2e[8], int* vflag, int* eflag);

		static void setVerticesPosFlag(int vreso, BitSAT<unsigned int>& vrtsat, int* flags);

		static void setV2E(int vreso, BitSAT<unsigned int>& vrtsat, BitSAT<unsigned int>& elsat, int* v2e[8]);

		static void setV2E_g(int vreso, BitSAT<unsigned int>& vrtsat, BitSAT<unsigned int>& elsat, int* v2e[8]);

		static void setV2V(int vreso, BitSAT<unsigned int>& vrtsat, int* v2v[27]);

		static void setV2V_g(int vreso, BitSAT<unsigned int>& vrtsat, int* v2v[27]);

		static void setV2VCoarse_g(int skip, int vresofine,
			grid::BitSAT<unsigned int>& vsatfine, grid::BitSAT<unsigned int>& vsatcoarse,
			int* v2vcoarse[8]
		);

		static void setV2VFine_g(int skip, int vresocoarse,
			grid::BitSAT<unsigned int>& vsatfine, grid::BitSAT<unsigned int>& vsatcoarse,
			int* v2vfine[27]
		);

		static void setV2VFineC_g(int vresocoarse,
			grid::BitSAT<unsigned int>& vsatfine2, grid::BitSAT<unsigned int>& vsatcoarse,
			int* v2vfinec[64]
		);

		void computeProjectionMatrix(int nv, int nv_gs, int vreso, const std::vector<int>& lexi2gs, const int* lexi2gs_dev, BitSAT<unsigned int>& vsat, int* vflaghost, int* vflagdev);

		int* getEidmap(void) { return _gbuf.eidmap; }

		int* getVidmap(void) { return _gbuf.vidmap; }

		void getV2V(void);

		void applyK(double* u[3], double* f[3]);

		void applyAjointK(double* usrc[3], double* fdst[3]);

		void filterSensitivity(double radii);

		Eigen::Matrix<double, 3, 1> outwardNormal(double p[3]);

		std::vector<int> getVflags(void);

		void getVflags(int nv, int* dst);

		void setVflags(int nv, int *src);

		std::vector<int> getEflags(void);

		void getEflags(int nv, int* dst);

		double** getForce(void) { return _gbuf.F; }

		double** getResidual(void) { return _gbuf.R; }

		float* getRho(void) { return _gbuf.rho_e; }

		float* getSens(void) { return _gbuf.g_sens; }

		double** getWorstForce(void) { return _gbuf.Fworst; }

		double** getWorstDisplacement(void) { return _gbuf.Uworst; }

		double** getSupportForce(void) { return _gbuf.Fsupport; }

		double** getDisplacement(void) { return _gbuf.U; }

		double supportForceCh(void);

		double supportForceCh(double* newf[3]);

		double supportForceNorm(void);

		size_t build(
			gpu_manager_t& gm,
			BitSAT<unsigned int>& vbit,
			BitSAT<unsigned int>& ebit,
			Grid* finer,
			//Grid* coarser,
			int vreso,
			int layer,
			int nv, int ne,
			int * v2ehost[8],
			int * v2vfinehost[27], 
			int * v2vcoarsehost[8],
			int * v2vhost[27],
			int * v2vfinec[64],
			int * vbitflags, 
			int * ebitflags
		);

		void gs_relax(int n_times = 1);

		//void gs_adjoint_relax(int n_times = 1);

		void reset_displacement(void);

		void reset_force(void);

		void reset_residual(void);

		void resetDirchlet(double* v_dev[3]);

		double compliance(void);

		void update_residual(void);

		//void update_adjoint_residual(void);

		void prolongate_correction(void);

		void restrict_residual(void);

		double relative_residual(void);

		double residual(void);

		void init_rho(double rh0);

		float volumeRatio(void);

		void use_grid(void);

		void solve_fem_host(void);

		void buildCoarsestSystem(void);

		void compute_gscolor(gpu_manager_t& gm, BitSAT<unsigned int>& vbitsat, BitSAT<unsigned int>& ebitsat, int vreso, int* vbitflaghost, int* ebitflaghost);

		void enumerate_gs_subset(int nv, int ne, int* vflags, int* eflags, int& nv_gs, int& ne_gs, std::vector<int>& vlexi2gs, std::vector<int>& elexi2gs);

		void randForce(void);

		void readForce(std::string forcefile);

		void readSupportForce(std::string fsfile);

		void readDisplacement(std::string displacementfile);

		double unitizeForce(void);

		void pertubDisplacement(double ratio);

		void pertubForce(double ratio);

		void elementCompliance(double* u[3], double* f[3], float* dst);

		double densityDiscretiness(void);

		double compliance(double* u[3], double* f[3]);

		void force2matlab(const std::string& nam);

		void displacement2matlab(const std::string& nam);

		void residual2matlab(const std::string& nam);

		void stencil2matlab(const std::string& nam);

		void vidmap2matlab(const std::string & nam);

		void eidmap2matlab(const std::string & nam);

		bool checkV2V(void);

		bool checkV2Vhost(int nv, int ne, int* v2e[8], int* v2v[27]);

		void elexibuf2matlab(const std::string& nam, float* p_gsbuf);

		void vlexibuf2matlab(const std::string& nam, double* p_gsbuf);

		float* getlexiEbuf(float* gs_src);

		double* getlexiVbuf(double* gs_src);

		bool isForceFree(void) { return _mode == no_support_free_force || _mode == with_support_free_force; }

		bool hasSupport(void) { return _mode == with_support_constrain_force_direction || _mode == with_support_free_force; }

		void sens2matlab(const std::string& nam);

		void v2v2matlab(const std::string& nam);

		void v2vcoarse2matlab(const std::string& nam);
	public:
		int n_nodes(void) { return n_gsvertices; }
		int n_valid_elements(void) { return n_elements; }
		int n_rho(void) { return n_gselements; }
		double v3norm(double* v[3]);

		void v3_create(double* dstv[3]);
		void v3_destroy(double* dstv[3]);
		hostbufbackup_t<double, 3> v3_backup(double* vdata[3]);
		void v3_init(double* v[3], double val[3]);
		void v3_rand(double* v[3], double low, double upp);
		void v3_pertub(double* v[3], double ratio);
		void v3_copy(double* vsrc[3], double* vdst[3]);
		void v3_add(double alpha, double* a[3], double beta, double* b[3]);
		bool v3_hasNaN(double* v[3]);
		void v3_add(double* a[3], double alpha, double* b[3]);
		void v3_minus(double* a[3], double alpha, double* b[3]);
		void v3_minus(double* dst[3], double* a[3], double alpha, double* b[3]);
		void v3_scale(double* v[3], double ampl);
		double v3_norm(double* v[3]);
		double v3_normalize(double* v[3]);
		double v3_dot(double* v[3], double* u[3]);
		double v3_diffdot(double* v1[3], double* v2[3], double* v3[3], double* v4[3]);
		void v3_toMatlab(const std::string& nam, double* v[3]);
	};

	class HierarchyGrid {
	private:

		std::vector<Grid*> _gridlayer;

		struct HierarchySetting {
			bool skiplayer1 = false;
			int prefer_reso = 128;
			int coarse_reso = 32;
			double shell_width = 0;
			gpu_manager_t* gmem;
		}_setting;

		int _nlayer = 0;

	public:
		std::vector<BitSAT<unsigned int>> elesatlist;
		std::vector<BitSAT<unsigned int>> vrtsatlist;

		std::function<bool(double[3])> _inLoadArea;
		std::function<bool(double[3])> _inFixedArea;
		std::function<Eigen::Matrix<double, 3, 1>(double[3])> _loadField;

		std::string _outdir;

		Mode _mode;

		//std::vector<float> _pcoords;
		//std::vector<int> _trifaces;

		enum LogMask {
			mask_log_density = 0b1,
			mask_log_compliance = 0b10
		};

		int _logFlag = 0;

	public:

		gpu_manager_t& get_gmem(void) { return *_setting.gmem; }

		HierarchyGrid(gpu_manager_t& gm) { _setting.gmem = &gm; }

		Grid* operator[](int i) { return _gridlayer[i]; }

		void setOutPath(const std::string& outpath) { _outdir = outpath; Grid::_outdir = outpath; }

		std::string getPath(const std::string& nfile) { return _outdir + nfile; }

		bool isForceFree(void) { return _mode == no_support_free_force || _mode == with_support_free_force; }

		bool hasSupport(void) { return _mode == with_support_constrain_force_direction || _mode == with_support_free_force; }

		void buildAABBTree(const std::vector<float>& pcoords, const std::vector<int>& trifaces);

		void setSolidShellElement(const std::vector<unsigned int>& ebitfine, BitSAT<unsigned int>& esat, float box[2][3], int ereso, std::vector<int>& eflags);

		void testShell(void);

		void fillShell(void);

		// log file
		void log(int itn);

		void enable_logs(int flag);

		void enable_logdensity(bool en) { if (en)_logFlag |= LogMask::mask_log_density; else _logFlag &= ~(int)mask_log_density; };

		void enable_logcompliance(bool en) { if (en)_logFlag |= LogMask::mask_log_compliance; else _logFlag &= ~(int)(mask_log_compliance); };

		void setMode(Mode mode);

		static std::string getModeStr(Mode mode);

		//void uploadTemplateMatrix(double element_len);

		void set_prefer_reso(int preferreso) { _setting.prefer_reso = preferreso; }

		void set_coarse_reso(int coarsereso) { _setting.coarse_reso = coarsereso; }

		void set_shell_width(double wshell) { _setting.shell_width = wshell; }

		void set_skip_layer(bool isskip) { _setting.skiplayer1 = isskip; }

		void genFromMesh(const std::vector<float>& pcoords, const std::vector<int>& facevertices);

		void resetAllResidual(void);

		void restrict_stencil_dyadic(Grid& dstcoarse, Grid& srcfine);

		void restrict_stencil_nondyadic(Grid& dstcoarse, Grid& srcfine);

		//void restrict_adjoint_stencil_nondyadic(Grid& dstcoarse, Grid& srcfine);

		//void restrict_adjoint_stencil_dyadic(Grid& dstcoarse, Grid& srcfine);

		void restrict_stencil(Grid& dstcoarse, Grid& srcfine);

		//void restrict_adjoint_stencil(Grid& dstcoarse, Grid& srcfine);

		void writeNodePos(const std::string& filename, Grid& g);

		void writeEidMap(const std::string& filename, Grid& g);

		void writeVidMap(const std::string& filename, Grid& g);

		void writeSurfaceElement(const std::string& filename);

		void writeV2Vcoarse(const std::string& filename, Grid& g);

		void writeV2Vfine(const std::string& filename, Grid& g);

		void writeV2V(const std::string& filename, Grid& g);

		void writeDensity(const std::string& filename);

		void readDensity(const std::string& filename);

		void writeSensitivity(const std::string& filename);

		void writeComplianceDistribution(const std::string& filename);

		void writeSupportForce(const std::string& filename);

		void writeDisplacement(const std::string& filename);

		void getNodePos(Grid& g, std::vector<double>& p3host);

		void update_stencil(void);

		//void update_adjoint_stencil(void);

		double v_cycle(int pre_relax = 1, int post_relax = 1);

		double v_halfcycle(int depth, int pre_relax = 1, int post_relax = 1);

		//double adjoint_v_cycle(void);

		void test_vcycle(void);

		void test_kernels(void);

		//void test_adjoint_v_cycle(void);

		int n_grid(void) { return _gridlayer.size(); }

		double elementLength(void);

		~HierarchyGrid() {
			if (!_gridlayer.empty()) {
				for (int i = 0; i < _gridlayer.size(); i++) {
					delete _gridlayer[i];
				}
			}
		}

	};

};

#endif

