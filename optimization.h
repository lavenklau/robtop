#pragma once

#ifndef __OPTIMIZATION_HH
#define __OPTIMIZATION_HH


#include "Grid.h"

extern grid::HierarchyGrid grids;

struct Parameter {
	float damp_ratio;
	float design_step;
	float power_penalty;
	float volume_ratio;
	float volume_decrease;
	int   filter_radius;
	float min_rho;
	int gridreso;
	float youngs_modulu;
	float poisson_ratio;
};

extern Parameter params;

extern gpu_manager_t gpu_manager;

void buildGrids(const std::vector<float>& coords, const std::vector<int>& trifaces);

void logParams(std::string file, std::string version_str, int argc, char** argv);

void setParameters(
	float volRatio, float volDecrease, float designStep, float filterRadi, float dampRatio, float powerPenal,
	float min_density, int gridreso, float youngs_modulu, float poisson_ratio, float shell_width,
	bool logdensity, bool logcompliance);

void setOutpurDir(const std::string& dirname);

void setWorkMode(const std::string& modestr);

void setDEBUG(bool debug = false);

double solveAdjointSystem(void);

void selfTest(void);

void solveFEM(void);

// solve the worst-case displacement using the modified power method, return the final compliance
double modifiedPM(void);

double eigenCG(void);

// (pnorm, knorm)
std::pair<double, double> RayleighGradient(double* u[3], double* g[3]);

double Pnorm(double* u[3]);

double Knorm(double* u[3]);

double inCompleteModiPM(double fch_thres = 1e-2);

double MGPSOR(void);

void computeSensitivity(void);

bool checkAdjointVariable(void);

float updateDensities(float Vgoal);

void optimization(void);

grid::HierarchyGrid& getGrids(void);

void setBoundaryCondition(std::function<bool(double[3])> fixarea, std::function<bool(double[3])> loadarea, std::function<Eigen::Matrix<double, 3, 1>(double[3])> forcefield);

// upload template matrix and power penalty coefficient
void uploadTemplateMatrix(void);

void initDensities(double rho);

void update_stencil(void);

void test_rigid_displacement(void);

#endif

