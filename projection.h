#pragma once

#ifndef __PROJECTION_H
#define __PROJECTION_H

#include<vector>
#include"Eigen/Eigen"
#include "Grid.h"

#include "cublas_v2.h"

extern cublasHandle_t cublas_handle;

using namespace grid;

void setNodes(BitSAT<unsigned int>& vbits, int vreso, const std::vector<int>& lex2gs, const int* vlex2gs_dev, const int* nodeflag, int n_gs);

void setLoadNodes(
	const std::vector<int>& loadnodes,
	const std::vector<Eigen::Matrix<double, 3, 1>>& loadpos,
	const std::vector<Eigen::Matrix<double, 3, 1>>& loadnormal,
	const std::vector<Eigen::Matrix<double, 3, 1>>& loadforce
);

void setLoadForce(std::vector<Eigen::Matrix<double, 3, 1>> flist);

const std::vector<int>& getLoadNodes(void);

void forceProject(std::vector<double> fsupport[3]);

void forceProject(double* f_dev[3]);

void forceProjectComplementary(double* f_dev[3], bool Ncoords = false);

void forceRestoreProjection(double* f_dev[3]);

void getForceSupport(double const * const f_dev[3], double* fsup[3]);

void getForceSupport(double const * const f_dev[3], std::vector<double> fs[3]);

void writeSupportForce(const std::string& filename, double const * const f_dev[3]);

void setForceSupport(double const * const fsup[3], double* f_dev[3]);

void displacementProject(double* u_dev[3]);

void uploadRigidMatrix(double* pdata[3], int n_gs);

void uploadNodeFlags(const int* nodeflags, int n_gs);

void uploadLoadNodes(const std::vector<int>& loadnodes, std::vector<double> vtang[2][3], std::vector<double> vnormal[3]);

void uploadLoadForce(double const* const fhost[3]);

double** getPreloadForce(void);

double** getForceNormal(void);

int n_loadnodes(void);

void cleanProjection(void);

void uploadRigidDisplacement(double* udst[3], int k);

size_t projectionGetMem(void);

#endif


