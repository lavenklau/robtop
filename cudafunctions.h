#pragma once


extern void calcDistanceSegs2Segs_g(
	int nPairs, const double* p0[3], const double* q0[3], const double* p1[3], const double* q1[3],
	double* distance, double* t1, double* t2,
	double* gradients[12]
);
	
extern void sqrt_g(int _size, double* _start);

// async
extern void nodeNodesAngles_gAsync(int n_node, const double* nodepos[3], const double* neighpos[4][3], bool use_cos = true);
extern void nodeNodesAngles_gsync(int n_node, double* angs[6], double* grad[6]);
