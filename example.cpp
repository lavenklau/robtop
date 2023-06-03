#include "optimization.h"
#include "projection.h"

using namespace grid;

std::vector<float> readVertices3D(const std::string& path) {
  std::string line;
  std::ifstream is(path);
  std::vector<float> vertices;
  while(std::getline(is,line)) {
    char c;
    double vx,vy,vz;
    if((int)line.size()>1 && line[0]=='v' && line[1]==' ') {
      std::istringstream(line) >> c >> vx >> vy >> vz;
      vertices.push_back(vx);
      vertices.push_back(vy);
      vertices.push_back(vz);
    }
  }
  return vertices;
}
std::vector<int> readTriangles(const std::string& path) {
  std::string line;
  std::ifstream is(path);
  std::vector<int> faces;
  while(std::getline(is,line)) {
    char c;
    std::string fx,fy,fz;
    if((int)line.size()>1 && line[0]=='f' && line[1]==' ') {
      std::istringstream(line) >> c >> fx >> fy >> fz;
      faces.push_back(atoi(strtok((char*)fx.c_str(),"/"))-1);
      faces.push_back(atoi(strtok((char*)fy.c_str(),"/"))-1);
      faces.push_back(atoi(strtok((char*)fz.c_str(),"/"))-1);
    }
  }
  return faces;
}
int ex_main() {
  //std::string path="cube.obj";
  std::string path="sphere.obj";
  std::vector<float> pcoords=readVertices3D(path);
  std::vector<int> facevertices=readTriangles(path);
  for(int i=0; i<(int)pcoords.size(); i+=3) {
    pcoords[i+1]*=0.5;
    pcoords[i+2]*=0.25;
  }
  setParameters(0.5,0.1,0.1,0.1,0.1,1,1e-3,128,1,0.3,2,false,false);

  setBoundaryCondition([](double pos[3])->bool {
    return pos[0]<0.1;
  },[](double pos[3])->bool {
    return pos[0]>0.9;
  },[](double pos[3])->Eigen::Matrix<double,3,1> {
    return Eigen::Matrix<double,3,1>(1,0,0);
  });

  grids.setMode(with_support_free_force);

  buildGrids(pcoords,facevertices);

  setForceSupport(getPreloadForce(), grids[0]->getForce());

  uploadTemplateMatrix();
  initDensities(0.3);
  update_stencil();

  grids.test_vcycle();
  /*grids.resetAllResidual();
  grids[0]->randForce();
  grids[0]->reset_displacement();
  for(int i=0; i<100; i++)
    std::cout << "VCycle " << i << " residual=" << grids.v_cycle() << std::endl;*/
  return 0;
}