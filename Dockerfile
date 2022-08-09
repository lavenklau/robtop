FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update && apt-get install -y tzdata \
 && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
 && dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y git cmake libeigen3-dev libgl1-mesa-dev libglu1-mesa-dev libxi-dev libgflags-dev wget libboost-iostreams-dev libtbb-dev libblosc-dev libcgal-dev libglm-dev rapidjson-dev

# download source
RUN cd /home && mkdir src && cd src && mkdir robtop

WORKDIR /home/src/robtop

# install openmesh
RUN wget https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh/-/archive/OpenMesh-8.1/OpenMesh-OpenMesh-8.1.tar.gz && tar zvxf OpenMesh-OpenMesh-8.1.tar.gz && cd OpenMesh-OpenMesh-8.1 && mkdir build && cd build && cmake .. && make -j4 && make install

# build trimesh2
RUN git clone https://github.com/Forceflow/trimesh2.git && cd trimesh2 && make -j4

# download spectra
RUN git clone https://github.com/yixuan/spectra.git

# install openvdb
RUN wget https://github.com/AcademySoftwareFoundation/openvdb/archive/refs/tags/v9.1.0.tar.gz -O openvdb9.1.0.tar.gz && tar zvxf openvdb9.1.0.tar.gz && cd openvdb-9.1.0 && mkdir build && cd build && cmake .. && make -j2 && make install

# build source
RUN git init && git remote add origin https://github.com/lavenklau/robtop.git && git pull origin main 
RUN mkdir build && cd build && cmake -DTRIMESH_LIBRARY=`pwd`/../trimesh2/lib.Linux64/libtrimesh.a -DTRIMESH_INCLUDE_DIR=`pwd`/../trimesh2/include/ -DSPECTRA_INCLUDE_DIR=`pwd`/../spectra/include/ -DCMAKE_MODULE_PATH=/usr/local/lib/cmake/OpenVDB .. && make -j4

