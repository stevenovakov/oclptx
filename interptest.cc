/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

/* interptest.h
 *
 *
 * Part of
 *    oclptx
 * OpenCL-based, GPU accelerated probtrackx algorithm module, to be used
 * with FSL - FMRIB's Software Library
 *
 * This file is part of oclptx.
 *
 * oclptx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * oclptx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with oclptx.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <time.h>
#include <cstdlib>
#include <ctime>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

#define __CL_ENABLE_EXCEPTIONS
// adds exception support from CL libraries
// define before CL headers inclusion

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "customtypes.h"

//
// Declarations
//
FloatVolume CreateVoxelSpace(unsigned int NX, unsigned int NY,
  unsigned int NZ, float4 min_bounds, float4 max_bounds);
FloatVolume CreateFlowSpace(FloatVolume voxel_space,
  float dr, float3 setpoints);
float4 FlowFunction(float4 coords, float dr, float3 setpoints);
std::vector<float4> RandSeedPoints(int n,
  IntVolume vvol, std::vector<unsigned int> seed_elem);
std::vector<unsigned int> RandSeedElem(int n,
  float3 mins, float3 maxs, IntVolume vvol);

void VolumeToFile(IntVolume ivol, FloatVolume fvol);
void PathsToFile(std::vector<float4> path_vector,
  unsigned int n_seeds, unsigned int n_steps);

void SimpleInterpolationTest( cl::Context* ocl_context,
                              cl::CommandQueue* cq,
                              cl::Kernel* test_kernel);

std::vector<float4> InterpolationTestRoutine
(
  FloatVolume voxel_space,
  FloatVolume flow_space,
  std::vector<float4> seed_space,
  std::vector<unsigned int> seed_elem,
  unsigned int n_seeds,
  unsigned int n_steps,
  float dr,
  float4 min_bounds,
  float4 max_bounds,
  cl::Context * ocl_context,
  std::vector<cl::CommandQueue> * cq,
  cl::Kernel * test_kernel
);

//
// Floating point voxel space
//
FloatVolume CreateVoxelSpace(unsigned int NX, unsigned int NY,
  unsigned int NZ, float4 min_bounds, float4 max_bounds)
{
  FloatVolume voxel_space;

  voxel_space.nx = NX;
  voxel_space.ny = NY;
  voxel_space.nz = NZ;

  float dx = (max_bounds.x - min_bounds.x)/static_cast<float>(NX);
  float dy = (max_bounds.y - min_bounds.y)/static_cast<float>(NY);
  float dz = (max_bounds.z - min_bounds.z)/static_cast<float>(NZ);

  float4 temp;
  temp.t = 0.0;

  for (unsigned int k = 0; k < NZ; k ++)
  {
        for (unsigned int j = 0; j < NY; j++)
        {
            for (unsigned int i = 0; i < NX; i++)
            {
              temp.x = i*dx + min_bounds.x;
              temp.y = j*dy + min_bounds.y;
              temp.z = k*dz + min_bounds.z;
              voxel_space.vol.push_back(temp);
              temp.x += dx;
              voxel_space.vol.push_back(temp);
              temp.y += dy;
              voxel_space.vol.push_back(temp);
              temp.x -= dx;
              voxel_space.vol.push_back(temp);
              temp.y -= dy;
              temp.z += dz;
              voxel_space.vol.push_back(temp);
              temp.x += dx;
              voxel_space.vol.push_back(temp);
              temp.y += dy;
              voxel_space.vol.push_back(temp);
              temp.x -= dx;
              voxel_space.vol.push_back(temp);
            }
        }
  }

  return voxel_space;
}

FloatVolume CreateFlowSpace(FloatVolume voxel_space,
  float dr, float3 setpoints)
{
  FloatVolume flow_space;
  flow_space.nx = voxel_space.nx;
  flow_space.ny = voxel_space.ny;
  flow_space.nz = voxel_space.nz;

  std::vector<float4>::iterator vit;

  for (vit = voxel_space.vol.begin();
    vit != voxel_space.vol.end(); ++vit)
  {
      flow_space.vol.push_back(FlowFunction(*vit,
                                            dr,
                                            setpoints));
  }

  return flow_space;
}

float4 FlowFunction(float4 coords, float dr, float3 setpoints)
{
  //
  // Right now its just like a "tree" from center of space to top
  //
  float r = dr;
  float theta = std::atan(1.0) * coords.z/ setpoints.z;
  float phi =
    std::atan2(static_cast<float>(coords.y - setpoints.y),
      static_cast<float>(coords.x - setpoints.x));

  float4 ret;
  ret.x = r;
  ret.y = phi;
  ret.z = theta;
  ret.t = 0;

  return ret;
}

std::vector<unsigned int> RandSeedElem(unsigned int n, float3 mins,
  float3 maxs, FloatVolume vvol)
{
  std::vector<unsigned int> seed_elem;
  int temp_elem;

  unsigned int x, y, z;

  unsigned int seed = time(NULL);

  for (unsigned int i = 0; i < n; i++)
  {
    rand_r(&seed);
    x = static_cast<unsigned int>(
      (seed%1000*(maxs.x - mins.x))/1000.0 + mins.x);
    rand_r(&seed);
    y = static_cast<unsigned int>(
      (seed%1000*(maxs.y - mins.y))/1000.0 + mins.y);
    rand_r(&seed);
    z = static_cast<unsigned int>(
      (seed%1000*(maxs.z - mins.z))/1000.0 + mins.z);

    temp_elem = z*(8*vvol.nx*vvol.ny) + y*8*vvol.nx + 8*x;
    // "elem #" is START of vertex list 0, 1, 2, ...., 6, 7

    seed_elem.push_back((unsigned int) temp_elem);
  }

  return seed_elem;
}

std::vector<float4> RandSeedPoints(int n,
  FloatVolume vvol, std::vector<unsigned int> seed_elem)
{
  std::vector<float4> seed_set;
  float4 temp_point;
  temp_point.t = 0;

  int maxx, minx, maxy, miny, maxz, minz;

  unsigned int seed = time(NULL);

  for (unsigned int i = 0; i < seed_elem.size(); i++)
  {
    minx = vvol.vol.at(seed_elem.at(i)).x;
    maxx = vvol.vol.at(seed_elem.at(i) + 1).x;

    miny = vvol.vol.at(seed_elem.at(i)).y;
    maxy = vvol.vol.at(seed_elem.at(i)+2).y;

    minz = vvol.vol.at(seed_elem.at(i)).z;
    maxz = vvol.vol.at(seed_elem.at(i)+4).z;

    rand_r(&seed);
    temp_point.x = seed%1000*(maxx - minx)/1000.0 + minx;
    rand_r(&seed);
    temp_point.y = seed%1000*(maxy - miny)/1000.0 + miny;
    rand_r(&seed);
    temp_point.z = seed%1000*(maxz - minz)/1000.0 + minz;

    seed_set.push_back(temp_point);
  }

  return seed_set;
}


void VolumeToFile(FloatVolume vvol, FloatVolume fvol)
{
  int vsize = vvol.vol.size()/8;
  int fsize = fvol.vol.size()/8;

  std::ostringstream convert(std::ostringstream::ate);

  std::string volume_filename;
  std::string flow_filename;

  time_t t = time(0);
  struct tm * now = localtime(&t);

  convert << "Test Data/"<< now->tm_yday << "-" <<
    static_cast<int>(now->tm_year) + 1900 << "_"<< now->tm_hour <<
      ":" << now->tm_min << ":" << now->tm_sec;

  volume_filename = convert.str() + "_VOL.dat";
  flow_filename = convert.str() + "_FLOW.dat";

  std::fstream volume_file;
  volume_file.open(volume_filename.c_str(),
    std::ios::app|std::ios::out);

  for (int i = 0; i < vsize; i ++)
  {
    for (int j = 0; j < 8; j++)
    {
      volume_file<< vvol.vol.at(j + i).x <<"," <<
        vvol.vol.at(j + i).y << "," << vvol.vol.at(j + i).z;

      if ( j < 7 )
        volume_file << ",";
    }

    if (i == vsize/8 - 1)
      break;
    else
      volume_file << "\n";
  }

  volume_file.close();


  std::fstream flow_file;
  flow_file.open(flow_filename.c_str(), std::ios::app|std::ios::out);

  for (int i = 0; i < fsize; i ++)
  {
    for (int j = 0; j < 8; j++)
    {
      flow_file<< fvol.vol.at(j + i).x <<"," << fvol.vol.at(j + i).y <<
        "," << fvol.vol.at(j + i).z;

      if ( j < 7 )
        flow_file << ",";
    }

    if (i == fsize - 1)
      break;
    else
      flow_file << "\n";
  }

  flow_file.close();
}


void PathsToFile(std::vector<float4> path_vector,
  unsigned int n_seeds, unsigned int n_steps)
{
  std::ostringstream convert(std::ostringstream::ate);

  std::string path_filename;

  std::vector<float> temp_x;
  std::vector<float> temp_y;
  std::vector<float> temp_z;

  time_t t = time(0);
  struct tm * now = localtime(&t);

  convert << "Test Data/"<< now->tm_yday << "-" <<
    static_cast<int>(now->tm_year) + 1900 << "_"<< now->tm_hour <<
      ":" << now->tm_min << ":" << now->tm_sec;

  path_filename = convert.str() + "_PATH.dat";
  std::cout << "Writing to " << path_filename << "\n";

  std::fstream path_file;
  path_file.open(path_filename.c_str(), std::ios::app|std::ios::out);

  for (unsigned int n = 0; n < n_seeds; n ++)
  {
    for (unsigned int s = 0; s < n_steps; s++)
    {
      temp_x.push_back(path_vector.at(n*n_steps + s).x);
      temp_y.push_back(path_vector.at(n*n_steps + s).y);
      temp_z.push_back(path_vector.at(n*n_steps + s).z);
    }

    for (unsigned int i = 0; i < (unsigned int) n_steps; i++)
    {
      path_file << temp_x.at(i);

      if (i < (unsigned int) n_steps - 1)
        path_file << ",";
      else
        path_file << "\n";
    }

    for (unsigned int i = 0; i < (unsigned int) n_steps; i++)
    {
      path_file << temp_y.at(i);

      if (i < (unsigned int) n_steps - 1)
        path_file << ",";
      else
        path_file << "\n";
    }

    for (unsigned int i = 0; i < (unsigned int) n_steps; i++)
    {
      path_file << temp_z.at(i);

      if (i < (unsigned int) n_steps - 1)
        path_file << ",";
      else
        path_file << "\n";
    }

    temp_x.clear();
    temp_y.clear();
    temp_z.clear();
  }

  path_file.close();
}

std::vector<float4> InterpolationTestRoutine
    (
      FloatVolume voxel_space,
      FloatVolume flow_space,
      std::vector<float4> seed_space,
      std::vector<unsigned int> seed_elem,
      unsigned int n_seeds,
      unsigned int n_steps,
      float dr,
      float4 min_bounds,
      float4 max_bounds,
      cl::Context * ocl_context,
      cl::CommandQueue * cq,
      cl::Kernel * test_kernel
    )
{

  std::vector<cl::Buffer> read_buffer_set;
  std::vector<cl::Buffer> write_buffer_set;
  std::vector<unsigned int> read_buffer_set_sizes;
  std::vector<unsigned int> write_buffer_set_sizes;

  bool blocking_write = false;

  write_buffer_set_sizes.clear();
  read_buffer_set_sizes.clear();

  unsigned int vol_size = voxel_space.vol.size();
  unsigned int vol_item_size = sizeof(voxel_space.vol.at(0));

  unsigned int flow_size = flow_space.vol.size();
  unsigned int flow_item_size = sizeof(flow_space.vol.at(0));

  unsigned int seed_size = seed_space.size();
  unsigned int seed_item_size = sizeof(seed_space.at(0));

  unsigned int vol_mem_size = vol_size * vol_item_size;
  unsigned int flow_mem_size = flow_size*flow_item_size;
  unsigned int seed_mem_size = seed_size*seed_item_size;

  read_buffer_set_sizes.push_back(vol_mem_size); //1
  read_buffer_set_sizes.push_back(flow_mem_size); //2
  read_buffer_set_sizes.push_back(seed_mem_size); //3

  unsigned int result_num = n_seeds*n_steps;
  unsigned int result_mem_size = result_num * flow_item_size;
  unsigned int seed_elem_mem_size = n_seeds*sizeof(seed_elem.at(0));

  write_buffer_set_sizes.push_back(result_mem_size); //1
  write_buffer_set_sizes.push_back(seed_elem_mem_size); //2

  cl::NDRange test_global_range(n_seeds);
  cl::NDRange test_local_range(1);

  // Instantiate Buffers

  for( unsigned int i = 0;
    i < read_buffer_set_sizes.size(); i++)
  {
    std::cout<<"RBUF: "<<read_buffer_set_sizes.at(i)<<"\n";
    read_buffer_set.push_back(
      cl::Buffer( *ocl_context,
                  CL_MEM_READ_ONLY,
                  read_buffer_set_sizes.at(i),
                  NULL,
                  NULL
                )
    );
  }
  for( unsigned int i = 0;
    i < write_buffer_set_sizes.size(); i++)
  {
    std::cout<<"WBUF: "<<write_buffer_set_sizes.at(i)<<"\n";
    write_buffer_set.push_back(
      cl::Buffer( *ocl_context,
                  CL_MEM_READ_WRITE,
                  write_buffer_set_sizes.at(i),
                  NULL,
                  NULL
                )
    );
  }

  //write to buffers on device

  //flow space
  cq->enqueueWriteBuffer(
        read_buffer_set.at(0),
        blocking_write,
        (unsigned int) 0,
        read_buffer_set_sizes.at(0),
        voxel_space.vol.data(),
        NULL,
        NULL
  );
  //flow space
  cq->enqueueWriteBuffer(
        read_buffer_set.at(1),
        blocking_write,
        (unsigned int) 0,
        read_buffer_set_sizes.at(1),
        flow_space.vol.data(),
        NULL,
        NULL
  );
  //seed initial pos
  cq->enqueueWriteBuffer(
        read_buffer_set.at(2),
        blocking_write,
        (unsigned int) 0,
        read_buffer_set_sizes.at(2),
        seed_space.data(),
        NULL,
        NULL
  );
  // seed elem
  cq->enqueueWriteBuffer(
        write_buffer_set.at(1),
        blocking_write,
        (unsigned int) 0,
        write_buffer_set_sizes.at(1),
        seed_elem.data(),
        NULL,
        NULL
  );

  // do NOT call delete on this.

  test_kernel->setArg(0, read_buffer_set.at(0)); //voxels
  test_kernel->setArg(1, read_buffer_set.at(1)); //flow
  test_kernel->setArg(2, read_buffer_set.at(2)); //seed pts
  test_kernel->setArg(3, write_buffer_set.at(1)); //seed elem
  test_kernel->setArg(4, voxel_space.nx); //nx
  test_kernel->setArg(5, voxel_space.ny); //ny
  test_kernel->setArg(6, voxel_space.nz); //nz
  test_kernel->setArg(7, min_bounds); //ny
  test_kernel->setArg(8, max_bounds); //nz
  test_kernel->setArg(9, n_steps); //n_steps
  test_kernel->setArg(10, write_buffer_set.at(0));//path container

  // OCL CQ BLOCK
  cq->finish();

  cq->enqueueNDRangeKernel(
    *test_kernel,
    cl::NullRange,
    test_global_range,
    test_local_range,
    NULL,
    NULL
  );

  // OCL CQ BLOCK
  cq->finish();

  //instantiate return container
  std::vector<float4> return_container(result_num);

  std::cout<<
    return_container.size()*sizeof(return_container.at(result_num))<<
      "\n";
  std::cout<<n_seeds*n_steps*sizeof(float4)<<"\n";
  std::cout<<result_mem_size<<"\n";

  //fill return container
  cq->enqueueReadBuffer(
    write_buffer_set.at(0),
    CL_TRUE, // blocking
    0,
    result_mem_size,
    &return_container[0]
  );

  return return_container;
}


//*******************************************************************
//
//  TEST ROUTINE (s)
//
//*******************************************************************
void SimpleInterpolationTest( cl::Context* ocl_context,
                              cl::CommandQueue* cq,
                              cl::Kernel* test_kernel)
{
  auto t_end = std::chrono::high_resolution_clock::now();
  auto t_start = std::chrono::high_resolution_clock::now();

  unsigned int XN = 20;
  unsigned int YN = 20;
  unsigned int ZN = 20;

  unsigned int nseeds = 500;
  unsigned int nsteps = 200;

  std::cout<<"\n\nInterpolation Test\n"<<"\n";
  std::cout<<"\tSeeds :" << nseeds << " Steps:" << nsteps <<"\n";
  std::cout<<"\tXN: " << XN << " YN: " << YN << " ZN: " << ZN <<"\n";
  std::cout<<"\n\n";

  float3 mins;
  mins.x = 8.0;
  mins.y = 8.0;
  mins.z = 0.0;
  float3 maxs;
  maxs.x = 12.0;
  maxs.y = 12.0;
  maxs.z = 1.0;

  float4 min_bounds;
  min_bounds.x = 0.0;
  min_bounds.y = 0.0;
  min_bounds.z = 0.0;
  min_bounds.t = 0.0;

  float4 max_bounds;
  max_bounds.x = 20.0;
  max_bounds.y = 20.0;
  max_bounds.z = 20.0;
  max_bounds.t = 0.0;

  float dr = 0.1;

  FloatVolume voxel_space = CreateVoxelSpace( XN, YN, ZN,
    min_bounds, max_bounds);

  float3 setpts;
  setpts.z = max_bounds.z - min_bounds.z;
  setpts.y = (max_bounds.y + min_bounds.y)/2.0;
  setpts.x = (max_bounds.x + min_bounds.x)/2.0;

  FloatVolume flow_space = CreateFlowSpace( voxel_space, dr, setpts);
  std::vector<unsigned int> seed_elem = RandSeedElem(
    nseeds,
    mins,
    maxs,
    voxel_space
  );

  std::vector<float4> seed_space = RandSeedPoints(  nseeds,
                                                    voxel_space,
                                                    seed_elem
                                                  );

  VolumeToFile(voxel_space, flow_space);

  t_start = std::chrono::high_resolution_clock::now();

  std::vector<float4> path_vector =
    InterpolationTestRoutine(   voxel_space,
                                flow_space,
                                seed_space,
                                seed_elem,
                                nseeds,
                                nsteps,
                                dr,
                                min_bounds,
                                max_bounds,
                                ocl_context,
                                cq,
                                test_kernel
  );

  t_end = std::chrono::high_resolution_clock::now();
  std::cout<< "Interpolation Test Time:" <<
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        t_end-t_start).count() << std::endl;

  PathsToFile(  path_vector,
                nseeds,
                nsteps
  );
}

// EOF