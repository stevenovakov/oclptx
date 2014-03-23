/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

/* oclptx.cc
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


#include <iostream>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
// adds exception support from CL libraries
// define before CL headers inclusion

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "oclptx.h"

//*********************************************************************
//
// Assorted Function Declerations
//
//*********************************************************************

void SimpleInterpolationTest( cl::Context * ocl_context,
                              cl::CommandQueue * cq,
                              cl::Kernel * test_kernel
                            );

std::string GenerateSaveFile();

//*********************************************************************
//
// Main
//
//*********************************************************************

int main(int argc, char *argv[] )
{
  // stuff set by s_manager and environment

  // Test Routine
  //OclEnv environment("interptest");
  //
  // OclEnv should only ever be declared once (can rewrite as singleton
  // class later). Recompile programs with ->SetOclRoutine(...)
  //
  //SimpleInterpolationTest(environment.GetContext(),
                          //environment.GetCq(0),
                          //environment.GetKernel(0));

  // Sample Manager

  SampleManager& s_manager = SampleManager::GetInstance();
  if(&s_manager == NULL)
  {
     std::cout<<"\n Null value"<<std::endl;
  }
  else
  {
    s_manager.ParseCommandLine(argc, argv);

    const BedpostXData* f_data = s_manager.GetFDataPtr();
    const BedpostXData* theta_data = s_manager.GetThetaDataPtr();
    const BedpostXData* phi_data = s_manager.GetPhiDataPtr();

    unsigned int n_particles = s_manager.GetNumParticles();
    unsigned int max_steps = s_manager.GetNumMaxSteps();

    // do not calld delete[] on
    const float4* initial_positions =
      s_manager.GetSeedParticles()->data();
      
    const unsigned short int* brain_mask =
      s_manager.GetBrainMaskToArray();
    
    // TODO get curv thresh from sample manager
    float curvature_threshold = 0.2;
      
    OclEnv environment("standard");
    unsigned int n_devices = environment.HowManyDevices();

    int enough_mem;
    enough_mem = environment.AvailableGPUMem(); // will pass args later

    if (enough_mem < 0)
    {
      printf("Insufficient GPU Memory: Terminating Program\n\n";)
    }
    else
    {
      //
      // and then (this is a naive, "serial" implementation;
      //

      std::cout<<"Using " << n_devices << " Devices\n";

      std::vector<OclPtxHandler*> handlers;

      for (unsigned int d = 0; d < n_devices; d++)
      {
        std::cout<<"Device " << d <<"\n";

        handlers.push_back(
          new OclPtxHandler(environment.GetContext(),
                            environment.GetCq(d),
                            environment.GetKernel(d),
                            curvature_threshold));

        std::cout<<"\tinit done\n";
        handlers.back()->WriteSamplesToDevice(
                                      f_data,
                                      phi_data,
                                      theta_data,
                                      static_cast<unsigned int>(1),
                                      brain_mask);
        std::cout<<"\tsamples done\n";
        handlers.back()->WriteInitialPosToDevice(
                                          initial_positions,
                                          n_particles,
                                          max_steps,
                                          n_devices,
                                          d);
        std::cout<<"\tpos done\n";
        handlers.back()->PrngInit();
        std::cout<<"\tPrng Done\n";
        handlers.back()->SingleBufferInit();
        //handler.DoubleBufferInit( n_particles/2, max_steps);
        std::cout<<"\tdbuff done\n";
      }

      for (unsigned int d = 0; d < n_devices; d++)
      {
        handlers.at(d)->BlockCq();
      }

      for (unsigned int d = 0; d < n_devices; d++)
      {
        std::cout<<"Device " << d << ", Total GPU Memory Allocated (MB): "<<
          handlers.at(d)->GpuMemUsed()/1e6 << "\n";
      }

      std::cout<<"Press Any Button To Continue...\n";
      std::cin.get();

      for (unsigned int d = 0; d < n_devices; d++)
      {
        handlers.at(d)->Interpolate();
        std::cout<<"Device " << d << ", interp done\n";
      }
      //handler.Reduce();
      //std::cout<<"reduce done\n";
      //handler.Interpolate();
      //std::cout<<"interp done\n";
      std::string path_filename = GenerateSaveFile();
      FILE * path_file;
      path_file = fopen(path_filename.c_str(), "wb");
      fprintf(path_file, "[");
      fclose(path_file);

      for (unsigned int d = 0; d < n_devices; d++)
      {
        handlers.at(d)->ParticlePathsToFile(path_filename);

        if( d < n_devices - 1)
        {
          path_file = fopen(path_filename.c_str(), "ab");
          fprintf(path_file, ",\n");
          fclose(path_file);
        }
      }

      path_file = fopen(path_filename.c_str(), "ab");
      fprintf(path_file, "]");
      fclose(path_file);

      for (unsigned int d = 0; d < n_devices; d++)
      {
        delete handlers.at(d);
      }
    }

    delete[] brain_mask;
  }

  std::cout<<"\n\nExiting...\n\n";
  return 0;
}

//*********************************************************************
//
// Assorted Functions
//
//*********************************************************************
std::string GenerateSaveFile()
{
  std::ostringstream convert(std::ostringstream::ate);
  std::string path_filename;

  std::vector<float> temp_x;
  std::vector<float> temp_y;
  std::vector<float> temp_z;

  time_t t = time(0);
  struct tm * now = localtime(&t);

  convert << "OclPtx Results/"<< now->tm_yday << "-" <<
    static_cast<int>(now->tm_year) + 1900 << "_"<< now->tm_hour <<
      ":" << now->tm_min << ":" << now->tm_sec;

  path_filename = convert.str() + "_PATHS.dat";

  return path_filename;
}



void SimpleInterpolationTest( cl::Context* ocl_context,
                              cl::CommandQueue* cq,
                              cl::Kernel* test_kernel)
{
  //*******************************************************************
  //
  //  TEST ROUTINE
  //
  //*******************************************************************

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