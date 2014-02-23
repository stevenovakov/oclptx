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

std::string DetermineKernel(); //args undetermined yet

//*********************************************************************
//
// Main
//
//*********************************************************************

int main(int argc, char *argv[] )
{
  // stuff set by s_manager and environment
  std::string compute_kernel;
  unsigned int n_particles;
  unsigned int n_max_steps;


  // Test Routine
  OclEnv environment("interptest");
  //
  // OclEnv should only ever be declared once (can rewrite as singleton
  // class later). Recompile programs with ->SetOclRoutine(...)
  //
  SimpleInterpolationTest(environment.GetContext(),
                          environment.GetCq(0),
                          environment.GetKernel(0));

  // Sample Manager

  SampleManager& s_manager = SampleManager::GetInstance();
  if(&s_manager == NULL)
  {
     std::cout<<"\n Null value"<<std::endl;
  }
  else
  {
    s_manager.ParseCommandLine(argc, argv);

    const vector<Matrix>* thetaSamples = s_manager.GetThetaSamples();
    const vector<Matrix>* phiSamples = s_manager.GetPhiSamples();
    const vector<Matrix>* fSamples = s_manager.GetFSamples();
    Matrix thetaMatrix = thetaSamples->at(0);
    Matrix phiMatrix = phiSamples->at(0);
    Matrix fMatrix = fSamples->at(0);

    std::cout<<"thetaMatrix, Rows: " << thetaMatrix.Nrows()<<
      " Cols: "<< thetaMatrix.Ncols() <<"\n";
    std::cout<<"phiMatrix, Rows: " << phiMatrix.Nrows() <<
      " Cols: "<< phiMatrix.Ncols() <<"\n";
    std::cout<<"fMatrix, Rows: " << fMatrix.Nrows() <<
      " Cols: "<< fMatrix.Ncols() <<"\n";
    //for (int row = 1; row < thetaMatrix.Nrows(); row++)
    //{
      //for (int col = 1; col < thetaMatrix.Ncols(); col++)
      //{
         //if (thetaMatrix(row,col) != 0)
         //{
            //float thetaTest=thetaMatrix(row,col);
            //float phiTest = phiMatrix(row,col);
            //float fTest = fMatrix(row,col);
            //std::cout <<"\n Rows Theta:"<<row<<" Cols Theta:"<<
              //col<<" "<<thetaTest<<std::endl;
            //std::cout <<"\n Rows Phi:"<<row<<" Cols Phi:"<<
              //col<<" "<<phiTest<<std::endl;
            //std::cout <<"\n Rows f:"<<row<<" Cols f:"<<
              //col<<" "<<fTest<<std::endl;
         //}
      //}
    //}

    // somwhere in here, this should initialize, based on s_manager
    // actions:
    //
    // OclEnv environment

    //
    // and then (this is a naive, "serial" implementation;
    //

    ////OclPtxHandler handler( args );
    //handler.WriteSamplesToDevice( args);
    //handler.WriteInitialPosToDevice( args);
    //handler.DoubleBufferInit( args);

    //handler.Interpolate();
    //handler.Reduce();
    //handler.Interpolate();

    //handler.ParticlePathsToFile());

  }

  std::cout<<"\n\nExiting...\n\n";
  return 0;
}

//*********************************************************************
//
// Assorted Functions
//
//*********************************************************************
std::string DetermineKernel()
{
  return std::string("interptest");   
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