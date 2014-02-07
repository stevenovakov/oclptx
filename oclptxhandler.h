/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */
 
/* oclptxhandler.h
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

#ifndef  OCLPTX_OCLPTXHANDLER_H_
#define  OCLPTX_OCLPTXHANDLER_H_

#include <iostream>
#include <vector>
//#include <thread>
//#include <mutex>

#define __CL_ENABLE_EXCEPTIONS
// adds exception support from CL libraries
// define before CL headers inclusion

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "customtypes.h" 

 
class OclPtxHandler{
  
  public:
    
    OclPtxHandler(){};
    
    OclPtxHandler( std::string ocl_routine);
  
    ~OclPtxHandler();
    
    //
    // Container Set/Get
    //
    
    void SetOclRoutine(std::string new_routine);
    
    //
    // Thread Management
    //
    
    void PTStart();
    
    void ThreadController();
    
    //
    // OpenCL API Interface/Helper Functions
    //
    
    void OclInit();
    
    void OclDeviceInfo();
    
    void NewCLCommandQueues();
    
    cl::Program CreateProgram();
    
    void InstantiateBuffers();

    //
    // Interpolation
    //
    
    std::vector<float4> InterpolationTestRoutine( 
      FloatVolume voxel_space,
      FloatVolume flow_space, 
      std::vector<float4> seed_space,
      std::vector<unsigned int> seed_elem,
      unsigned int n_seeds,
      unsigned int n_steps,
      float dr,
      float4 min_bounds, 
      float4 max_bounds
    );
    
  
  private:
  
    //
    // Containers
    
  
    //
    // OpenCL Objects
    //
    cl::Context ocl_context;
    
    std::vector<cl::Platform> ocl_platforms;    
    
    std::vector<cl::Device> ocl_devices;
    
    std::vector<cl::CommandQueue> ocl_device_queues;
    //std::vector<MutexWrapper> ocl_device_queue_mutexs;
    
    std::vector<cl::Kernel> ocl_kernel_set;
    //Every compiled kernel is stored here.
    
    
    // read = CL_MEM_READ_ONLY
    // write = CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY
    std::vector<cl::Buffer> write_buffer_set;
    std::vector<unsigned int> write_buffer_set_sizes;
    std::vector<cl::Buffer> read_buffer_set;
    std::vector<unsigned int> read_buffer_set_sizes;
    
    bool ocl_profiling;
    
    std::string ocl_routine_name;
    //
    // Various
    //
    
    bool interpolation_complete;

};
 
#endif

//EOF