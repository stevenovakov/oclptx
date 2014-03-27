/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

/* oclenv.h
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

#ifndef  OCLPTX_OCLENV_H_
#define  OCLPTX_OCLENV_H_

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

class OclEnv{

  public:

    OclEnv(){};

    OclEnv(std::string ocl_routine);

    ~OclEnv();

    //
    // Container Set/Get
    //

    cl::Context * GetContext();
    
    cl::Device * GetDevice(unsigned int device_num);
    unsigned int HowManyDevices();
    
    cl::CommandQueue * GetCq(unsigned int device_num);
    cl::Kernel * GetKernel(unsigned int kernel_num);
    cl::Kernel * GetSumKernel(unsigned int kernel_num);

    EnvironmentData * GetEnvData();
    // TODO:
    // not sure if better to generate new cl::kernel  object for
    // every oclptxhandler object, or if can just point to this->kernels

    void SetOclRoutine(std::string new_routine);

    //
    // OpenCL API Interface/Helper Functions
    //

    void OclInit();

    void OclDeviceInfo();

    void NewCLCommandQueues();

    void CreateKernels(
      bool two_dir = false,
      bool three_dir = false,
      bool waypoints = false,
      bool termination = false,
      bool exclusion = false,
      bool euler_stream = false
    );

    std::string OclErrorStrings(cl_int error);

    //
    // Resource Allocation
    //

    int AvailableGPUMem(
      const BedpostXData* f_data,
      uint32_t n_bpx_dirs,
      uint32_t num_waypoint_masks,
      uint32_t loopcheck_fraction,
      bool exclusion_mask,
      bool termination_mask,
      uint32_t n_waypoints,
      bool save_particle_paths,
      uint32_t max_steps,
      float mem_fraction //also pass oclptxoptions here
    );

    void AllocateSamples(
      const BedpostXData* f_data,
      const BedpostXData* phi_data,
      const BedpostXData* theta_data,
      const unsigned short int* brain_mask,
      const unsigned short int* exclusion_mask,
      const unsigned short int* termination_mask,
      const unsigned short int** waypoint_masks
    );

    //void ProcessOptions( oclptxOptions* options);

  private:
    //
    // OpenCL Objects
    //
    cl::Context ocl_context;

    std::vector<cl::Platform> ocl_platforms;

    std::vector<cl::Device> ocl_devices;
    
    std::vector<cl::CommandQueue> ocl_device_queues;
    //std::vector<MutexWrapper> ocl_device_queue_mutexs;

    std::vector<cl::Kernel> ocl_kernel_set;
    std::vector<cl::Kernel> sum_kernel_set;
    //Every compiled kernel is stored here.

    std::string ocl_routine_name;

    bool ocl_profiling;

    EnvironmentData env_data;
};

#endif

//EOF