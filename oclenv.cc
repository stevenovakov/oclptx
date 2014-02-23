/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

/* oclenv.cc
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
#include <fstream>
#include <sstream>
#include <vector>
//#include <mutex>
//#include <thread>


#define __CL_ENABLE_EXCEPTIONS
// adds exception support from CL libraries
// define before CL headers inclusion

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "oclenv.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
static const std::string slash="\\";
#else
static const std::string slash="/";
#endif

//*********************************************************************
//
// OclEnv Constructors/Destructors
//
//*********************************************************************
//
// Constructor(s)
//
OclEnv::OclEnv(
  std::string ocl_routine
)
{
  this->ocl_routine_name = ocl_routine;
  this->ocl_profiling = false;

  this->OclInit();
  this->OclDeviceInfo();
  this->NewCLCommandQueues();
  this->CreateProgram();
}

//
// Destructor
//
OclEnv::~OclEnv()
{
  std::cout<<"~OclPtxHandler\n";
  // no pointer data elements at the moment...
}

//*********************************************************************
//
// OclEnv Container Set/Get
//
//*********************************************************************

cl::Context * OclEnv::GetContext()
{
  return &(this->ocl_context);
}

cl::CommandQueue * OclEnv::GetCq(unsigned int device_num)
{
  return &(this->ocl_device_queues.at(device_num));
}

cl::Kernel * OclEnv::GetKernel(unsigned int kernel_num)
{
  return &(this->ocl_kernel_set.at(kernel_num));
}

void OclEnv::SetOclRoutine(std::string new_routine)
{
  this->ocl_routine_name = new_routine;
  this->CreateProgram();
}


//*********************************************************************
//
// OclEnv OpenCL Interface
//
//*********************************************************************

//
// Currently ignores all other devices that arent GPU.
//
//
void OclEnv::OclInit()
{
  cl::Platform::get(&(this->ocl_platforms));

  cl_context_properties con_prop[3] =
  {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) (this->ocl_platforms[0]) (),
    0
  };

  this->ocl_platforms.at(0);

  this->ocl_context = cl::Context(CL_DEVICE_TYPE_GPU, con_prop);
  // GPU DEVICES ONLY, FOR CPU, (don't use CPU unless informed, not
  // quite the same physical interface):
  // this->oclContext = cl::Context(CL_DEVICE_TYPE_CPU, conProp);

  this->ocl_devices = this->ocl_context.getInfo<CL_CONTEXT_DEVICES>();
}

void OclEnv::OclDeviceInfo()
{
  std::cout<<"\nLocal OpenCL Devices\n\n";

  size_t siT[3];
  cl_uint print_int;
  cl_ulong print_ulong;
  std::string print_string;

  std::string device_name;

  for(std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();
    dit != this->ocl_devices.end(); ++dit){

    dit->getInfo(CL_DEVICE_NAME, &print_string);
    dit->getInfo(CL_DEVICE_NAME, &device_name);
    dit->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &print_int);
    dit->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &siT);
    dit->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &siT);


    std::cout<<"\tDEVICE\n\n";
    std::cout<<"\tDevice Name: " << print_string << "\n";
    std::cout<<"\tMax Compute Units: " << print_int << "\n";
    std::cout<<"\tMax Work Group Size (x*y*z): " << siT[0] << "\n";
    std::cout<<"\tMax Work Item Sizes (x, y, z): " << siT[0] <<
      ", " << siT[1] << ", " << siT[2] << "\n";


    dit->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &print_ulong);
    std::cout<<"\tMax Mem Alloc Size: " << print_ulong << "\n";

    std::cout<<"\n";
  }
}

unsigned int OclEnv::HowManyDevices()
{
  return this->ocl_devices.size();
}
//
//
//
void OclEnv::NewCLCommandQueues()
{
  this->ocl_device_queues.clear();
  //this->ocl_device_queue_mutexs.clear();

  for (unsigned int k = 0; k < this->ocl_devices.size(); k++ )
  {
    std::cout<<"Create CommQueue, Kernel, Device: "<<k<<"\n";

    if (this->ocl_profiling)
    {
      this->ocl_device_queues.push_back(  cl::CommandQueue(
                                            this->ocl_context,
                                            this->ocl_devices[k],
                                            CL_QUEUE_PROFILING_ENABLE
                                          )
                                        );
    }
    else
    {
      this->ocl_device_queues.push_back(  cl::CommandQueue(
                                            this->ocl_context,
                                            this->ocl_devices[k]
                                          )
                                        );
    }
    //this->ocl_device_queue_mutexs.push_back(MutexWrapper());
  }
}


//
//
//
cl::Program OclEnv::CreateProgram()
{
  this->ocl_kernel_set.clear();

  // Read Source

  std::string line_str, kernel_source;

  std::ifstream source_file;

  std::vector<std::string> source_list;
  std::vector<std::string>::iterator sit;
  
  std::string fold = "oclkernels";

  if (this->ocl_routine_name == "oclptx")
  {
    source_list.push_back(fold + slash + "prngmethods.cl");
    source_list.push_back(fold + slash + "interpolate.cl");
  }
  else if (this->ocl_routine_name == "prngtest")
  {
    source_list.push_back(fold + slash + "prngmethods.cl");
    source_list.push_back(fold + slash + "prngtest.cl");
  }
  else if (this->ocl_routine_name == "interptest")
  {
    //source_list.push_back("prngmethods.cl");
    source_list.push_back(fold + slash + "interptest.cl");
  }
  else if (this->ocl_routine_name == "basic")
  {
    source_list.push_back(fold + slash + "basic.cl");
  }
  for (sit = source_list.begin(); sit != source_list.end(); ++sit)
  {
    line_str = *sit;
    source_file.open(line_str.c_str());

    std::getline(source_file, line_str);

    while(source_file){

      kernel_source += line_str + "\n";

      std::getline(source_file, line_str);
    }
    source_file.close();
  }
  //TODO
  // no else  because we will check for routine_name at init
  //

  //
  // Build Program files here
  //

  //std::cout<<kernel_source;


  cl::Program::Sources prog_source(
    1,
    std::make_pair(kernel_source.c_str(), kernel_source.length())
  );

  cl::Program ocl_program(this->ocl_context, prog_source);

  try
  {
    ocl_program.build(this->ocl_devices);
  }
  catch(cl::Error err){

    // TODO
    //  dump all error logging to logfile
    //  maybe differentiate b/w regular errors and cl errors

    if( this->OclErrorStrings(err.err()) != "CL_SUCCESS")
    {
      std::cout<<"ERROR: " << err.what() <<
        " ( " << this->OclErrorStrings(err.err()) << ")\n";
      std::cin.get();

      for(std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();
        dit != this->ocl_devices.end(); dit++)
      {

        std::cout<<"BUILD OPTIONS: \n" <<
          ocl_program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*dit) <<
           "\n";
        std::cout<<"BUILD LOG: \n" <<
          ocl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*dit) <<"\n";
      }
    }
  }

  //
  // Compile Kernels from Program
  //
  for( unsigned int k = 0; k < this->ocl_devices.size(); k++)
  {
    if (this->ocl_routine_name == "oclptx" )
    {
      this->ocl_kernel_set.push_back(cl::Kernel(ocl_program,
                                                "OclPtxKernel",
                                                NULL ));
    }
    else if (this->ocl_routine_name == "prngtest" )
    {
      this->ocl_kernel_set.push_back(cl::Kernel(ocl_program,
                                                "PrngTestKernel",
                                                NULL));
    }
    else if (this->ocl_routine_name == "interptest" )
    {
      this->ocl_kernel_set.push_back(cl::Kernel(ocl_program,
                                                "InterpolateTestKernel",
                                                NULL));
    }
    else if (this->ocl_routine_name == "basic" )
    {
      this->ocl_kernel_set.push_back(cl::Kernel(ocl_program,
                                                "BasicInterpolate",
                                                NULL));
    }
  }

  return ocl_program;
}


//
// Matches OCL error codes to their meaning.
//
std::string OclEnv::OclErrorStrings(cl_int error)
{
  const std::string cl_error_string[] =
  {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE"
  };

  return cl_error_string[ -1*error];
}


//EOF