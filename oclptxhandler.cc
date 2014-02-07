/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */
 
/* oclptxhandler.cc
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

#include "oclptxhandler.h"

//
// Assorted Functions Declerations
//

std::string oclErrorStrings(cl_int error);

//*********************************************************************
//
// OclPtxHandler Constructors/Destructors
//
//*********************************************************************

//
// Constructor(s)
//
OclPtxHandler::OclPtxHandler(
  std::string ocl_routine
)
{  
  this->ocl_routine_name = ocl_routine;
  
  this->ocl_profiling = true;
  this->interpolation_complete = true;
  
  this->OclInit();
  this->OclDeviceInfo();
  std::cout<<"init\n";
  this->NewCLCommandQueues();
  std::cout<<"cq\n";
  this->CreateProgram();
  std::cout<<"cprog\n";
  
}

//
// Destructor
//
OclPtxHandler::~OclPtxHandler()
{
  std::cout<<"~OclPtxHandler\n";
  // no pointer data elements at the moment...
}

//*********************************************************************
//
// OclPtxHandler Container Set/Get
//
//*********************************************************************

void OclPtxHandler::SetOclRoutine(std::string new_routine)
{
  this->ocl_routine_name = new_routine;
}



//*********************************************************************
//
// OclPtxHandler Thread Management
//
//*********************************************************************

//
//
//
void OclPtxHandler::PTStart()
{
  this->interpolation_complete = false;
  
  
}


//
//
//
void OclPtxHandler::ThreadController()
{
  while(!(this->interpolation_complete))
  {
    
    //
    // One while/for loop for thread CREATION/QUEUEING/EXECUTION
    // 
    
    //
    // One while/for loop for thread collection/joining
    //
    
    //if(termination_condition)
    // this->interpolation_complete = true;
    
  }

}


//*********************************************************************
//
// OclPtxHandler OpenCL Interface
//
//*********************************************************************

//
//
//
void OclPtxHandler::OclInit()
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
  //this->oclContext = cl::Context(CL_DEVICE_TYPE_CPU, conProp);
  
  this->ocl_devices = this->ocl_context.getInfo<CL_CONTEXT_DEVICES>();

}

void OclPtxHandler::OclDeviceInfo()
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
//
//
//
void OclPtxHandler::NewCLCommandQueues()
{
  
  this->ocl_device_queues.clear();
  //this->ocl_device_queue_mutexs.clear();
  
  for(unsigned int k = 0; k < this->ocl_devices.size(); k++ )
  {

    std::cout<<"Create CommQueue, Kernel, Device: "<<k<<"\n";
    if(this->ocl_profiling)
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
cl::Program OclPtxHandler::CreateProgram()
{
  
  this->ocl_kernel_set.clear();  
  
  // Read Source
  
  std::string line_str, kernel_source;
  
  std::ifstream source_file;
  
  std::vector<std::string> source_list;
  std::vector<std::string>::iterator sit;
  
  if(this->ocl_routine_name == "oclptx")
  {
    source_list.push_back("prngmethods.cl");
    source_list.push_back("interpolate.cl");
  }
  else if( this->ocl_routine_name == "prngtest")
  {
    source_list.push_back("prngmethods.cl");
    source_list.push_back("prngtest.cl");    
  }
  else if(this->ocl_routine_name == "interptest")
  {
    //source_list.push_back("prngmethods.cl");
    source_list.push_back("interptest.cl");    
  }
  
  for(sit = source_list.begin(); sit != source_list.end(); ++sit)
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
    
  try{
    
    ocl_program.build(this->ocl_devices);
    
  }
  catch(cl::Error err){
    
    // TODO 
    //  dump all error logging to logfile
    //  maybe differentiate b/w regular errors and cl errors
    
    if( oclErrorStrings(err.err()) != "CL_SUCCESS")
    {
      std::cout<<"ERROR: " << err.what() <<
        " ( " << oclErrorStrings(err.err()) << ")\n";
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
  if( this->ocl_routine_name == "oclptx" )
  {
    this->ocl_kernel_set.push_back(cl::Kernel(  ocl_program, 
                                                "InterpolateKernel",
                                                NULL
                                              ));
  }
  else if( this->ocl_routine_name == "oclptx" )
  {
    this->ocl_kernel_set.push_back(cl::Kernel(  ocl_program, 
                                                "PrngTestKernel",
                                                NULL
                                              ));
  }
  else if( this->ocl_routine_name == "interptest" )
  {
    this->ocl_kernel_set.push_back(cl::Kernel(  ocl_program, 
                                                "InterpolateTestKernel",
                                                NULL
                                              ));
  }
  
  return ocl_program;
}


//
//
//
void OclPtxHandler::InstantiateBuffers()
{
  
  this->read_buffer_set.clear();
  this->write_buffer_set.clear();
  
  try{
    for( unsigned int i = 0;
      i < this->read_buffer_set_sizes.size(); i++)
    { 
      std::cout<<"RBUF: "<<this->read_buffer_set_sizes.at(i)<<"\n";
      this->read_buffer_set.push_back(
            cl::Buffer( this->ocl_context,
                        CL_MEM_READ_ONLY,
                        this->read_buffer_set_sizes.at(i),
                        NULL,
                        NULL                            
                      )
      );
    }
    for( unsigned int i = 0;
      i < this->write_buffer_set_sizes.size(); i++)
    { 
      std::cout<<"WBUF: "<<this->write_buffer_set_sizes.at(i)<<"\n"; 
      this->write_buffer_set.push_back(
        cl::Buffer( this->ocl_context,
                    CL_MEM_READ_WRITE,
                    this->write_buffer_set_sizes.at(i),
                    NULL,
                    NULL                            
                  )
      );      
    }  
  }
  catch(cl::Error err){
    
    // TODO 
    //  dump all error logging to logfile
    //  maybe differentiate b/w regular errors and cl errors
    
    if( oclErrorStrings(err.err()) != "CL_SUCCESS")
    {
      std::cout<<"ERROR: " << err.what() <<
        " ( " << oclErrorStrings(err.err()) << ")\n";
    }
  }
}

//*********************************************************************
//
// OclPtxHandler Tractography
//
//*********************************************************************


//
// Single Device Interpolation Test
//
std::vector<float4> OclPtxHandler::InterpolationTestRoutine
    ( 
      FloatVolume voxel_space,
      FloatVolume flow_space, 
      std::vector<float4> seed_space,
      std::vector<unsigned int> seed_elem,
      unsigned int n_seeds,
      unsigned int n_steps,
      float dr, 
      float4 min_bounds, 
      float4 max_bounds
    )
{
  
  bool blocking_write = false;

  this->write_buffer_set_sizes.clear();
  this->read_buffer_set_sizes.clear();
  
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

  //
  this->InstantiateBuffers();
  //

  //write to buffers on device

  //flow space
  this->ocl_device_queues.at(0).enqueueWriteBuffer(
        this->read_buffer_set.at(0),
        blocking_write,
        (unsigned int) 0,
        this->read_buffer_set_sizes.at(0),
        voxel_space.vol.data(),
        NULL,
        NULL
  );
  //flow space
  this->ocl_device_queues.at(0).enqueueWriteBuffer(
        this->read_buffer_set.at(1),
        blocking_write,
        (unsigned int) 0,
        this->read_buffer_set_sizes.at(1),
        flow_space.vol.data(),
        NULL,
        NULL
  ); 
  //seed initial pos
  this->ocl_device_queues.at(0).enqueueWriteBuffer(
        this->read_buffer_set.at(2),
        blocking_write,
        (unsigned int) 0,
        this->read_buffer_set_sizes.at(2),
        seed_space.data(),
        NULL,
        NULL
  ); 
  // seed elem
  this->ocl_device_queues.at(0).enqueueWriteBuffer(
        this->write_buffer_set.at(1),
        blocking_write,
        (unsigned int) 0,
        this->write_buffer_set_sizes.at(1),
        seed_elem.data(),
        NULL,
        NULL
  );
  
  cl::Kernel * test_kernel = &(this->ocl_kernel_set.at(0));
  // do NOT call delete on this.
  
  test_kernel->setArg(0, this->read_buffer_set.at(0)); //voxels
  test_kernel->setArg(1, this->read_buffer_set.at(1)); //flow
  test_kernel->setArg(2, this->read_buffer_set.at(2)); //seed pts
  test_kernel->setArg(3, this->write_buffer_set.at(1)); //seed elem
  test_kernel->setArg(4, voxel_space.nx); //nx
  test_kernel->setArg(5, voxel_space.ny); //ny
  test_kernel->setArg(6, voxel_space.nz); //nz
  test_kernel->setArg(7, min_bounds); //ny
  test_kernel->setArg(8, max_bounds); //nz
  test_kernel->setArg(9, n_steps); //n_steps
  test_kernel->setArg(10, this->write_buffer_set.at(0));//path container

  // OCL CQ BLOCK
  this->ocl_device_queues.at(0).finish();
  // OCL CQ BLOCK
  
  this->ocl_device_queues.at(0).enqueueNDRangeKernel(
    *test_kernel,
    cl::NullRange,
    test_global_range,
    test_local_range,
    NULL,
    NULL
  );

  this->ocl_device_queues.at(0).finish();
  
  //instantiate return container
  std::vector<float4> return_container(result_num);
  
  std::cout<<return_container.size()*sizeof(return_container.at(result_num))<<"\n";
  std::cout<<n_seeds*n_steps*sizeof(float4)<<"\n";
  std::cout<<result_mem_size<<"\n";

  //fill return container
  this->ocl_device_queues.at(0).enqueueReadBuffer(
    this->write_buffer_set.at(0),
    CL_TRUE, // blocking
    0,
    result_mem_size,
    &return_container[0]
  );
  
  return return_container;  
}


//*********************************************************************
//
// Assorted Functions
//
//*********************************************************************

//
// Matches OCL error codes to their meaning.
//
std::string oclErrorStrings(cl_int error)
{
  
  //
  // redeclaration on each call obviously not efficient, but had some 
  // problems putting this as a global variable, revisit later
  //
  std::string error_string[] = 
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
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };
    
    return error_string[ -1*error];
}


//EOF