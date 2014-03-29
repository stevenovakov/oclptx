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
#include <vector>
#include <cmath>
//#include <mutex>
//#include <thread>


//#define __CL_ENABLE_EXCEPTIONS
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
}

void OclEnv::Init()
{
  this->OclInit();
  this->OclDeviceInfo();
  this->NewCLCommandQueues();
  this->CreateKernels();

  this->env_data.f_samples_buffers = NULL;
  this->env_data.phi_samples_buffers = NULL;
  this->env_data.theta_samples_buffers = NULL;
  this->env_data.brain_mask_buffer = NULL;
  this->env_data.exclusion_mask_buffer = NULL;
  this->env_data.termination_mask_buffer = NULL;
  this->env_data.waypoint_masks = NULL;
}

//
// Destructor
//
OclEnv::~OclEnv()
{
  std::cout<<"~OclEnv\n";
  uint32_t n_dirs = this->env_data.bpx_dirs;

  if (this->env_data.f_samples_buffers != NULL)
  {
    for (uint32_t s = 0; s < n_dirs; s++)
    {
      delete this->env_data.f_samples_buffers[s];
      delete this->env_data.phi_samples_buffers[s];
      delete this->env_data.theta_samples_buffers[s];
    }
    
    delete[] this->env_data.f_samples_buffers;
    delete[] this->env_data.phi_samples_buffers;
    delete[] this->env_data.theta_samples_buffers;
  }

  delete this->env_data.brain_mask_buffer;
  
  if (this->env_data.exclusion_mask_buffer != NULL)
    delete this->env_data.exclusion_mask_buffer;
  if (this->env_data.termination_mask_buffer != NULL)
    delete this->env_data.termination_mask_buffer;
  if (this->env_data.waypoint_masks != NULL)
  {
    for (uint32_t w = 0; w < this->env_data.n_waypts; w ++)
      delete this->env_data.waypoint_masks[w];

    delete[] this->env_data.waypoint_masks;
  }
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

cl::Kernel * OclEnv::GetSumKernel(unsigned int kernel_num)
{
  return &(this->sum_kernel_set.at(kernel_num));
}

void OclEnv::SetOclRoutine(std::string new_routine)
{
  this->ocl_routine_name = new_routine;
  this->CreateKernels();
}

EnvironmentData * OclEnv::GetEnvData()
{
  return &(this->env_data);
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
void OclEnv::CreateKernels(
  bool two_dir,
  bool three_dir,
  bool waypoints,
  bool termination,
  bool exclusion,
  bool euler_stream
)
{
  this->ocl_kernel_set.clear();
  this->sum_kernel_set.clear();

  cl_int err;

  // Read Source
  std::string fold = "oclkernels";

  std::string interp_kernel_source;
  std::string sum_kernel_source = fold + slash + "summing.cl";
  std::string define_list =  "-I ./oclkernels -D PRNG";

  if (this->ocl_routine_name == "standard")
  {
    interp_kernel_source = fold + slash + "interpolate.cl";
  }
  else if (this->ocl_routine_name == "rng_test")
  {
    interp_kernel_source = fold + slash + "rng_test.cl";
  }
  else if (this->ocl_routine_name == "collatz")
  {
    interp_kernel_source = fold + slash + "collatz.cl";
  }
  else if (this->ocl_routine_name == "interptest")
  {
    //source_list.push_back("prngmethods.cl");
    interp_kernel_source = fold + slash + "interptest.cl";
  }
  else if (this->ocl_routine_name == "basic")
  {
    interp_kernel_source = fold + slash + "basic.cl";
  }

  if (two_dir)
    define_list += " -D TWO_DIR";
  if (three_dir)
    define_list += " -D THREE_DIR";
  if (waypoints)
    define_list += " -D WAYPOINTS";
  if (termination)
    define_list += " -D TERMINATION";
  if (exclusion)
    define_list += " -D EXCLUSION";
  if (euler_stream)
    define_list += " -D EULER_STREAMLINE";

  std::ifstream main_stream(interp_kernel_source);
  std::string main_code(  (std::istreambuf_iterator<char>(main_stream) ),
                            (std::istreambuf_iterator<char>()));

  std::ifstream sum_stream(sum_kernel_source);
  std::string sum_code(  (std::istreambuf_iterator<char>(sum_stream) ),
                            (std::istreambuf_iterator<char>()));
  //
  // Build Program files here
  //

  cl::Program::Sources main_source(
    1,
    std::make_pair(main_code.c_str(), main_code.length())
  );

  cl::Program::Sources sum_source(
    1,
    std::make_pair(sum_code.c_str(), sum_code.length())
  );

  cl::Program main_program(this->ocl_context, main_source);

  err = main_program.build(this->ocl_devices, define_list.c_str());

  if( this->OclErrorStrings(err) != "CL_SUCCESS")
  {
    std::cout<<"ERROR: " <<
      " ( " << this->OclErrorStrings(err) << ")\n";

    std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();

    std::cout<<"BUILD OPTIONS: \n" <<
      main_program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*dit) <<
       "\n";
    std::cout<<"BUILD LOG: \n" <<
      main_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*dit) <<"\n";

    exit(EXIT_FAILURE);
  }

  cl::Program sum_program(this->ocl_context, sum_source);

  err = sum_program.build(this->ocl_devices, "-I ./oclkernels");

  if( this->OclErrorStrings(err) != "CL_SUCCESS")
  {
    std::cout<<"ERROR: " <<
      " ( " << this->OclErrorStrings(err) << ")\n";

    std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();

    std::cout<<"BUILD OPTIONS: \n" <<
      sum_program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*dit) <<
       "\n";
    std::cout<<"BUILD LOG: \n" <<
      sum_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*dit) <<"\n";

    exit(EXIT_FAILURE);
  }
  

  //
  // Compile Kernels from Program
  //
  for( unsigned int k = 0; k < this->ocl_devices.size(); k++)
  {
    if (this->ocl_routine_name == "standard" )
    {
      this->ocl_kernel_set.push_back(cl::Kernel(main_program,
                                                "OclPtxInterpolate",
                                                NULL));
      this->sum_kernel_set.push_back(cl::Kernel(sum_program,
                                                "PdfSum",
                                                NULL));
    }
    else if (this->ocl_routine_name == "rng_test")
    {
      this->ocl_kernel_set.push_back(cl::Kernel(main_program,
                                                "RngTest",
                                                NULL));
    }
    else if (this->ocl_routine_name == "collatz")
    {
      this->ocl_kernel_set.push_back(cl::Kernel(main_program,
                                                "Collatz",
                                                NULL));
    }
    else if (this->ocl_routine_name == "interptest" )
    {
      this->ocl_kernel_set.push_back(cl::Kernel(main_program,
                                                "InterpolateTestKernel",
                                                NULL));
    }
    else if (this->ocl_routine_name == "basic" )
    {
      this->ocl_kernel_set.push_back(cl::Kernel(main_program,
                                                "BasicInterpolate",
                                                NULL));
    }
  }
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

//*********************************************************************
//
// Resource Allocation
//
//*********************************************************************

// TODO @STEVE
// Right now I've just kluged together AllocateSamples, but really
// AvailableGPUMem should run more thoroughly and calculate a lot of the vallues
// currently bneing calculated in AllocateSamples
//
int OclEnv::AvailableGPUMem(
  const BedpostXData* f_data,
  uint32_t n_bpx_dirs,
  uint32_t num_waypoint_masks,
  uint32_t loopcheck_fraction,
  bool exclusion_mask,
  bool termination_mask,
  uint32_t n_waypoints,
  bool save_particle_paths,
  uint32_t m_steps,
  float mem_fraction //also pass oclptxoptions here
)
{
  // ***********************************************
  //  Hardware Parameters
  // ***********************************************
  cl_ulong max_buff_size;
  cl_ulong gl_mem_size;
  cl_ulong useful_gl_mem_size;
  cl_ulong dynamic_mem_left;
  cl_ulong dynamic_mem_per_particle = 0;
  uint32_t r_particles;

  std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();

  dit->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_buff_size);
  this->env_data.max_buffer_size = max_buff_size;

  dit->getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &gl_mem_size);
  this->env_data.global_mem_size = gl_mem_size;

  useful_gl_mem_size = std::floor(gl_mem_size * mem_fraction);

  // ***********************************************
  //  BPX Sample Parameters + Masks
  // ***********************************************

  this->env_data.bpx_dirs = n_bpx_dirs;
  this->env_data.nx = f_data->nx;
  this->env_data.ny = f_data->ny;
  this->env_data.nz = f_data->nz;
  this->env_data.ns = f_data->ns;

  cl_uint single_direction_size =
    f_data->nx * f_data->ny * f_data->nz;

  cl_uint single_pdf_mask_size = (single_direction_size / 32)  +
    ((single_direction_size%32 > 0)? 1 : 0);

  cl_uint brain_mem_size =
    single_direction_size * sizeof(unsigned short int);

  this->env_data.mask_mem_size = brain_mem_size;

  cl_uint single_direction_mem_size =
    single_direction_size*f_data->ns*sizeof(float);

  this->env_data.single_sample_mem_size = single_direction_mem_size;

  cl_uint total_mem_size = 3*single_direction_mem_size * n_bpx_dirs +
    brain_mem_size*(1 + n_waypoints);

  if (exclusion_mask)
    total_mem_size += brain_mem_size;
  if (termination_mask)
    total_mem_size += brain_mem_size;

  this->env_data.global_pdf_size = single_pdf_mask_size * 32;
  this->env_data.global_pdf_mem_size =
    this->env_data.global_pdf_size * sizeof(uint32_t);

  this->env_data.total_static_gpu_mem =
    total_mem_size + this->env_data.global_pdf_mem_size;

  dynamic_mem_left = useful_gl_mem_size - this->env_data.total_static_gpu_mem;

  // ***********************************************
  //  Dynamic Particle Containers/Parameters
  // ***********************************************

  this->env_data.particle_pdf_mask_mem_size =
    single_pdf_mask_size * sizeof(uint32_t);
  this->env_data.pdf_entries_per_particle = single_pdf_mask_size;

  dynamic_mem_per_particle += this->env_data.particle_pdf_mask_mem_size;
  
  // Loopcheck
  uint32_t loopcheck_x = (f_data->nx / loopcheck_fraction) +
    ((f_data->nx %loopcheck_fraction > 0)? 1 : 0);
  uint32_t loopcheck_y = (f_data->ny / loopcheck_fraction) +
    ((f_data->ny %loopcheck_fraction > 0)? 1 : 0);
  uint32_t loopcheck_z = (f_data->nz / loopcheck_fraction) +
    ((f_data->nz %loopcheck_fraction > 0)? 1 : 0);

  cl_uint single_loopcheck_size = loopcheck_x * loopcheck_y * loopcheck_z;
  cl_uint single_loopcheck_mask_size = (single_loopcheck_size / 32)  +
    ((single_loopcheck_size%32 > 0)? 1 : 0);

  this->env_data.particle_loopcheck_mem_size = single_loopcheck_mask_size * 32;

  dynamic_mem_per_particle += this->env_data.particle_loopcheck_mem_size;

  // particle_done, steps_taken
  dynamic_mem_per_particle += 2 * sizeof(uint32_t);

  // rng

  dynamic_mem_per_particle += sizeof(cl_ulong8);

  // paths?
  this->env_data.max_steps = m_steps;
  this->env_data.save_paths = save_particle_paths;

  if (save_particle_paths)
  {
    dynamic_mem_per_particle += m_steps * sizeof(float4);
  }

  // Compute Max Particles per Batch:
  r_particles = (dynamic_mem_left/dynamic_mem_per_particle) /2;
  this->env_data.max_particles_per_batch = r_particles;

  printf("/**************************************************\n");
  printf("\tOCLENV::AVAILABLEGPUMEM\n");
  printf("/**************************************************\n\n");
  printf("Voxel Dims (x,y,z): %u, %u, %u\n", f_data->nx, f_data->ny, f_data->nz);
  printf("Num Samples: %u\n", f_data->ns);
  printf("Single Dir Sample Mem Size: %u (B), %.4f (MB), \n",
  single_direction_mem_size, single_direction_mem_size/1e6);
  printf("Num_directions: %u \n", n_bpx_dirs);
  printf("Total GPU Device Memory: %.4f (MB) \n", gl_mem_size/1e6);
  printf("Total USEFUL GPU Device Memory: %.4f (MB) \n", useful_gl_mem_size/1e6);
  printf("Total Static Data Memory Requirement: %.4f (MB) \n",
    this->env_data.total_static_gpu_mem/1e6);
  printf("Remaining GPU Device Memory: %.4f (MB) \n", dynamic_mem_left/1e6);
  printf("Dynamic GPU Memory per Particle: %.4f (kB)\n",
    dynamic_mem_per_particle/1e3);
  printf("Maximum Number of Particles per Batch: %u\n", r_particles);
  printf("Total Dynamic Data Memory Requirement: %.4f (MB)\n\n",
    2*r_particles*dynamic_mem_per_particle/1e6);

  // 100 is an arbitrary minimum, have user selectable minimum?
  if (r_particles > 100)
    return 1;
  else
    return -1;
}


void OclEnv::AllocateSamples(
  const BedpostXData* f_data,
  const BedpostXData* phi_data,
  const BedpostXData* theta_data,
  const unsigned short int* brain_mask,
  const unsigned short int* exclusion_mask,
  const unsigned short int* termination_mask,
  const unsigned short int** waypoint_masks
)
{
  uint32_t n_dirs = this->env_data.bpx_dirs;

  this->env_data.f_samples_buffers = new cl::Buffer*[n_dirs];
  this->env_data.phi_samples_buffers = new cl::Buffer*[n_dirs];
  this->env_data.theta_samples_buffers = new cl::Buffer*[n_dirs];

  for (uint32_t s = 0; s < n_dirs; s++)
  {
    this->env_data.f_samples_buffers[s] = new
      cl::Buffer(
        this->ocl_context,
        CL_MEM_READ_ONLY,
        this->env_data.single_sample_mem_size,
        NULL,
        NULL
      );

    this->env_data.theta_samples_buffers[s] = new
      cl::Buffer(
        this->ocl_context,
        CL_MEM_READ_ONLY,
        this->env_data.single_sample_mem_size,
        NULL,
        NULL
      );

    this->env_data.phi_samples_buffers[s] = new
      cl::Buffer(
        this->ocl_context,
        CL_MEM_READ_ONLY,
        this->env_data.single_sample_mem_size,
        NULL,
        NULL
      );
  }

  this->env_data.brain_mask_buffer = new
    cl::Buffer(
      this->ocl_context,
      CL_MEM_READ_ONLY,
      this->env_data.mask_mem_size,
      NULL,
      NULL
    );

    //
    // TODO @STEVE
    // I should instantiate/write globalpdf here, it has to go to each
    // device anyways...
    //
    for (uint32_t d = 0; d < this->ocl_devices.size(); d++)
    {
      for (uint32_t s = 0; s < n_dirs; s++)
      {
        this->ocl_device_queues.at(d).enqueueWriteBuffer(
          *(this->env_data.f_samples_buffers[s]),
          CL_FALSE,
          static_cast<unsigned int>(0),
          this->env_data.single_sample_mem_size,
          f_data->data.at(s),
          NULL,
          NULL
        );

        this->ocl_device_queues.at(d).enqueueWriteBuffer(
          *(this->env_data.theta_samples_buffers[s]),
          CL_FALSE,
          static_cast<unsigned int>(0),
          this->env_data.single_sample_mem_size,
          theta_data->data.at(s),
          NULL,
          NULL
        );

        this->ocl_device_queues.at(d).enqueueWriteBuffer(
          *(this->env_data.phi_samples_buffers[s]),
          CL_FALSE,
          static_cast<unsigned int>(0),
          this->env_data.single_sample_mem_size,
          phi_data->data.at(s),
          NULL,
          NULL
        );
      }

      this->ocl_device_queues.at(d).enqueueWriteBuffer(
        *(this->env_data.brain_mask_buffer),
        CL_FALSE,
        static_cast<unsigned int>(0),
        this->env_data.mask_mem_size,
        const_cast<unsigned short int*>(brain_mask),
        NULL,
        NULL
      );

      this->ocl_device_queues.at(d).flush();
    }

    // can maybe move this to oclptxhandler, for slight performance improvement
    for (uint32_t d = 0; d < this->ocl_devices.size(); d++)
    {
      this->ocl_device_queues.at(d).finish();
    }
}

// void OclEnv::ProcessOptions( oclptxOptions* options)
// {

// }

//EOF
