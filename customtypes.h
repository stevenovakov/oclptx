/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */
 
/* customtypes.h
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
#include <vector>

 #ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif


#ifndef OCLPTX_CUSTOMTYPES_H_
#define OCLPTX_CUSTOMTYPES_H_


struct float3{
  float x, y, z;
};

struct float4{
  float x, y, z, t;
};

struct int3{
  float x, y, z;
};

struct int4{
  float x, y, z, t;
};


struct IntVolume
{
  std::vector<int4> vol;
  int nx, ny, nz;
  // Access x, y, z:
  //                var_name.vol[z*(ny*nx) + y*8*nx + 8*x + v]
};

//struct FloatVolume
//{
//  std::vector<float4> vol;
//  int nx, ny, nz;
//  // Access x, y, z:
//  //                var_name.vol[z*(ny*nx) + y*8*nx + 8*x + v]
//};

//struct MutexWrapper {
    //std::mutex m;
    //MutexWrapper() {}
    //MutexWrapper(MutexWrapper const&) {}
    //MutexWrapper& operator=(MutexWrapper const&) { return *this; }
//};

struct BedpostXData
{
  std::vector<float*> data;
  uint32_t nx, ny, nz;  // discrete dimensions of mesh
  uint32_t ns;          //number of samples
  // uint32_t num_dir;
  // cl::Buffer data_cl_buffer;
};
//
// Note on particle positions re:bedpostx mesh :
// if a particle is at x,y,z, can find nearest "root" vertex:
// (lowest x,y,z coordinate vertex), at X, Y, Z and find element #
// thusly:
//
// elem = nssample*(X*nz*ny + Y*nz + Z)
//
// Device side, where there may be multiple directions included, simply
// multiply by the direction #, (from 0 to n-1)
//

//TODO @STEVE
//
// Declare these all as const, and then have oclEnv initialize them
// Maybe this should be a class.
//
struct EnvironmentData
{
  //OpenCL Related
  cl::Context* ocl_context; //

  cl_ulong max_buffer_size; //
  cl_ulong global_mem_size; //

  //Input Data - Size Related
  uint32_t nx; //
  uint32_t ny; // also serves as mask dims
  uint32_t nz; //
  uint32_t ns; //

  uint32_t bpx_dirs; //
  uint32_t n_waypts;
  bool exclusion_mask;
  bool terminate_mask;
  bool prefdir;
  bool loopcheck;
  uint32_t loopcheck_location_size;
  uint32_t loopcheck_dir_size;
  uint32_t lx;
  uint32_t ly;
  uint32_t lz;

  // Interpolation Options
  uint32_t max_steps;
  bool save_paths;
  bool way_and;
  bool euler_streamline;
  bool deterministic;

  // Particle Containers
  uint32_t section_size;
  uint32_t pdf_entries_per_particle;
  uint32_t global_pdf_size;
  
  //values to use for computation, buffers
  uint32_t interval_size; //2R in Oclptx Data Diagram 2.odg

  cl_uint single_sample_mem_size;
  cl_uint particle_paths_mem_size;
  cl_uint particle_uint_mem_size;
  cl_uint particles_prng_mem_size;
  cl_uint mask_mem_size;
  cl_uint particle_pdf_mask_mem_size;
  cl_uint global_pdf_mem_size;
  cl_uint particle_loopcheck_location_mem_size;
  cl_uint particle_loopcheck_dir_mem_size;
  cl_long dynamic_mem_left;

  uint32_t total_static_gpu_mem;
  //uint32_t dynamic_gpu_mem_left;
  uint32_t max_particles_per_batch;

  //
  // These are allocated/deallocated by OclEnv
  //

  cl::Buffer** f_samples_buffers;
  cl::Buffer** phi_samples_buffers;
  cl::Buffer** theta_samples_buffers;
  cl::Buffer* brain_mask_buffer;

  cl::Buffer* waypoint_masks_buffer;
  cl::Buffer* exclusion_mask_buffer;
  cl::Buffer* termination_mask_buffer;
};

#endif
