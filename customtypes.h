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

struct FloatVolume
{
  std::vector<float4> vol;
  int nx, ny, nz;
  // Access x, y, z:
  //                var_name.vol[z*(ny*nx) + y*8*nx + 8*x + v]
};

//struct MutexWrapper {
    //std::mutex m;
    //MutexWrapper() {}
    //MutexWrapper(MutexWrapper const&) {}
    //MutexWrapper& operator=(MutexWrapper const&) { return *this; }
//};

struct BedpostXData
{
  std::vector<float*> data;
  unsigned int nx, ny, nz;  // discrete dimensions of mesh
  unsigned int ns;          //number of samples
  // unsigned int num_dir;
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
  cl::Context* ocl_context;

  cl_ulong max_buffer_size;
  cl_ulong global_mem_size;

  float mem_safety_fraction;

  cl_ulong total_gpu_mem_used;
  // maximum fraction of global_mem_size to be used
  // in an attempt to stave off fragmentation issues.

  //Input Data - Size Related
  unsigned int nx; //
  unsigned int ny; // also serves as mask dims
  unsigned int nz; //
  unsigned int ns;

  unsigned int bpx_dirs;
  unsigned int n_waypts;
  bool exclusion_mask;
  bool terminate_mask;
  bool prefdir;
  unsigned int loopcheck_fraction;
  unsigned int max_steps;
  bool save_paths;
  bool euler_streamline;

  unsigned int section_size;
  unsigned int pdf_entries_per_particle;

  //values to use for computation, buffers
  unsigned int interval_size; //2R in Oclptx Data Diagram 2.odg

  cl_uint samples_buffer_size;
  cl_uint particle_paths_mem_size;
  cl_uint particle_uint_mem_size;
  cl_uint particles_prng_mem_size;
  cl_uint mask_mem_size;
  cl_uint particle_pdf_mask_size;
  cl_uint global_pdf_mem_size;

  unsigned int total_static_gpu_mem;
  unsigned int dynamic_gpu_mem_left;

  //
  // TODO (these probably should not be here)
  // Sample Data Buffers (figure out how to delete properly later
  // may need to make this a class)
  //

  cl::Buffer* f_samples_buffer;
  cl::Buffer* phi_samples_buffer;
  cl::Buffer* theta_samples_buffer;
  cl::Buffer* brain_mask_buffer;

  std::vector<cl::Buffer*> waypoint_masks;
  cl::Buffer* exclusion_mask_buffer;
  cl::Buffer* termination_mask_buffer;
};

#endif