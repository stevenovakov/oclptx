/*  Copyright (C) 2014
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
#include <mutex>
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

#include "collatz_particle.h"
#include "customtypes.h"

class OclPtxHandler{
 public:
  struct particle_data
  {
    cl_ulong value;
  } __attribute__ ((aligned(8)));

  struct particle_attrs
  {
    cl_int num_steps;
    cl_int particles_per_side;
  } __attribute__ ((aligned(8)));

  OclPtxHandler() {};
  ~OclPtxHandler();
  void Init(
      cl::Context *cc,
      cl::CommandQueue *cq,
      cl::Kernel *ck,
      const BedpostXData *f,
      const BedpostXData *phi,
      const BedpostXData *theta,
      unsigned int num_directions,
      const unsigned short int *brain_mask,
      struct particle_attrs *attrs);

  // TODO(jeff) Kill these two after I check that I do everything.
  void ParticlePathsToFile();
  void Interpolate();

  // Run Kernel asyncronously
  void RunKernel(int side);
  // Write a single particle
  void WriteParticle(struct particle_data *data, int offset);
  // Read the "completion" buffer back into the vector pointed to by ret.
  void ReadStatus(int offset, int count, cl_ushort *ret);
  void DumpPath(int offset, int count, FILE *fd);

  int particles_per_side();
 private:
  // Init helpers
  void WriteSamplesToDevice( const BedpostXData* f_data,
                              const BedpostXData* phi_data,
                              const BedpostXData* theta_data,
                              unsigned int num_directions,
                              const unsigned short int* brain_mask);
  void InitParticles(struct particle_attrs *attrs);

  // OpenCL Interface
  cl::Context* context_;
  cl::CommandQueue* cq_;
  cl::Kernel* kernel_;

  // BedpostX Data
  cl::Buffer f_samples_buffer;
  cl::Buffer phi_samples_buffer;
  cl::Buffer theta_samples_buffer;
  cl::Buffer brain_mask_buffer;

  unsigned int samples_buffer_size;
  unsigned int sample_nx, sample_ny, sample_nz, sample_ns;

  // Output Data
  unsigned int n_particles;
  unsigned int max_steps;

  unsigned int section_size;
  unsigned int num_steps;
  unsigned int particles_size;

  cl::Buffer particle_paths_buffer;
  cl::Buffer particle_steps_taken_buffer;
  cl::Buffer particle_elem_buffer;
  cl::Buffer particle_done_buffer;
  
  unsigned int particles_mem_size;
  unsigned int particle_uint_mem_size;
  // size (Total Particles)/numDevices * (sizeof(float4))
  
  // Particle Data
  cl::Buffer *gpu_data;  // Type particle_data
  cl::Buffer *gpu_complete;  // Type cl_ushort array

  cl::Buffer *gpu_path;  // Type ulong

  struct particle_attrs attrs_;
};

#endif

