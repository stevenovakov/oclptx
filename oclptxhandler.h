/* Copyright 2014
 *  Afshin Haidari
 *  Steve Novakov
 *  Jeff Taylor
 */

#ifndef OCLPTXHANDLER_H_
#define OCLPTXHANDLER_H_

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "customtypes.h"

class OclPtxHandler{
 public:
  struct particle_data
  {
    cl_ulong8 rng;
    cl_float4 position;
  } __attribute__((aligned(64)));

  struct particle_attrs
  {
    cl_int steps_per_kernel;
    cl_int max_steps;
    cl_int particles_per_side;
    cl_uint sample_nx;
    cl_uint sample_ny;
    cl_uint sample_nz;
    cl_uint num_samples;
    cl_float curvature_threshold;
    cl_uint n_waypoint_masks;
    cl_float step_length;
    cl_uint lx;  // Loopcheck sizes
    cl_uint ly;
    cl_uint lz;
  } __attribute__((aligned(8)));

  OclPtxHandler() {}
  void Init(
      cl::Context *cc,
      cl::CommandQueue *cq,
      cl::Kernel* ptx_kernel,
      cl::Kernel* sum_kernel,
      struct particle_attrs *attrs,
      FILE *path_dump_fd,
      EnvironmentData *env_dat,
      cl::Buffer *global_pdf);
  ~OclPtxHandler();

  int particles_per_side();

  // Write a single particle
  void WriteParticle(struct particle_data *data, int offset);
  // Run Kernel asyncronously
  void RunKernel(int side);
  // Read the "completion" buffer back into the vector pointed to by ret.
  void ReadStatus(int offset, int count, cl_ushort *ret);
  // Dump path to file.
  void DumpPath(int offset, int count);

 private:
  void InitParticles(struct particle_attrs *attrs);
  void GetPdfData(cl_int* container);
  void PdfSum();
  void SetKArg(int pos, cl::Buffer *buf);

  struct particle_attrs attrs_;

  // OpenCL Interface
  cl::Context* context_;
  cl::CommandQueue* cq_;
  cl::Kernel* ptx_kernel_;
  cl::Kernel* sum_kernel_;

  // Particle Data
  cl::Buffer *gpu_data_;  // Type particle_data
  cl::Buffer *gpu_complete_;  // Type ushort array
  cl::Buffer *gpu_local_pdf_;
  cl::Buffer *gpu_waypoints_;
  cl::Buffer *gpu_exclusion_;
  cl::Buffer *gpu_loopcheck_;
  cl::Buffer *gpu_global_pdf_;

  // Debug Data
  cl::Buffer *gpu_path_;  // Type ulong
  cl::Buffer *gpu_step_count_; // Type ushort

  FILE *path_dump_fd_;
  bool first_time_;

  // TODO: Steve's
  EnvironmentData * env_dat_;

  cl_ulong GpuMemUsed();
  cl_ulong total_gpu_mem_used;
};

#endif  // OCLPTXHANDLER_H_

