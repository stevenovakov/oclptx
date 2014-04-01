/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

#include "oclptxhandler.h"

#include <cstdlib>

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

void OclPtxHandler::Init(
  cl::Context *cc,
  cl::CommandQueue *cq,
  cl::Kernel *ptx_kernel,
  cl::Kernel *sum_kernel,
  struct OclPtxHandler::particle_attrs *attrs,
  FILE *path_dump_fd,
  EnvironmentData *env_dat)
{
  context_ = cc;
  cq_ = cq;
  ptx_kernel_ = ptx_kernel;
  sum_kernel_ = sum_kernel;
  first_time_ = 1;
  path_dump_fd_ = path_dump_fd;
  env_dat_ = env_dat;

  InitParticles(attrs);
}

void OclPtxHandler::InitParticles(struct OclPtxHandler::particle_attrs *attrs)
{
  attrs_ = *attrs;

  // TODO(jeff) compute num_particles
  attrs_.particles_per_side = 2;

  gpu_data_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * sizeof(struct particle_data));
  if (!gpu_data_)
    abort();

  gpu_complete_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * sizeof(cl_ushort));
  if (!gpu_complete_)
    abort();

  gpu_path_ = new cl::Buffer(
      *context_,
      CL_MEM_WRITE_ONLY,
      2 * attrs_.particles_per_side * attrs_.steps_per_kernel * sizeof(cl_float4));
  if (!gpu_path_)
    abort();

  gpu_step_count_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * sizeof(cl_ushort));
  if (!gpu_step_count_)
    abort();

  int entries = (attrs_.sample_nx
              * attrs_.sample_ny
              * attrs_.sample_nz);

  gpu_local_pdf_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * ((entries / 32)+1));
  if (!gpu_local_pdf_)
    abort();

  // TODO(jeff): Check whether these are actually needed, otherwise, NULL
  gpu_waypoints_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * attrs_.n_waypoint_masks);
  if (!gpu_waypoints_)
    abort();

  gpu_exclusion_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side);
  if (!gpu_exclusion_)
    abort();

  gpu_global_pdf_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * entries);
  if (!gpu_global_pdf_)
    abort();

  // Initialize "completion" buffer.
  cl_ushort *temp_completion = new cl_ushort[2*attrs_.particles_per_side];
  for (int i = 0; i < 2 * attrs_.particles_per_side; ++i)
    temp_completion[i] = 1;

  cq_->enqueueWriteBuffer(
      *gpu_complete_,
      true,
      0,
      2 * attrs_.particles_per_side * sizeof(cl_ushort),
      reinterpret_cast<void*>(temp_completion));

  delete[] temp_completion;

  int global_entries = attrs_.sample_nx
                     * attrs_.sample_ny
                     * attrs_.sample_nz;

  cl_ushort *temp_global_pdf = new cl_ushort[2 * global_entries];
  for (int i = 0; i < 2 * global_entries; ++i)
    temp_global_pdf[i] = 0;

  // This one is *not* a double buffer, nor related to number of particles.
  cq_->enqueueWriteBuffer(
      *gpu_global_pdf_,
      true,
      0,
      global_entries * sizeof(cl_uint),
      reinterpret_cast<void*>(temp_global_pdf));

  delete[] temp_global_pdf;
}

OclPtxHandler::~OclPtxHandler()
{
  delete gpu_path_;
  delete gpu_complete_;
  delete gpu_data_;
  delete gpu_step_count_;
  delete gpu_local_pdf_;
  delete gpu_global_pdf_;
}

int OclPtxHandler::particles_per_side()
{
  return attrs_.particles_per_side;
}

void OclPtxHandler::WriteParticle(
    struct particle_data *data,
    int offset)
{
  // Note: locking.  This function is technically thread-unsafe, but that
  // shouldn't matter because threading is set up for only one thread to ever
  // call these methods.
  cl_int ret;
  cl_ushort zero = 0;
  assert(offset < 2 * attrs_.particles_per_side);

  if (NULL != path_dump_fd_)
    fprintf(path_dump_fd_, "%i:%f,%f,%fn\n",
        offset,
        data->position.s[0],
        data->position.s[1],
        data->position.s[2]);

  // Write particle_data
  ret = cq_->enqueueWriteBuffer(
      *gpu_data_,
      true,
      offset * sizeof(struct particle_data),
      sizeof(struct particle_data),
      reinterpret_cast<void*>(data));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    abort();
  }

  // gpu_complete_ = 0
  ret = cq_->enqueueWriteBuffer(
      *gpu_complete_,
      true,
      offset * sizeof(cl_ushort),
      sizeof(cl_ushort),
      reinterpret_cast<void*>(&zero));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    abort();
  }

  // step_count = 0
  ret = cq_->enqueueWriteBuffer(
      *gpu_step_count_,
      true,
      offset * sizeof(cl_ushort),
      sizeof(cl_ushort),
      reinterpret_cast<void*>(&zero));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    abort();
  }

  // Fill in the "particle pdf" buffer.
  // TODO(jeff): this is going to be horrendously slow.  This HAS to be done in
  // the summing kernel.
  int entries_per_particle = (attrs_.sample_nx
                            * attrs_.sample_ny
                            * attrs_.sample_nz / 32) + 1;
  cl_ushort *temp_local_pdf = new cl_ushort[entries_per_particle];
  for (int i = 0; i < entries_per_particle; ++i)
    temp_local_pdf[i] = 0;

  cq_->enqueueWriteBuffer(
      *gpu_local_pdf_,
      true,
      offset * entries_per_particle * sizeof(cl_uint),
      entries_per_particle * sizeof(cl_uint),
      temp_local_pdf);

  delete[] temp_local_pdf;

  //TODO(jeff): Can we allocate in Init?
  //TODO(jeff): Don't write if these are not used.
  cl_ushort *temp_waypoints = new cl_ushort[attrs_.n_waypoint_masks];
  for (int i = 0; i < attrs_.n_waypoint_masks; ++i)
    temp_waypoints[i] = 0;

  cq_->enqueueWriteBuffer(
      *gpu_waypoints_,
      true,
      offset * attrs_.n_waypoint_masks * sizeof(cl_ushort),
      attrs.n_waypoint_masks * sizeof(cl_ushort),
      temp_waypoints);

  delete[] temp_waypoints;

  cl_ushort temp_zero = 0;

  cq_->enqueueWriteBuffer(
    *gpu_exclusion_,
    true,
    offset * sizeof(cl_ushort),
    sizeof(cl_ushort));
}

void OclPtxHandler::RunKernel(int side)
{
  cl::NDRange particles_to_compute(attrs_.particles_per_side);
  cl::NDRange particle_offset(attrs_.particles_per_side * side);
  cl::NDRange local_range(1);

  ptx_kernel_->setArg(
      0,
      sizeof(struct OclPtxHandler::particle_attrs),
      reinterpret_cast<void*>(&attrs_));
  ptx_kernel_->setArg(1, *gpu_data_);
  ptx_kernel_->setArg(2, *gpu_path_);
  ptx_kernel_->setArg(3, *gpu_step_count_);
  ptx_kernel_->setArg(4, *gpu_complete_);
  ptx_kernel_->setArg(5, *gpu_local_pdf_);
  ptx_kernel_->setArg(6, *gpu_waypoints_);
  ptx_kernel_->setArg(7, *gpu_exclusion_);

  ptx_kernel_->setArg(8, *env_dat_->f_samples_buffers[0]);
  ptx_kernel_->setArg(9, *env_dat_->phi_samples_buffers[0]);
  ptx_kernel_->setArg(10, *env_dat_->theta_samples_buffers[0]);
  ptx_kernel_->setArg(11, *env_dat_->brain_mask_buffer);
  ptx_kernel_->setArg(12, *env_dat_->waypoint_masks_buffer);
  ptx_kernel_->setArg(13, *env_dat_->termination_mask_buffer);
  ptx_kernel_->setArg(14, *env_dat_->exclusion_mask_buffer);

  cq_->enqueueNDRangeKernel(
    *(ptx_kernel_),
    particle_offset,
    particles_to_compute,
    local_range,
    NULL,
    NULL);

  cq_->finish();
}

void OclPtxHandler::ReadStatus(int offset, int count, cl_ushort *ret)
{
  cq_->enqueueReadBuffer(
      *gpu_complete_,
      true,
      offset * sizeof(cl_ushort),
      count * sizeof(cl_ushort),
      reinterpret_cast<cl_ushort*>(ret));
}

void OclPtxHandler::DumpPath(int offset, int count)
{
  cl_float4 *path_buf = new cl_float4[count * attrs_.steps_per_kernel];
  cl_ushort *step_count_buf = new cl_ushort[count];
  int ret;
  cl_float4 value;

  assert(NULL != path_dump_fd_);

  // kludge(jeff): The first time this is called by threading::Worker, there
  // is only garbage on the GPU, which we'd like to avoid dumping to file---
  // as it makes automatic verification more challenging.  My 5-second kludge
  // is to see if this is the first time we've been called.  If so, return
  // without printing anything.
  if (first_time_)
  {
    first_time_ = 0;
    return;
  }

  ret = cq_->enqueueReadBuffer(
      *gpu_path_,
      true,
      offset * attrs_.steps_per_kernel * sizeof(cl_float4),
      count * attrs_.steps_per_kernel * sizeof(cl_float4),
      reinterpret_cast<void*>(path_buf));
  if (CL_SUCCESS != ret)
  {
    puts("Failed to read back path");
    abort();
  }

  ret = cq_->enqueueReadBuffer(
      *gpu_step_count_,
      true,
      offset * sizeof(cl_ushort),
      count * sizeof(cl_ushort),
      reinterpret_cast<void*>(step_count_buf));
  if (CL_SUCCESS != ret)
  {
    puts("Failed to read back path");
    abort();
  }

  // Now dumpify.
  for (int id = 0; id < count; ++id)
  {
    for (int step = 0; step < attrs_.steps_per_kernel; ++step)
    {
      value = path_buf[id * attrs_.steps_per_kernel + step];
      // Only dump if this element is before the path's end.
      if ((0 == step_count_buf[id] % attrs_.steps_per_kernel
        && 0 != step_count_buf[id])
        || step < step_count_buf[id] % attrs_.steps_per_kernel)
        fprintf(path_dump_fd_, "%i:%f,%f,%f\n",
            id + offset,
            value.s[0],
            value.s[1],
            value.s[2]);
    }
  }

  delete path_buf;
  delete step_count_buf;
}

// TODO(jeff) Double buffer!  Add offsets to the two "particle" parameters.
// May need to be done in kernel itself.
void OclPtxHandler::PdfSum()
{
  cl::NDRange global_range(this->env_dat_->pdf_entries_per_particle);
  cl::NDRange local_range(1);

  sum_kernel_->setArg(0, gpu_global_pdf_);
  sum_kernel_->setArg(1, gpu_local_pdf_);  // TODO: Add offset
  sum_kernel_->setArg(2, gpu_complete_);  // TODO: Add offset
  sum_kernel_->setArg(3, attrs_.particles_per_side);
  sum_kernel_->setArg(4, this->env_dat_->pdf_entries_per_particle);

  cq_->enqueueNDRangeKernel(
    *(sum_kernel_),
    cl::NullRange,
    global_range,
    local_range,
    NULL,
    NULL
  );

  cq_->finish();
}

// TODO(jeff): needs work.  Offset, count would be nice, even if only used to
// assert.
void OclPtxHandler::GetPdfData(cl_int* container)
{
  cq_->enqueueReadBuffer(
    *gpu_global_pdf_,
    CL_FALSE,
    0,
    this->env_dat_->global_pdf_mem_size,
    container
  );
  cq_->finish();
}

