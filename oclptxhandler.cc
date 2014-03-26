/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

#include "oclptxhandler.h"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

void OclPtxHandler::Init(
  cl::Context *cc,
  cl::CommandQueue *cq,
  cl::Kernel *ck,
  const BedpostXData *f,
  const BedpostXData *phi,
  const BedpostXData *theta,
  int num_directions,
  const unsigned short int *brain_mask,
  struct OclPtxHandler::particle_attrs *attrs)
{
  context_ = cc;
  cq_ = cq;
  kernel_ = ck;
  first_time_ = 1;

  // TODO(jeff): bring this line back to run PTX.
  // WriteSamplesToDevice(f, phi, theta, num_directions, brain_mask);
  InitParticles(attrs);
}

void OclPtxHandler::InitParticles(struct OclPtxHandler::particle_attrs *attrs)
{
  attrs_ = *attrs;

  // TODO(jeff) compute num_particles
  attrs_.particles_per_side = 2;

  gpu_data = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * sizeof(struct particle_data));
  if (!gpu_data)
    abort();

  gpu_complete = new cl::Buffer(
      *context_,
      CL_MEM_WRITE_ONLY,
      2 * attrs_.particles_per_side * sizeof(cl_ushort));
  if (!gpu_complete)
    abort();

  gpu_path = new cl::Buffer(
      *context_,
      CL_MEM_WRITE_ONLY,
      2 * attrs_.particles_per_side * attrs_.num_steps * sizeof(cl_ulong));
  if (!gpu_path)
    abort();

  gpu_step_count = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * sizeof(cl_ushort));
  if (!gpu_step_count)
    abort();

  // Initialize "completion" buffer.
  cl_ushort *temp_completion = new cl_ushort[2*attrs_.particles_per_side];
  for (int i = 0; i < 2 * attrs_.particles_per_side; ++i)
    temp_completion[i] = 1;

  cq_->enqueueWriteBuffer(
      *gpu_complete,
      true,
      0,
      2 * attrs_.particles_per_side * sizeof(cl_ushort),
      reinterpret_cast<void*>(temp_completion));

  delete[] temp_completion;
}

OclPtxHandler::~OclPtxHandler()
{
  delete gpu_path;
  delete gpu_complete;
  delete gpu_data;
  delete gpu_step_count;
}

void OclPtxHandler::ParticlePathsToFile()
{
  float4 * particle_paths;
  particle_paths = new float4[this->n_particles*this->max_steps];
  unsigned int * particle_steps;
  particle_steps = new unsigned int[this->n_particles];

  cq_->enqueueReadBuffer(
    this->particle_paths_buffer,
    CL_FALSE,  // blocking
    0,
    this->particles_mem_size,
    particle_paths);
  cq_->enqueueReadBuffer(
    this->particle_steps_taken_buffer,
    CL_FALSE,  // blocking
    0,
    this->particle_uint_mem_size,
    particle_steps);
  cq_->finish();

  // now dump to file

  std::ostringstream convert(std::ostringstream::ate);

  std::string path_filename;

  std::vector<float> temp_x;
  std::vector<float> temp_y;
  std::vector<float> temp_z;

  time_t t = time(0);
  struct tm * now = localtime(&t);

  convert << "OclPtx Results/"<< now->tm_yday << "-" <<
    static_cast<int>(now->tm_year) + 1900 << "_"<< now->tm_hour <<
      ":" << now->tm_min << ":" << now->tm_sec;

  path_filename = convert.str() + "_PATHS.dat";
  std::cout << "Writing to " << path_filename << "\n";

  std::fstream path_file;
  path_file.open(path_filename.c_str(), std::ios::app|std::ios::out);

  for (unsigned int n = 0; n < this->n_particles; n ++)
  {
    unsigned int p_steps = particle_steps[n];

    for (unsigned int s = 0; s < p_steps; s++)
    {
      temp_x.push_back(particle_paths[n*this->max_steps + s].x);
      temp_y.push_back(particle_paths[n*this->max_steps + s].y);
      temp_z.push_back(particle_paths[n*this->max_steps + s].z);
    }

    for (unsigned int i = 0; i < (unsigned int) p_steps; i++)
    {
      path_file << temp_x.at(i);

      if (i < (unsigned int) p_steps - 1)
        path_file << ",";
      else
        path_file << "\n";
    }

    for (unsigned int i = 0; i < (unsigned int) p_steps; i++)
    {
      path_file << temp_y.at(i);

      if (i < (unsigned int) p_steps - 1)
        path_file << ",";
      else
        path_file << "\n";
    }

    for (unsigned int i = 0; i < (unsigned int) p_steps; i++)
    {
      path_file << temp_z.at(i);

      if (i < (unsigned int) p_steps - 1)
        path_file << ",";
      else
        path_file << "\n";
    }

    temp_x.clear();
    temp_y.clear();
    temp_z.clear();
  }

  path_file.close();
  delete[] particle_paths;
  delete[] particle_steps;
}

// TODO(steve) add brain mask support
void OclPtxHandler::WriteSamplesToDevice(
  const BedpostXData* f_data,
  const BedpostXData* phi_data,
  const BedpostXData* theta_data,
  int num_directions,
  const unsigned short int* brain_mask
  )
{
  int single_direction_size =
    f_data->nx * f_data->ny * f_data->nz;

  int brain_mem_size =
    single_direction_size * sizeof(short);

  int single_direction_mem_size =
    single_direction_size*f_data->ns*sizeof(float4);

  int total_mem_size =
    single_direction_mem_size*num_directions;

  this->samples_buffer_size = total_mem_size;

  this->sample_nx = f_data->nx;
  this->sample_ny = f_data->ny;
  this->sample_nz = f_data->nz;
  this->sample_ns = f_data->ns;

  this->f_samples_buffer =
    cl::Buffer(
        *(context_),
        CL_MEM_READ_ONLY,
        total_mem_size,
        NULL,
        NULL);

  this->theta_samples_buffer =
    cl::Buffer(
        *(context_),
        CL_MEM_READ_ONLY,
        total_mem_size,
        NULL,
        NULL);

  this->phi_samples_buffer =
    cl::Buffer(
        *(context_),
        CL_MEM_READ_ONLY,
        total_mem_size,
        NULL,
        NULL);

  this->brain_mask_buffer =
    cl::Buffer(
        *(context_),
        CL_MEM_READ_ONLY,
        brain_mem_size,
        NULL,
        NULL);

  // enqueue writes

  for (int d = 0; d < num_directions; d++)
  {
    cq_->enqueueWriteBuffer(
        this->f_samples_buffer,
        CL_FALSE,
        d * single_direction_mem_size,
        single_direction_mem_size,
        f_data->data.at(d),
        NULL,
        NULL);

    cq_->enqueueWriteBuffer(
      this->theta_samples_buffer,
      CL_FALSE,
      d * single_direction_mem_size,
      single_direction_mem_size,
      theta_data->data.at(d),
      NULL,
      NULL);

    cq_->enqueueWriteBuffer(
      this->phi_samples_buffer,
      CL_FALSE,
      d * single_direction_mem_size,
      single_direction_mem_size,
      phi_data->data.at(d),
      NULL,
      NULL);
  }

  cq_->enqueueWriteBuffer(
    this->phi_samples_buffer,
    CL_FALSE,
    static_cast<unsigned int>(0),
    brain_mem_size,
    brain_mask,
    NULL,
    NULL);

  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  cq_->finish();
}

void OclPtxHandler::RunKernel(int side)
{
  cl::NDRange particles_to_compute(attrs_.particles_per_side);
  cl::NDRange particle_offset(attrs_.particles_per_side * side);
  cl::NDRange local_range(1);

  kernel_->setArg(
      0,
      sizeof(struct OclPtxHandler::particle_attrs),
      reinterpret_cast<void*>(&attrs_));
  kernel_->setArg(1, *gpu_data);
  kernel_->setArg(2, *gpu_complete);
  kernel_->setArg(3, *gpu_step_count);
  kernel_->setArg(4, *gpu_path);

  cq_->enqueueNDRangeKernel(
    *(kernel_),
    particle_offset,
    particles_to_compute,
    local_range,
    NULL,
    NULL);

  cq_->finish();
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

  // Write particle_data
  ret = cq_->enqueueWriteBuffer(
      *gpu_data,
      true,
      offset * sizeof(struct particle_data),
      sizeof(struct particle_data),
      reinterpret_cast<void*>(data));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    abort();
  }

  // gpu_complete = 0
  ret = cq_->enqueueWriteBuffer(
      *gpu_complete,
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
      *gpu_step_count,
      true,
      offset * sizeof(cl_ushort),
      sizeof(cl_ushort),
      reinterpret_cast<void*>(&zero));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    abort();
  }
}

void OclPtxHandler::ReadStatus(int offset, int count, cl_ushort *ret)
{
  cq_->enqueueReadBuffer(
      *gpu_complete,
      true,
      offset * sizeof(cl_ushort),
      count * sizeof(cl_ushort),
      reinterpret_cast<cl_ushort*>(ret));
}

void OclPtxHandler::DumpPath(int offset, int count, FILE *fd)
{
  cl_ulong *path_buf = new cl_ulong[count * attrs_.num_steps];
  cl_ushort *step_count_buf = new cl_ulong[count * attrs_.num_steps];
  int ret;
  int value;

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
      *gpu_path,
      true,
      offset * attrs_.num_steps * sizeof(cl_ulong),
      count * attrs_.num_steps * sizeof(cl_ulong),
      reinterpret_cast<void*>(path_buf));
  if (CL_SUCCESS != ret)
  {
    puts("Failed to read back path");
    abort();
  }

  ret = cq_->enqueueReadBuffer(
      *gpu_step_count,
      true,
      offset * attrs_.num_steps * sizeof(cl_ulong),
      count * attrs_.num_steps * sizeof(cl_ulong),
      reinterpret_cast<void*>(step_count_buf));
  if (CL_SUCCESS != ret)
  {
    puts("Failed to read back path");
    abort();
  }

  // Now dumpify.
  for (int id = 0; id < count; ++id)
  {
    for (int step = 0; step < attrs_.num_steps; ++step)
    {
      value = path_buf[id * attrs_.num_steps + step];
      // Only dump if this element is before the path's end.
      if (step <= step_count_buf[id] % attrs_.num_steps)
        fprintf(fd, "%i:%i\n", id + offset, value);
    }
  }

  delete path_buf;
  delete step_count_buf
}

int OclPtxHandler::particles_per_side()
{
  return attrs_.particles_per_side;
}

