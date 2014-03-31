/*  Copyright (C) 2014
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
#include <cstdlib>
#include <vector>

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>


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
uint64_t rand_64();
int init_rng(cl_ulong8* rng, int seed, int count);

//*********************************************************************
//
// OclPtxHandler Constructors/Destructors
//
//*********************************************************************

//
// Constructor(s)
//
OclPtxHandler::OclPtxHandler(
    cl::Context* cc,
    cl::CommandQueue* cq,
    cl::Kernel* ptx,
    cl::Kernel* sum,
    float curv_thresh,
    float dr,
    EnvironmentData * env
)
{
  this->interpolation_complete = false;

  this->ocl_context = cc;
  this->ocl_cq = cq;
  this->ptx_kernel = ptx;
  this->sum_kernel = sum;
  
  this->curvature_threshold = curv_thresh;
  this->delta_r = dr;

  this->env_dat = env;

  this->total_gpu_mem_used = this->env_dat->total_static_gpu_mem;
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
// OclPtxHandler  Set/Get
//
//*********************************************************************

void OclPtxHandler::ParticlePathsToFile(std::string path_filename)
{
  float4 * particle_paths;
  particle_paths =
    new float4[this->section_size*this->particle_path_size];
  unsigned int * particle_steps;
  particle_steps = new unsigned int[this->section_size];

  uint32_t * particle_exclusion;
  uint32_t * particle_waypoints;
  uint32_t waypts = this->env_dat->n_waypts;

  this->ocl_cq->enqueueReadBuffer(
    this->particle_paths_buffer,
    CL_FALSE,
    0,
    this->particles_mem_size,
    particle_paths
  );
  this->ocl_cq->enqueueReadBuffer(
    this->particle_steps_taken_buffer,
    CL_FALSE,
    0,
    this->particle_uint_mem_size,
    particle_steps
  );

  if (this->env_dat->exclusion_mask)
  {
    particle_exclusion = new unsigned int[this->section_size];

    this->ocl_cq->enqueueReadBuffer(
      this->particle_exclusion_buffer,
      CL_FALSE,
      0,
      this->particle_uint_mem_size,
      particle_exclusion
    );
  }

  if (waypts > 0)
  {
    particle_waypoints = new unsigned int[waypts * this->section_size];

      particle_exclusion = new unsigned int[this->section_size];
    this->ocl_cq->enqueueReadBuffer(
      this->particle_waypoints_buffer,
      CL_FALSE,
      0,
      waypts * this->particle_uint_mem_size,
      particle_waypoints
    );
  }

  // blocking
  this->ocl_cq->finish();

  // now dump to file

  FILE * path_file;

  std::vector<float> temp_x, temp_y, temp_z;

  printf("Writing Path Data to %s\n", path_filename.c_str());

  path_file = fopen(path_filename.c_str(), "ab");

  uint32_t waybreak = 1;

  for (unsigned int n = 0; n < this->section_size; n++)
  {
    unsigned int p_steps = particle_steps[n];

    if (this->env_dat->exclusion_mask)
    {
      if (particle_exclusion[n] > 0)
        waybreak = 0;
    }

    if (waypts > 0)
    {
      for (uint32_t w = 0; w < waypts; w++)
        waybreak *= particle_waypoints[n*waypts + w];
    }

    if (waybreak == 0)
    {
      waybreak = 1;
      continue;
    }


    //if (p_steps > 0)
    //printf("Particle: %d, Steps Taken: %d\n", n, p_steps);
    
    // took this out for comparison test to ptx2, add back in.
    p_steps += 1;

    fprintf(path_file, "[");

    for (unsigned int s = 0; s < p_steps; s++)
    {
      temp_x.push_back(particle_paths[n*this->particle_path_size+ s].x);
      temp_y.push_back(particle_paths[n*this->particle_path_size+ s].y);
      temp_z.push_back(particle_paths[n*this->particle_path_size+ s].z);
    }

    for (unsigned int i = 0; i < (unsigned int) p_steps; i++)
    {
      fprintf(path_file, "[%.6f,%.6f,%.6f]", temp_x.at(i), temp_y.at(i),
        temp_z.at(i));

      if (i < (unsigned int) p_steps - 1)
        fprintf(path_file, ",");
      else
        fprintf(path_file, "]");
    }

    temp_x.clear();
    temp_y.clear();
    temp_z.clear();

    if (n < this->n_particles -1)
      fprintf(path_file, ",\n");
  }

  fclose(path_file);
  delete[] particle_paths;
  delete[] particle_steps;
}

void OclPtxHandler::GetPdfData( uint32_t* container)
{
  this->ocl_cq->enqueueReadBuffer(
    this->global_pdf_buffer,
    CL_FALSE,
    0,
    this->env_dat->global_pdf_mem_size,
    container
  );
  this->ocl_cq->finish();
}

cl_ulong OclPtxHandler::GpuMemUsed()
{
  return this->total_gpu_mem_used;
}
//*********************************************************************
//
// OclPtxHandler Container Initializations
//
//*********************************************************************

void OclPtxHandler::WriteInitialPosToDevice(
  const float4* initial_positions,
  uint32_t nparticles,
  uint32_t maximum_steps,
  uint32_t ndevices,
  uint32_t device_num
)
{
  uint32_t sec_size = nparticles/ndevices;

  this->section_size = 2*sec_size;
  this->n_particles = 2*nparticles;
  this->max_steps = maximum_steps;
  this->particle_path_size = maximum_steps + 1;

  uint32_t path_mem_size =
    this->section_size*this->particle_path_size*sizeof(float4);
  uint32_t path_steps_mem_size = this->section_size*sizeof(unsigned int);

  uint32_t pdfs_size =
    this->env_dat->pdf_entries_per_particle * this->n_particles;
  uint32_t pdfs_mem_size =
    this->env_dat->particle_pdf_mask_mem_size * this->n_particles;

  this->particle_uint_mem_size = path_steps_mem_size;
  this->particles_mem_size = path_mem_size;

  // if MT: wrap in mutex (to avoid race on initial_positions)
  const float4* start_pos_data =
    initial_positions + (this->section_size*device_num);
  // if MT: wrap in mutex (to avoid race on initial_positions)

  // also doubles as the "is done" initial data
  std::vector<uint32_t> initial_steps(this->section_size, 0);
  std::vector<uint32_t> initial_waypts(
    this->section_size*this->env_dat->n_waypts, 0);
  std::vector<uint32_t> init_pdfs(pdfs_size, 0);
  std::vector<uint32_t> initial_global_pdf(this->env_dat->global_pdf_size, 0);

  std::vector<uint32_t> loopcheck_locs;
  std::vector<float4> loopcheck_dirs;

  if (this->env_dat->loopcheck)
  {
    loopcheck_locs.resize(this->n_particles *
      this->env_dat->loopcheck_location_size);
    loopcheck_dirs.resize(this->n_particles *
      this->env_dat->loopcheck_dir_size);
  }

  // delete this at end of function always
  float4* pos_container;
  pos_container = new float4[this->section_size * this->particle_path_size];

  // the first entry in row i will be the particle start location
  // the rest is garbage data (that's fine)
  for (uint32_t i = 0; i < sec_size; i++)
  {
    pos_container[this->particle_path_size*(2*i)] = *start_pos_data;
    pos_container[this->particle_path_size*(2*i + 1)] = *start_pos_data;
    start_pos_data++;
    this->particle_indeces_left.push_back(2*i);
    this->particle_indeces_left.push_back(2*i + 1);
    this->particle_complete.push_back(static_cast<uint32_t>(0));
    this->particle_complete.push_back(static_cast<uint32_t>(0));
  }

  // std::cout<<"Sec Size: "<< this->section_size <<"\n";
  // std::cout<<"N Particles: " << this->n_particles <<"\n";
  // std::cout<<"Max Steps: " << this->max_steps <<"\n";
  // std::cout<<"Particle Steps Mem Size: "<<
  //   this->particle_uint_mem_size<<"\n";
  // std::cout<<"Particle Paths Mem Size: " <<
  //   this->particles_mem_size<<"\n";

  this->particle_paths_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      path_mem_size,
      NULL,
      NULL
    );

  this->particle_steps_taken_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      path_steps_mem_size,
      NULL,
      NULL
    );

  this->particle_done_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      path_steps_mem_size,
      NULL,
      NULL
    );

  if (this->env_dat->exclusion_mask)
    this->particle_exclusion_buffer =
      cl::Buffer(
        *(this->ocl_context),
        CL_MEM_READ_WRITE,
        path_steps_mem_size,
        NULL,
        NULL
      );

  if (this->env_dat->n_waypts > 0)
    this->particle_waypoints_buffer =
      cl::Buffer(
        *(this->ocl_context),
        CL_MEM_READ_WRITE,
        this->env_dat->n_waypts*path_steps_mem_size,
        NULL,
        NULL
      );

  if (this->env_dat->loopcheck)
  {
    this->particle_loopcheck_location_buffer =
      cl::Buffer(
        *(this->ocl_context),
        CL_MEM_READ_WRITE,
        this->env_dat->particle_loopcheck_location_mem_size * this->n_particles,
        NULL,
        NULL
      );
    this->particle_loopcheck_dir_buffer =
      cl::Buffer(
        *(this->ocl_context),
        CL_MEM_READ_WRITE,
        this->env_dat->particle_loopcheck_dir_mem_size * this->n_particles,
        NULL,
        NULL
      );
  }

  this->global_pdf_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      this->env_dat->global_pdf_mem_size,
      NULL,
      NULL
    );

  this->particles_pdf_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      pdfs_mem_size,
      NULL,
      NULL
    );
  // enqueue writes
  // both "steps taken" and "done" write the same array (all zeros)

  this->ocl_cq->enqueueWriteBuffer(
    this->particle_paths_buffer,
    CL_FALSE,
    static_cast<uint32_t>(0),
    path_mem_size,
    pos_container,
    NULL,
    NULL
  );

  this->ocl_cq->enqueueWriteBuffer(
    this->particle_steps_taken_buffer,
    CL_FALSE,
    static_cast<uint32_t>(0),
    path_steps_mem_size,
    initial_steps.data(),
    NULL,
    NULL
  );

  this->ocl_cq->enqueueWriteBuffer(
    this->particle_done_buffer,
    CL_FALSE,
    static_cast<uint32_t>(0),
    path_steps_mem_size,
    initial_steps.data(),
    NULL,
    NULL
  );

  if (this->env_dat->exclusion_mask)
    this->ocl_cq->enqueueWriteBuffer(
      this->particle_exclusion_buffer,
      CL_FALSE,
      static_cast<uint32_t>(0),
      path_steps_mem_size,
      initial_steps.data(),
      NULL,
      NULL
    );

  if (this->env_dat->n_waypts > 0)
    this->ocl_cq->enqueueWriteBuffer(
      this->particle_waypoints_buffer,
      CL_FALSE,
      static_cast<uint32_t>(0),
      this->env_dat->n_waypts*path_steps_mem_size,
      initial_waypts.data(),
      NULL,
      NULL
    );

  if (this->env_dat->loopcheck)
  {
    this->ocl_cq->enqueueWriteBuffer(
      this->particle_loopcheck_location_buffer,
      CL_FALSE,
      static_cast<uint32_t>(0),
      this->env_dat->particle_loopcheck_location_mem_size*this->n_particles,
      loopcheck_locs.data(),
      NULL,
      NULL
    );
    this->ocl_cq->enqueueWriteBuffer(
      this->particle_loopcheck_dir_buffer,
      CL_FALSE,
      static_cast<uint32_t>(0),
      this->env_dat->particle_loopcheck_dir_mem_size*this->n_particles,
      loopcheck_dirs.data(),
      NULL,
      NULL
    );
  }

  this->ocl_cq->enqueueWriteBuffer(
    this->global_pdf_buffer,
    CL_FALSE,
    static_cast<uint32_t>(0),
    this->env_dat->global_pdf_mem_size,
    initial_global_pdf.data(),
    NULL,
    NULL
  );

  this->ocl_cq->enqueueWriteBuffer(
    this->particles_pdf_buffer,
    CL_FALSE,
    static_cast<uint32_t>(0),
    pdfs_mem_size,
    init_pdfs.data(),
    NULL,
    NULL
  );

  this->total_gpu_mem_used += path_mem_size + 2*path_steps_mem_size +
    this->env_dat->particle_pdf_mask_mem_size * this->n_particles;
  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->ocl_cq->finish();

  delete[] pos_container;
}

void OclPtxHandler::PrngInit()
{
  unsigned int path_rng_mem_size = this->section_size *
    static_cast<unsigned int>(sizeof(cl_ulong8));
  
  cl_ulong8 *rng = new cl_ulong8[this->section_size];

  // TODO @STEVE
  // Use for now, for testing, replace with user seed options later
  int seed = time(NULL);

  init_rng(rng, seed, this->section_size);

  this->particle_rng_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      path_rng_mem_size,
      NULL,
      NULL
    );

  this->ocl_cq->enqueueWriteBuffer(
    this->particle_rng_buffer,
    CL_FALSE,
    static_cast<unsigned int>(0),
    path_rng_mem_size,
    rng,
    NULL,
    NULL
  );

  this->total_gpu_mem_used += path_rng_mem_size;
  delete[] rng;

  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->ocl_cq->finish();
}

void OclPtxHandler::SingleBufferInit()
{
  unsigned int interval_mem_size =
    this->section_size*sizeof(unsigned int);

  // first iteration is same size
  this->todo_range = this->section_size;

  std::vector<unsigned int> temp;
  this->particle_todo.push_back(temp);

  for (unsigned int k=0; k<this->section_size; k++)
  {
    this->particle_todo.at(0).push_back(
      this->particle_indeces_left.back());
    this->particle_indeces_left.pop_back();
  }

  // Initialize buffer

  this->compute_index_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      interval_mem_size,
      NULL,
      NULL
    );

  // Copy over initial data to buffer
  this->ocl_cq->enqueueWriteBuffer(
    this->compute_index_buffer,
    CL_FALSE,
    static_cast<unsigned int>(0),
    interval_mem_size,
    this->particle_todo.at(0).data(),
    NULL,
    NULL
  );

  this->total_gpu_mem_used += interval_mem_size;
  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->ocl_cq->finish();
}

//*********************************************************************
//
// OclPtxHandler Tractography
//
//*********************************************************************

void OclPtxHandler::Interpolate()
{
  //
  // Currently Handles single voxel/mask + No other options ONLY
  //
  cl::NDRange global_range(this->todo_range);
  cl::NDRange local_range(1);

  // the indeces to compute, always first
  this->ptx_kernel->setArg(0, this->compute_index_buffer);

  // particle status buffers
  this->ptx_kernel->setArg(1, this->particle_paths_buffer);
  this->ptx_kernel->setArg(2, this->particles_pdf_buffer);
  this->ptx_kernel->setArg(3, this->particle_steps_taken_buffer);
  this->ptx_kernel->setArg(4, this->particle_done_buffer);
  this->ptx_kernel->setArg(5, this->particle_rng_buffer);

  // sample data buffers
  this->ptx_kernel->setArg(6, *(this->env_dat->f_samples_buffers[0]));
  this->ptx_kernel->setArg(7, *(this->env_dat->phi_samples_buffers[0]));
  this->ptx_kernel->setArg(8, *(this->env_dat->theta_samples_buffers[0]));
  this->ptx_kernel->setArg(9, *(this->env_dat->brain_mask_buffer));

  this->ptx_kernel->setArg(10, this->max_steps);
  this->ptx_kernel->setArg(11, this->n_particles);
  this->ptx_kernel->setArg(12, this->env_dat->nx);
  this->ptx_kernel->setArg(13, this->env_dat->ny);
  this->ptx_kernel->setArg(14, this->env_dat->nz);
  this->ptx_kernel->setArg(15, this->env_dat->ns);

  this->ptx_kernel->setArg(16, this->max_steps);
  this->ptx_kernel->setArg(17, this->curvature_threshold);
  this->ptx_kernel->setArg(18, this->delta_r);
  printf("DR %f\n", this->delta_r);
  // optional buffers
  uint32_t last_index = 18;

  if (this->env_dat->n_waypts > 0)
  {
    last_index += 1;
    this->ptx_kernel->setArg(
      last_index, this->env_dat->waypoint_masks_buffer);
    last_index += 1;
    this->ptx_kernel->setArg(last_index, this->particle_waypoints_buffer);
    last_index += 1;
    this->ptx_kernel->setArg(last_index, this->env_dat->n_waypts);
  }

  if (this->env_dat->exclusion_mask)
  {
    last_index += 1;
    this->ptx_kernel->setArg(
      last_index, this->env_dat->exclusion_mask_buffer);
    last_index += 1;
    this->ptx_kernel->setArg(last_index, this->particle_exclusion_buffer);
  }

  if (this->env_dat->terminate_mask)
  {
    last_index += 1;
    this->ptx_kernel->setArg(
      last_index, this->env_dat->termination_mask_buffer);
  }

  if (this->env_dat->loopcheck)
  {
    last_index += 1;
    this->ptx_kernel->setArg(
      last_index, this->particle_loopcheck_location_buffer);
    last_index += 1;
    this->ptx_kernel->setArg(
      last_index, this->particle_loopcheck_dir_buffer);
  }

  // execute

  this->ocl_cq->enqueueNDRangeKernel(
    *(this->ptx_kernel),
    cl::NullRange,
    global_range,
    local_range,
    NULL,
    NULL
  );

  // BLOCK
  this->ocl_cq->finish();
}

void OclPtxHandler::PdfSum()
{
  // cl::NDRange global_range(
  //     this->env_dat->nx, this->env_dat->ny, this->env_dat->nz);
  // cl::NDRange local_range(1,1,1);

  cl::NDRange global_range(this->env_dat->pdf_entries_per_particle);
  cl::NDRange local_range(1);

  this->sum_kernel->setArg(0, this->global_pdf_buffer);
  this->sum_kernel->setArg(1, this->particles_pdf_buffer);
  this->sum_kernel->setArg(2, this->particle_done_buffer);
  this->sum_kernel->setArg(3, this->n_particles);
  this->sum_kernel->setArg(4, this->env_dat->pdf_entries_per_particle);

  // optional arguments
  uint32_t last_index = 4;

  if (this->env_dat->exclusion_mask)
  {
    last_index += 1;
    this->ptx_kernel->setArg(last_index, this->particle_exclusion_buffer);
  }

  if (this->env_dat->n_waypts > 0)
  {
    last_index += 1;
    this->ptx_kernel->setArg(last_index, this->particle_waypoints_buffer);
    last_index += 1;
    this->ptx_kernel->setArg(last_index, this->env_dat->n_waypts);
  }

  // execulte

  try{
    this->ocl_cq->enqueueNDRangeKernel(
      *(this->sum_kernel),
      cl::NullRange,
      global_range,
      local_range,
      NULL,
      NULL
    );
  }
  catch(cl::Error err){
    if( (err.err()) != 0)
    {
      std::cout<<"ERROR: " << err.what() << ":" << err.err() << "\n";
      std::cin.get();
    }
  }
  // BLOCK
  this->ocl_cq->finish();
}

//*********************************************************************
//
// Assorted Functions
//
//*********************************************************************

uint64_t rand_64()
{
  // Assumption: rand() gives at least 16 bits.  It gives 31 on my system.
  assert(RAND_MAX >= (1<<16));

  uint64_t a,b,c,d;
  a = rand() & ((1<<16)-1);
  b = rand() & ((1<<16)-1);
  c = rand() & ((1<<16)-1);
  d = rand() & ((1<<16)-1);

  uint64_t r = ((a << (16*3)) | (b << (16*2)) | (c << (16)) | d);

  return r;
}

int init_rng(cl_ulong8* rng, int seed, int count)
{
  // TODO
  // Hmm, not sure how to use this to cleanly exit the program yet,
  //
  // if (RAND_MAX < (1<<16)) {
  //   puts("RAND_MAX is too small");
  //   return -1;
  // }

  srand(seed);

  uint64_t init;
  for (int i = 0; i < count; ++i)
    for (int j = 0; j < 5; j++)
    {
      init = rand_64();
      rng[i].s[j] = init;
    }

  return 0;
}

//EOF