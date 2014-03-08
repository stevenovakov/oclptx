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
#include <fstream>
#include <sstream>
#include <vector>
#include <mutex>
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
    cl::Kernel* ck
)
{
  this->interpolation_complete = false;
  this->target_section = 0;

  this->ocl_context = cc;
  this->ocl_cq = cq;
  this->ptx_kernel = ck;

  this->total_gpu_mem_size = 0;
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

void OclPtxHandler::ParticlePathsToFile()
{
  float4 * particle_paths;
  particle_paths =
    new float4[this->n_particles*this->particle_path_size];
  unsigned int * particle_steps;
  particle_steps = new unsigned int[this->n_particles];

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

  // blocking
  this->ocl_cq->finish();

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

  for (unsigned int n = 0; n < this->n_particles; n++)
  {
    unsigned int p_steps = particle_steps[n];

    //if (p_steps > 0)
    std::cout<<"Particle " << n << " Steps Taken: " << p_steps <<"\n";
    
    p_steps += 1;

    for (unsigned int s = 0; s < p_steps; s++)
    {
      temp_x.push_back(particle_paths[n*this->particle_path_size+ s].x);
      temp_y.push_back(particle_paths[n*this->particle_path_size+ s].y);
      temp_z.push_back(particle_paths[n*this->particle_path_size+ s].z);
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

unsigned int OclPtxHandler::GpuMemUsed()
{
  return this->total_gpu_mem_size;
}

//*********************************************************************
//
// OclPtxHandler Container Initializations
//
//*********************************************************************

// May want overarching initialize() that simply wraps everything
// below to be called by std::thread
//
// void Initialize()

// TODO @STEVE add brain mask support
void OclPtxHandler::WriteSamplesToDevice(
  const BedpostXData* f_data,
  const BedpostXData* phi_data,
  const BedpostXData* theta_data,
  unsigned int num_directions,
  const unsigned short int* brain_mask
)
{
  unsigned int single_direction_size =
    f_data->nx * f_data->ny * f_data->nz;

  unsigned int brain_mem_size =
    single_direction_size * sizeof(unsigned short int);

  unsigned int single_direction_mem_size =
    single_direction_size*f_data->ns*sizeof(float);

  unsigned int total_mem_size =
    single_direction_mem_size*num_directions;

  this->samples_buffer_size = total_mem_size;

  this->sample_nx = f_data->nx;
  this->sample_ny = f_data->ny;
  this->sample_nz = f_data->nz;
  this->sample_ns = f_data->ns;

  // diagnostics
  std::cout<<"Brain Mem Size: "<< brain_mem_size <<"\n";
  std::cout<<"Samples Size: "<< single_direction_mem_size << "\n";
  std::cout<<"Nx : " << this->sample_nx <<"\n";
  std::cout<<"Ny : " << this->sample_ny <<"\n";
  std::cout<<"Nz : " << this->sample_nz <<"\n";
  std::cout<<"Ns : " << this->sample_ns <<"\n";
  // diagnostics

  this->f_samples_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_ONLY,
      total_mem_size,
      NULL,
      NULL
    );

  this->theta_samples_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_ONLY,
      total_mem_size,
      NULL,
      NULL
    );

  this->phi_samples_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_ONLY,
      total_mem_size,
      NULL,
      NULL
    );

  this->brain_mask_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_ONLY,
      brain_mem_size,
      NULL,
      NULL
    );

  // enqueue writes

  for (unsigned int d=0; d<num_directions; d++)
  {

    this->ocl_cq->enqueueWriteBuffer(
        this->f_samples_buffer,
        CL_FALSE,
        d * single_direction_mem_size,
        single_direction_mem_size,
        f_data->data.at(d),
        NULL,
        NULL
    );

    this->ocl_cq->enqueueWriteBuffer(
      this->theta_samples_buffer,
      CL_FALSE,
      d * single_direction_mem_size,
      single_direction_mem_size,
      theta_data->data.at(d),
      NULL,
      NULL
    );

    this->ocl_cq->enqueueWriteBuffer(
      this->phi_samples_buffer,
      CL_FALSE,
      d * single_direction_mem_size,
      single_direction_mem_size,
      phi_data->data.at(d),
      NULL,
      NULL
    );
  }

  this->ocl_cq->enqueueWriteBuffer(
    this->phi_samples_buffer,
    CL_FALSE,
    static_cast<unsigned int>(0),
    brain_mem_size,
    brain_mask,
    NULL,
    NULL
  );

  this->total_gpu_mem_size += 3*total_mem_size + brain_mem_size;

  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->ocl_cq->finish();
}

void OclPtxHandler::WriteInitialPosToDevice(
  const float4* initial_positions,
  unsigned int nparticles,
  unsigned int maximum_steps,
  unsigned int ndevices,
  unsigned int device_num
)
{
  unsigned int sec_size = nparticles/ndevices;

  this->section_size = sec_size;
  this->n_particles = nparticles;
  this->max_steps = maximum_steps;
  this->particle_path_size = maximum_steps + 1;

  unsigned int path_mem_size =
    sec_size*particle_path_size*sizeof(float4);
  unsigned int path_steps_mem_size = sec_size*sizeof(unsigned int);
  this->particle_uint_mem_size = path_steps_mem_size;
  this->particles_mem_size = path_mem_size;

  // if MT: wrap in mutex (to avoid race on initial_positions)
  const float4* start_pos_data =
    initial_positions + (sec_size*device_num);
  // if MT: wrap in mutex (to avoid race on initial_positions)

  // also doubles as the "is done" initial data
  std::vector<unsigned int> initial_steps(sec_size, 0);

  // delete this at end of function always
  float4* pos_container;
  pos_container = new float4[sec_size * particle_path_size];

  // the first entry in row i will be the particle start location
  // the rest is garbage data (that's fine)
  for (unsigned int i = 0; i < sec_size; i++)
  {
    pos_container[particle_path_size*i] = *start_pos_data;
    start_pos_data++;
    this->particle_indeces_left.push_back(i);
    this->particle_complete.push_back(static_cast<unsigned int>(0));
  }

  std::cout<<"Sec Size: "<< this->section_size <<"\n";
  std::cout<<"N Particles: " << this->n_particles <<"\n";
  std::cout<<"Max Steps: " << this->max_steps <<"\n";
  std::cout<<"Particle Steps Mem Size: "<<
    this->particle_uint_mem_size<<"\n";
  std::cout<<"Particle Paths Mem Size: " <<
    this->particles_mem_size<<"\n";


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

  // enqueue writes
  // both "steps taken" and "done" write the same array (all zeros)

  this->ocl_cq->enqueueWriteBuffer(
    this->particle_paths_buffer,
    CL_FALSE,
    static_cast<unsigned int>(0),
    path_mem_size,
    pos_container,
    NULL,
    NULL
  );

  this->ocl_cq->enqueueWriteBuffer(
    this->particle_steps_taken_buffer,
    CL_FALSE,
    static_cast<unsigned int>(0),
    path_steps_mem_size,
    initial_steps.data(),
    NULL,
    NULL
  );

  this->ocl_cq->enqueueWriteBuffer(
    this->particle_done_buffer,
    CL_FALSE,
    static_cast<unsigned int>(0),
    path_steps_mem_size,
    initial_steps.data(),
    NULL,
    NULL
  );

  this->total_gpu_mem_size += path_mem_size + 2*path_steps_mem_size;
  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->ocl_cq->finish();

  delete[] pos_container;
}


void OclPtxHandler::SingleBufferInit(
  unsigned int particle_interval_size,
  unsigned int step_interval_size
)
{
  this->num_steps = step_interval_size;

  this->particles_size = particle_interval_size;
  unsigned int interval_mem_size =
    particle_interval_size*sizeof(unsigned int);

  // first iteration is same size
  this->todo_range.push_back( particle_interval_size );

  std::vector<unsigned int> temp;
  this->particle_todo.push_back(temp);

  for (unsigned int k=0; k<particle_interval_size; k++)
  {
    this->particle_todo.at(0).push_back(
      this->particle_indeces_left.back());
    this->particle_indeces_left.pop_back();
  }

  // Initialize both buffers

  this->compute_index_buffers.push_back(
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      interval_mem_size,
      NULL,
      NULL)
  );

  // Copy over initial data to both buffers
  this->ocl_cq->enqueueWriteBuffer(
    this->compute_index_buffers.at(0),
    CL_FALSE,
    static_cast<unsigned int>(0),
    interval_mem_size,
    this->particle_todo.at(0).data(),
    NULL,
    NULL
  );

  this->total_gpu_mem_size += interval_mem_size;

  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->ocl_cq->finish();

}

//
// This should be totally changed to support Jeff's scheme
// Currently does not work for some reason.
//
void OclPtxHandler::DoubleBufferInit(
  unsigned int particle_interval_size,
  unsigned int step_interval_size
)
{
  this->num_steps = step_interval_size;

  this->particles_size = particle_interval_size;
  unsigned int interval_mem_size =
    particle_interval_size*sizeof(unsigned int);

  // first iteration is same size
  this->todo_range.push_back( particle_interval_size );
  this->todo_range.push_back( particle_interval_size );

  std::vector<unsigned int> temp;
  this->particle_todo.push_back(temp);
  this->particle_todo.push_back(temp);

  for (unsigned int k=0; k<particle_interval_size; k++)
  {
    this->particle_todo.at(0).push_back(
      this->particle_indeces_left.back());
    this->particle_indeces_left.pop_back();
    this->particle_todo.at(1).push_back(
      this->particle_indeces_left.back());
    this->particle_indeces_left.pop_back();
  }

  // Initialize both buffers

  this->compute_index_buffers.push_back(
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      interval_mem_size,
      NULL,
      NULL)
  );
  this->compute_index_buffers.push_back(
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      interval_mem_size,
      NULL,
      NULL)
  );

  // Copy over initial data to both buffers
  this->ocl_cq->enqueueWriteBuffer(
    this->compute_index_buffers.at(0),
    CL_FALSE,
    static_cast<unsigned int>(0),
    interval_mem_size,
    this->particle_todo.at(0).data(),
    NULL,
    NULL
  );

  this->ocl_cq->enqueueWriteBuffer(
    this->compute_index_buffers.at(1),
    CL_FALSE,
    static_cast<unsigned int>(0),
    interval_mem_size,
    this->particle_todo.at(1).data(),
    NULL,
    NULL
  );

  this->total_gpu_mem_size += 2*interval_mem_size;

  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->ocl_cq->finish();

}

//*********************************************************************
//
// OclPtxHandler Reduction
//
//*********************************************************************

void OclPtxHandler::ReduceInit(unsigned int particles_per,
                                std::string reduction_style)
{
  // actually might not need this thanks to DoubleBufferInit

}

void OclPtxHandler::Reduce()
{
  unsigned int t_sec = this->target_section  & 0x00000000;

  std::cout<<"t_sec : " << t_sec <<"\n";

  std::vector<unsigned int>* reduce_vector =
    &(this->particle_todo.at(t_sec));
  std::vector<unsigned int>* done_vector =
    &(this->particle_complete);
  std::vector<unsigned int>* left_vector =
    &(this->particle_indeces_left);

  unsigned int s_size = this->num_steps;

  unsigned int new_todo_range = 0;

  // first pop all of the finished indeces

  // apparently in CL1.1 + ,
  // all command queue CALLS are thread safe
  // std::unique_lock<std::mutex> cqlock(this->cq_mutex);

  this->ocl_cq->enqueueReadBuffer(
    this->particle_done_buffer,
    CL_TRUE, // blocking
    0,
    this->particle_uint_mem_size,
    done_vector->data()
  );

  //cqlock.unlock();

  for (unsigned int i = 0; i < reduce_vector->size(); i++)
  {
    if (done_vector->at(i) > 0)
      reduce_vector->erase(reduce_vector->begin() + i);
  }

  unsigned int old_size_left = reduce_vector->size();
  unsigned int particles_left = this->particle_indeces_left.size();
  unsigned int gap_size = s_size - old_size_left;

  if ( gap_size > particles_left )
    gap_size = particles_left;

  for (unsigned int i = 0; i < gap_size; i++)
  {
    reduce_vector->push_back(left_vector->back());
    left_vector->pop_back();
  }

  if (left_vector->size() == 0)
    this->interpolation_complete = true;

  new_todo_range = old_size_left + gap_size;

  this->ocl_cq->enqueueWriteBuffer(
    this->compute_index_buffers.at(t_sec),
    CL_TRUE, //blocking
    (unsigned int) 0,
    new_todo_range*sizeof(unsigned int),
    left_vector->data(),
    NULL,
    NULL
  );

  std::unique_lock<std::mutex> rdlock(this->reduce_mutex);
  this->target_section = t_sec;
  this->todo_range.at(t_sec) = new_todo_range;

  std::cout<<"new range: " << new_todo_range <<"\n";
  rdlock.unlock();
}

//*********************************************************************
//
// OclPtxHandler Tractography
//
//*********************************************************************

void OclPtxHandler::Interpolate()
{
  //std::lock_guard<std::mutex> klock(this->kernel_mutex);

  unsigned int t_sec = this->target_section;

  //
  // Currently Handles single voxel/mask + No other options ONLY
  //

  cl::NDRange global_range(this->todo_range.at(t_sec));
  cl::NDRange local_range(1);

  // the indeces to compute, always first
  this->ptx_kernel->setArg(0, this->compute_index_buffers.at(t_sec));

  // particle status buffers
  this->ptx_kernel->setArg(1, this->particle_paths_buffer);
  this->ptx_kernel->setArg(2, this->particle_steps_taken_buffer);
  this->ptx_kernel->setArg(3, this->particle_done_buffer);

  // sample data buffers
  this->ptx_kernel->setArg(4, this->f_samples_buffer);
  this->ptx_kernel->setArg(5, this->phi_samples_buffer);
  this->ptx_kernel->setArg(6, this->theta_samples_buffer);
  this->ptx_kernel->setArg(7, this->brain_mask_buffer);

  this->ptx_kernel->setArg(8, this->section_size);
  this->ptx_kernel->setArg(9, this->max_steps);
  this->ptx_kernel->setArg(10, this->sample_nx);
  this->ptx_kernel->setArg(11, this->sample_ny);
  this->ptx_kernel->setArg(12, this->sample_nz);
  this->ptx_kernel->setArg(13, this->sample_ns);

  this->ptx_kernel->setArg(14, this->num_steps);
  // Now I have to write a kernel!!! Yaaaay : )

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


//*********************************************************************
//
// Assorted Functions
//
//*********************************************************************



//EOF