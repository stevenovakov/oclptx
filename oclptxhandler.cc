/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

#include "oclptxhandler.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mutex>

#include "collatz_particle.h"

#define __CL_ENABLE_EXCEPTIONS
// adds exception support from CL libraries
// define before CL headers inclusion
// jeff: seriously?  What about C++ ODR?

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif


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
  particle_paths = new float4[this->n_particles*this->max_steps];
  unsigned int * particle_steps;
  particle_steps = new unsigned int[this->n_particles];

  this->ocl_cq->enqueueReadBuffer(
    this->particle_paths_buffer,
    CL_FALSE, // blocking
    0,
    this->particles_mem_size,
    particle_paths
  );
  this->ocl_cq->enqueueReadBuffer(
    this->particle_steps_taken_buffer,
    CL_FALSE, // blocking
    0,
    this->particle_uint_mem_size,
    particle_steps
  );
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
    single_direction_size*f_data->ns*sizeof(float4);

  unsigned int total_mem_size =
    single_direction_mem_size*num_directions;

  this->samples_buffer_size = total_mem_size;

  this->sample_nx = f_data->nx;
  this->sample_ny = f_data->ny;
  this->sample_nz = f_data->nz;
  this->sample_ns = f_data->ns;

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

  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->ocl_cq->finish();
}

#if 0
void OclPtxHandler::WriteInitialPosToDevice(
  const float4* initial_positions,
  const int4* initial_elem,
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

  unsigned int path_mem_size = sec_size*maximum_steps*sizeof(float4);
  unsigned int path_steps_mem_size = sec_size*sizeof(unsigned int);
  this->particle_uint_mem_size = path_steps_mem_size;
  this->particles_mem_size = path_mem_size;

  unsigned int elem_size = sec_size;
  unsigned int elem_mem_size = elem_size*sizeof(int4);

  // if MT: wrap in mutex (to avoid race on initial_positions)
  const float4* start_pos_data =
    initial_positions + (sec_size*device_num);
  const int4* start_elem_data = initial_elem + (sec_size*device_num);
  // if MT: wrap in mutex (to avoid race on initial_positions)

  // also doubles as the "is done" initial data
  std::vector<unsigned int> initial_steps(sec_size, 0);

  // delete this at end of function always
  float4* pos_container;
  pos_container = new float4[sec_size * maximum_steps];

  // the first entry in row i will be the particle start location
  // the rest is garbage data (that's fine)
  for ( unsigned int i = 0; i < sec_size; i++)
  {
    pos_container[maximum_steps*i] = *start_pos_data;
    start_pos_data++;
    this->particle_indeces_left.push_back(i);
    this->particle_complete.push_back(static_cast<unsigned int>(0));
  }

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

  this->particle_elem_buffer =
    cl::Buffer(
      *(this->ocl_context),
      CL_MEM_READ_WRITE,
      elem_mem_size,
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
    this->particle_elem_buffer,
    CL_FALSE,
    static_cast<unsigned int>(0),
    elem_mem_size,
    start_elem_data,
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

  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->ocl_cq->finish();

  delete[] pos_container;
}
#endif

void OclPtxHandler::RunCollatzKernel(struct particle::particles *p, int side)
{
  cl::NDRange particles_to_compute(p->attrs.particles_per_side);
  cl::NDRange particle_offset(p->attrs.particles_per_side * side);
  cl::NDRange local_range(1);

  this->ptx_kernel->setArg(
      0,
      sizeof(struct particle::particle_attrs),
      reinterpret_cast<void*>(&p->attrs));
  this->ptx_kernel->setArg(1, *p->gpu_data);
  this->ptx_kernel->setArg(2, *p->gpu_complete);
  this->ptx_kernel->setArg(3, *p->gpu_path);

  this->ocl_cq->enqueueNDRangeKernel(
    *(this->ptx_kernel),
    particle_offset,
    particles_to_compute,
    local_range,
    NULL,
    NULL);

  this->ocl_cq->finish();
}

