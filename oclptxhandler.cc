/*  Copyright (C) 2004
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



//*********************************************************************
//
// OclPtxHandler Container Initializations
//
//*********************************************************************

// May want overarching initialize() that simply wraps everything
// below to be called by std::thread
//
// void Initialize()

void WriteSamplesToDevice( BedpostXData* f_data,
                            BedpostXData* phi_data,
                            BedpostXData* theta_data,
                            unsigned int num_directions
                          )
{
  unsigned int single_direction_size =
    f_data->nx * f_data->ny * f_data->nz * f_data->ns;

  unsigned int single_direction_mem_size =
    single_direction_size*sizeof(float4);

  unsigned int total_mem_size =
    single_direction_mem_size*num_directions;

  this->samples_buffer_size = total_mem_size;

  this->sample_nx = f_data.nx;
  this->sample_ny = f_data.ny;
  this->sample_nz = f_data.nz;
  this->sample_ns = f_data.ns;

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

  // enqueue writes

  for (unsigned int d=0; d<num_directions; d++)
  {
    this->cq->enqueueWriteBuffer(
        this->f_samples_buffer,
        CL_FALSE,
        d * single_direction_mem_size,
        single_direction_mem_size,
        f_data->data.at(d),
        NULL,
        NULL
    );

    this->cq->enqueueWriteBuffer(
      this->theta_samples_buffer,
      CL_FALSE,
      d * single_direction_mem_size,
      single_direction_mem_size,
      theta_data->data.at(d),
      NULL,
      NULL
    );

    this->cq->enqueueWriteBuffer(
      this->phi_samples_buffer,
      CL_FALSE,
      d * single_direction_mem_size,
      single_direction_mem_size,
      phi_data->data.at(d),
      NULL,
      NULL
    );
  }
  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->cq->finish();
}

void WriteInitialPosToDevice(float4* initial_positions,
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
  unsigned int path_steps_size = sec_size*sizeof(unsigned int);

  // if MT: wrap in mutex (to avoid race on initial_positions)
  float4* start_pos_data = initial_positions + (sec_size*device_num);
  // if MT: wrap in mutex (to avoid race on initial_positions)

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
      path_steps_size,
      NULL,
      NULL
    );

  // enqueue writes

  this->cq->enqueueWriteBuffer(
    this->particle_paths_buffer,
    CL_FALSE,
    static_cast<unsigned int>(0),
    path_mem_size,
    pos_container,
    NULL,
    NULL
  );

  this->cq->enqueueWriteBuffer(
    this->particle_steps_taken_buffer,
    CL_FALSE,
    static_cast<unsigned int>(0),
    path_steps_size,
    initial_steps.data(),
    NULL,
    NULL
  );

  // may not need to do this here, may want to wait to block until
  // all "initialization" operations are finished.
  this->cq->finish();

  delete pos_container[];
}

void OclPtxHandler::DoubleBufferInit()
{


}

//*********************************************************************
//
// OclPtxHandler Reduction
//
//*********************************************************************

void OclPtxHandler::ReduceInit()
{


}

void OclPtxHandler:Reduce()
{



}

//*********************************************************************
//
// OclPtxHandler Tractography
//
//*********************************************************************

void OclPtxHandler::Interpolate()
{


}


//*********************************************************************
//
// Assorted Functions
//
//*********************************************************************



//EOF