/*  Copyright (C) 2004
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

#include "customtypes.h"

class OclPtxHandler{

  public:
    OclPtxHandler(){};

    OclPtxHandler(  cl::Context* cc,
                    cl::CommandQueue* cq,
                    cl::Kernel* ck);

    ~OclPtxHandler();

    //
    // Set/Get
    //

    std::vector<float4> GetParticlePaths();   // do at end

    bool IsFinished(){ return this->interpolation_complete; };

    //
    // OCL Initialization
    //

    // May want overarching initialize() that simply wraps everything
    // below to be called by std::thread
    //
    // void Initialize()

    void WriteSamplesToDevice( BedpostXData* f_data,
                                BedpostXData* phi_data,
                                BedpostXData* theta_data,
                                unsigned int num_directions
                              );
    // may want to compute offset beforehand in samplemanager,
    // can decide later.

    void WriteInitialPosToDevice( float4* initial_positions,
                                  unsigned int nparticles,
                                  unsigned int max_steps,
                                  unsigned int ndevices,
                                  unsigned int device_num
                                );

    void DoubleBufferInit(  unsigned int particle_interval_size,
                            unsigned int step_interval_size
                          );
    //
    // Reduction
    //

    void ReduceInit(  unsigned int particles_per,
                      std::string reduction_style); //ran once only.
    void Reduce();

    //
    // Interpolation
    //

    void Interpolate();


  private:
    //
    // OpenCL Interface
    //
    cl::Context* ocl_context;

    cl::CommandQueue* ocl_cq;

    cl::Kernel* ptx_kernel;
    //
    // BedpostX Data
    //

    cl::Buffer f_samples_buffer;
    cl::Buffer phi_samples_buffer;
    cl::Buffer theta_samples_buffer;

    unsigned int samples_buffer_size;
    unsigned int sample_nx, sample_ny, sample_nz, sample_ns;

    //
    // Output Data
    //

    unsigned int n_particles;
    unsigned int max_steps;

    // TODO @STEVE:  Some of this stuff will be GPU memory limited
    // figure out which and how

    unsigned int section_size;
    unsigned int step_size;
    unsigned int particles_size;

    cl::Buffer particle_paths_buffer;
    cl::Buffer particle_steps_taken_buffer;

    cl::Buffer particle_done_buffer;

    // size (Total Particles)/numDevices * (sizeof(float4))

    //
    // These are the "double buffer" objects
    //
    // Might have to rethink this within scope of mutexing, if
    // there is a common root/owner object, then access through that
    // may initiate a race condition.
    //
    std::vector<cl::Buffer> compute_index_buffers;

    std::vector< unsigned int > particle_indeces_left;
    std::vector< bool > particle_complete;
    // Vector of max size (N/2 ) , where n is the total number
    // of particles
    std::vector< std::vector<unsigned int> > particle_todo;
    // NDRange of current pair of enqueueNDRangeKernel
    std::vector<unsigned int> todo_range;


    // may want to just re-compute element # based off position
    // INSIDE KERNEL rather than store.
    //std::vector<cl::Buffer> compute_elements;
    //std::vector< std::vector<unsigned int> > particle_elements;



    // which half of particle_indeces/particle_complete needs to be
    // interpolated next (either 0, or 1)
    // TODO: Make sure there are no access conflicts within multithread
    // scheme (e.g.  go to interpolate second half, but reduction
    // method accidentally changed target_section to "0" again.
    unsigned int target_section;
    // this might need a mutex

    bool interpolation_complete;
    // false until there are zero particle paths left to compute.
};

#endif

//EOF