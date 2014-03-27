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
                    cl::Kernel* ptx_kernel,
                    cl::Kernel* sum_kernel,
                    float curv_thresh,
                    EnvironmentData * env
                  );

    ~OclPtxHandler();

    //
    // Set/Get
    //

    void ParticlePathsToFile(std::string filename = "");   // do at end

    bool IsFinished(){ return this->interpolation_complete; };
    
    cl_ulong GpuMemUsed();

    void GetPdfData(uint32_t* container);

    //
    // OCL Initialization
    //

    // May want overarching initialize() that simply wraps everything
    // below to be called by std::thread
    //
    // void Initialize()

    // void WriteSamplesToDevice(  const BedpostXData* f_data,
    //                             const BedpostXData* phi_data,
    //                             const BedpostXData* theta_data,
    //                             unsigned int num_directions,
    //                             const unsigned short int* brain_mask
    //                           );
    // may want to compute offset beforehand in samplemanager,
    // can decide later.

    void WriteInitialPosToDevice( const float4* initial_positions,
                                  unsigned int nparticles,
                                  unsigned int max_steps,
                                  unsigned int ndevices,
                                  unsigned int device_num
                                );
                                
    void SingleBufferInit();

    void PrngInit();

    void BlockCq(){this->ocl_cq->finish();};
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

    void PdfSum();


  private:
    //
    // OpenCL Interface
    //
    cl::Context* ocl_context;

    cl::CommandQueue* ocl_cq;

    cl::Kernel* ptx_kernel;
    cl::Kernel* sum_kernel;

    //
    // Environment (Includes Pointers to BedpostX Data)
    //

    EnvironmentData * env_dat;

    cl_ulong total_gpu_mem_used;

    //
    // Output Data
    //

    unsigned int n_particles;
    unsigned int max_steps;
    unsigned int particle_path_size;
    
    //
    // Check Data
    //

    float curvature_threshold;

    // TODO @STEVE:  Some of this stuff will be GPU memory limited
    // figure out which and how
    
    // TODO @STEVE: May have to introduce additional "status" buffers
    // for things like waypoint/stop mask ,etc to check which particles
    // are desireable.

    unsigned int section_size;

    cl::Buffer particle_paths_buffer;
    cl::Buffer particle_steps_taken_buffer;
    cl::Buffer particle_rng_buffer;
    cl::Buffer global_pdf_buffer;
    cl::Buffer particles_pdf_buffer;

    cl::Buffer particle_done_buffer;
    //cl:Buffer particle_waypoint_buffer;
    
    unsigned int particles_mem_size;
    unsigned int particle_uint_mem_size;
    // size (Total Particles)/numDevices * (sizeof(float4))

    //
    // These are the "single buffer" objects
    //
    cl::Buffer compute_index_buffer;

    std::vector< unsigned int > particle_indeces_left;
    std::vector< unsigned int > particle_complete;
    // Vector of max size (N/2 ) , where n is the total number
    // of particles
    std::vector< std::vector<unsigned int> > particle_todo;
    // NDRange of current pair of enqueueNDRangeKernel
    unsigned int todo_range;

    bool interpolation_complete;
    // false until there are zero particle paths left to compute.
    // operations should finalize after NEXT interpolate call.
};

#endif

//EOF