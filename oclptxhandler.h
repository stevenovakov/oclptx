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
    OclPtxHandler();

    ~OclPtxHandler();

    //
    // I/O
    //

    std::vector<float4> GetParticlePaths();   // do at end

    //
    // OCL Initialization
    //

    void

    void WriteSamplesToDevice( float4 * f_data,
                                float4 * phi_data,
                                float4 * theta_data,
                                unsigned int offset
                              );
    // may want to compute offset beforehand in samplemanager,
    // can decide later.

    //
    // Reduction
    //

    void ReduceInit(std::string reduction_style); //ran once only.

    void Reduce();

    //
    // Interpolation
    //

    void Interpolate();


  private:
    //
    // Constant Data
    //

    cl::Buffer fsamples_buffer;
    cl::Buffer phisamples_buffer;
    cl::Buffer thetasamples_buffer;

    unsigned int samples_buffer_size;

    //
    // Variable Data
    //

    cl::Buffer particle_paths_buffer;
    // size (Total Particles)/numDevices * (sizeof(float4))

    //
    // These are the "double buffer" objects
    //
    // Might have to rethink this within scope of mutexing, if
    // there is a common root/owner object, then access through that
    // may initiate a race condition.
    //
    std::vector<cl::Buffer> compute_indices;

    // Vector of size   2x (N/2 ) , where n is the total number
    // of particles
    std::vector< std::vector<unsigned int> > particle_indeces;

    std::vector< std::vector<bool> > particle_complete;

    // may want to just re-compute element # based off position
    // INSIDE KERNEL rather than store.
    //std::vector<cl::Buffer> compute_elements;
    //std::vector< std::vector<unsigned int> > particle_elements;

    // NDRange of current pair of enqueueNDRangeKernel
    std::vector<unsigned int> compute_range;

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