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

    ~OclPtxHandler();


    //
    // Thread Management
    //

    void PTStart();

    void ThreadController();


    //
    // Interpolation
    //



  private:
    //
    // Various
    //

    //
    // Might have to rethink this within scope of mutexing, if
    // there is a common root/owner object, then access through that
    // may initiate a race condition.
    //

    // Vector of size   2x (N/2 ) , where n is the total number
    // of particles
    std::vector< std::vector<unsigned int> > particle_indeces;
    std::vector< std::vector<bool> > particle_complete;

    std::vector<unsigned int> compute_range;

    // which half of particle_indeces/particle_complete needs to be
    // interpolated next.
    // TODO: Make sure there are no access conflicts within multithread
    // scheme (e.g.  go to interpolate second half, but reduction
    // method accidentally changed target_section to "0" again.
    unsigned int target_section;
    // this might need a mutex

    bool interpolation_complete;
};

#endif

//EOF