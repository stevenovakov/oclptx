/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */
 
/* customtypes.h
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
#include <vector>
#include <thread>
#include <mutex>


#ifndef OCLPTX_CUSTOMTYPES_H_
#define OCLPTX_CUSTOMTYPES_H_


struct float3{
  float x, y, z;
};

struct float4{
  float x, y, z, t;
};

struct int3{
  float x, y, z;
};

struct int4{
  float x, y, z, t;
};


struct IntVolume
{
  std::vector<int4> vol;
  int nx, ny, nz;  
  // Access x, y, z:
  //                var_name.vol[z*(ny*nx) + y*8*nx + 8*x + v]
};

struct FloatVolume
{
  std::vector<float4> vol;
  int nx, ny, nz;
  // Access x, y, z:
  //                var_name.vol[z*(ny*nx) + y*8*nx + 8*x + v]
};

//struct MutexWrapper {
    //std::mutex m;
    //MutexWrapper() {}
    //MutexWrapper(MutexWrapper const&) {}
    //MutexWrapper& operator=(MutexWrapper const&) { return *this; }
//};

struct BedpostXData
{
  std::vector<float*> data;
  unsigned int nx, ny, nz;  // discrete dimensions of mesh
  unsigned int ns;          //number of samples
};
//
// Note on particle positions re:bedpostx mesh :
// if a particle is at x,y,z, can find nearest "root" vertex:
// (lowest x,y,z coordinate vertex), at X, Y, Z and find element #
// thusly: 
//
// elem = nssample*(X*nz*ny + Y*nz + Z)
//
// Device side, where there may be multiple directions included, simply
// multiply by the direction #, (from 0 to n-1)
// 

#endif