/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

/* prngmethods.cl
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


// TODO
// Lets rewrite these PRNG functions in a more readable/efficient manner
// with the added bonus that we can rename everything and not
// reference any external influence
//



//*********************************************************************
//
// PRNG Helper Functions
//
//*********************************************************************

// from:
//    https://developer.nvidia.com/content/gpu-gems-3-chapter-37-
//    efficient-random-number-generation-and-application-using-cuda

// S1, S2, S3, and M are all constants, and z is part of the
// private per-thread generator state.
unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)
{
  unsigned b=(((z << S1) ^ z) >> S2);
  return z = (((z & M) << S3) ^ b);
}

// A and C are constants
 unsigned LCGStep(unsigned &z, unsigned A, unsigned C)
{
  return z=(A*z+C);
}


unsigned z1, z2, z3, z4;
float HybridTaus()
{
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
   return 2.3283064365387e-10 * (                 // Periods
    TausStep(z1, 13, 19, 12, 4294967294UL) ^      // p1=2^31-1
    TausStep(z2, 2, 25, 4, 4294967288UL) ^       // p2=2^30-1
    TausStep(z3, 3, 11, 17, 4294967280UL) ^      // p3=2^28-1
    LCGStep(z4, 1664525, 1013904223UL)           // p4=2^32
   );
}

float2 BoxMuller()
{
  float u0=HybridTaus (), u1=HybridTaus ();
  float r=sqrt(-2 log(u0));
  float theta=2*PI*u1;
  return make_float2(r*sin(theta),r*cos(theta));
}

//EOF