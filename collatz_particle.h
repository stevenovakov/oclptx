// Collatz data required.

#ifndef COL_PARTICLE_H_
#define COL_PARTICLE_H_

#include "particle.h"

namespace particle
{
  struct particle_data
  {
    cl_ulong value;
  } __attribute__ ((aligned(8)));

  struct particle_attrs
  {
    cl_int num_steps;
    cl_int particles_per_side;
  } __attribute__ ((aligned(8)));

  struct particles
  {
    cl::Buffer *gpu_data;  // Type particle_data
    cl::Buffer *gpu_complete;  // Type cl_ushort array

    cl::Buffer *gpu_path;  // Type ulong

    OclEnv *env;
    struct particle_attrs attrs;
  };
}  // namespace particle

#endif  // COL_PARTICLE_H_
