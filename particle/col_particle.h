// Collatz data required.

#ifndef COL_PARTICLE_H_
#define COL_PARTICLE_H_

#include "particle/particle.h"

namespace particle
{
  struct particles
  {
    cl::Buffer gpu_particle_data *gpu_data;  // Type particle_data
    cl::Buffer *gpu_completion;  // Type cl_bool array

    cl::Buffer *path_data;  // Type ulong

    OclEnv *env;
    struct particle_attrs attrs;
    int num_particles;
  };

  struct particle_data
  {
    cl_ulong value;
  } __attribute__ ((aligned(8)));

  struct particle_attrs
  {
    int num_steps;
  };
}  // namespace particle

#endif  // COL_PARTICLE_H_
