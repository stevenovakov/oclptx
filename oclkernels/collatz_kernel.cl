/*  Copyright 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

#include "oclkernels/collatz_kernel.cl"

struct particle_data
{
  ulong value;
} __attribute__ ((aligned(8)));

struct particle_attrs
{
  int num_steps;
  int particles_per_side;
} __attribute__ ((aligned(8)));

__kernel void CollatzTest(
  __global struct particle_attrs attrs,  /* RO */
  __global struct particle_data *state,  /* RW */
  __global bool  *complete,
  __global ulong *path_output)
{
  int step;
  int index;
  int glid = get_global_id(0);

  for (step = 0; step < attrs.num_steps; ++step)
  {
    if (state[glid].value & 1)
      state[glid].value = state[glid].value * 3 + 1;
    else
      state[glid].value = state[glid].value >> 1;

    if (1 == state[glid].value)
      complete[glid] = true;

    index = glid * attrs.num_steps + step;
    path_output[index] = Rand(rng + glid);
  }
}

