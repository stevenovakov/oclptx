/*  Copyright 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

struct particle_data
{
  ulong value;
} __attribute__((aligned(8)));

struct particle_attrs
{
  int num_steps;
  int particles_per_side;
} __attribute__((aligned(8)));

__kernel void Collatz(
  struct particle_attrs attrs,  /* RO */
  __global struct particle_data *state,  /* RW */
  __global ushort *complete,
  __global ulong *path_output)
{
  int step;
  int index;
  int glid = get_global_id(0);

  for (step = 0; step < attrs.num_steps; ++step)
  {
    index = glid * attrs.num_steps + step;
    path_output[index] = state[glid].value;

    if (state[glid].value & 1)
      state[glid].value = state[glid].value * 3 + 1;
    else
      state[glid].value = state[glid].value >> 1;

    if (4 == state[glid].value ||
        2 == state[glid].value ||
        1 == state[glid].value)
      complete[glid] = true;
    else
      complete[glid] = false;
  }
}
