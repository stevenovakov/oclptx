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

  // Debugging info
  __global ulong *path_output)
  __global ushort *num_steps,

  // Output
  __global ushort *complete,
{
  int step;
  int index;
  int glid = get_global_id(0);

  // Signal to the dumping script that no new particle has been placed in this
  // thread.
  if (complete[glid])
    num_steps[glid] = 0;

  for (step = 0; step < attrs.num_steps; ++step)
  {
    if (state[glid].value & 1)
      state[glid].value = state[glid].value * 3 + 1;
    else
      state[glid].value = state[glid].value >> 1;

    if (1 == state[glid].value || complete[glid])
    {
      complete[glid] = true;
      if (0 == step)  // Stop counting steps.
        num_steps[glid] = 0;
    }

    if (!complete[glid])
      num_steps[glid] += 1;

    index = glid * attrs.num_steps + step;
    path_output[index] = state[glid].value;
  }
}

