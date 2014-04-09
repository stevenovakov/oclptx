// Copyright 2014
//  Afshin Haidari
//  Steve Novakov
//  Jeff Taylor
//
// Summing kernel.  This kernel adds up the paths tracked by finished particles
// into a single global buffer.  Each thread takes on a small piece of physical
// space, iterating through all the particles to see if they've crossed it.

#include "attrs.h"

// uint32_t list of particle_pdfs
//          b=0             b=31
// [ ... ], [0 1 2 3 .....  31], [ ... ]
//
// voxel = glid*num_entries_per_particle + b
//
// Note, unpacking to a global NDRange(entries, 32)
// was attempted, and was ~100x SLOWER than the 1D work range


// TODO(jeff): Clear the local_pdf's when we finish with them, so host doesn't
// have to copy over zeroes.
__kernel void PdfSum(
  struct particle_attrs attrs,  /* RO */
  uint side,

  // Particle data
  __global ushort* particles_done,
  __global uint* particle_pdfs,
  __global ushort* particle_waypoints,
  __global ushort* particle_exclusion,
  __global ushort* particle_steps,

  // Output
  __global uint* global_pdf
)
{
  uint glid = get_global_id(0);

  uint root_vertex = glid * 32;
  uint entries_per_particle =
    (attrs.sample_nx * attrs.sample_ny * attrs.sample_nz / 32) + 1;
  uint particle_check = 0;
  uint particle_entry = 0;
  uint to_total;

  ushort steps_taken;

  __local uint running_total[32];

  for (int b = 0; b < 32; b++)
  {
    running_total[b] = 0;
  }

  int first_particle = side * attrs.particles_per_side;
  int last_particle = (1 + side) * attrs.particles_per_side;
  for (int p = first_particle; p < last_particle; ++p)
  {
    steps_taken = particle_steps[p];
    if (steps_taken < attrs.min_steps)
      continue;

#if EXCLUSION
    particle_check = particle_exclusion[p];

    if (particle_check > 0)
      continue;
#endif  // EXCLUSION

#if WAYPOINTS
    particle_check = 1;
    for (uint w = 0; w < attrs.n_waypoint_masks; w++)
    {
#if WAYAND
      particle_check &= particle_waypoints[p*attrs.n_waypoint_masks + w];
#else  // WAYOR
      particle_check |= particle_waypoints[p*attrs.n_waypoint_masks + w];
#endif  // WAYAND
    }

    if (particle_check == 0)
      continue;
#endif  // WAYPOINTS

    particle_check = particles_done[p];

    if ((BREAK_INIT != particle_check)
     && (BREAK_INVALID != particle_check)
     && (particle_check > 0)
     && (STILL_FINISHED != particle_check))
    {
      particle_entry = particle_pdfs[p * entries_per_particle + glid];
      particle_pdfs[p * entries_per_particle + glid] = 0;
    }
    else
    {
      particle_pdfs[p * entries_per_particle + glid] = 0;
      continue;
    }

    for (int b = 0; b < 32; b++)
    {
      to_total = (particle_entry >> b) & 1;
      running_total[b] += to_total;
    }
  }

  uint vertex_num;
  for (uint b = 0; b < 32; b++)
  {
    vertex_num = root_vertex + b;
    global_pdf[vertex_num] += running_total[b];
  }
}
