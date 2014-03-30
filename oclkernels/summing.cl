/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

 //
 // uint32_t list of particle_pdfs
//           b=0             b=31
 // [ ... ], [0 1 2 3 .....  31], [ ... ]
 //
 // voxel = entry_num*num_entries_per_particle + b
 //
 // Note, unpacking to a global NDRange(entries, 32)
 // was attempted, and was ~100x SLOWER than the 1D work range
 //
__kernel void PdfSum( __global uint* total_pdf,
                      __global uint* particle_pdfs,
                      __global uint* particles_done,
                      uint num_particles,
                      uint entries_per_particle
#ifdef EXCLUSION
                      , __global uint* particle_exclusion
#endif
#ifdef WAYPOINTS
                      , __global uint* particle_waypoints,
                      uint num_waypts
#endif
)
{
  uint entry_num = get_global_id(0);

  uint vertex_num;
  uint root_vertex = entry_num * 32;
  uint p,b;
  uint particle_check = 0;

  uint particle_entry = 0;

  __local uint running_total[32];

  for (b = 0; b < 32; b++)
  {
    running_total[b] = 0;
  }

  uint to_total;

  for ( p = 0; p < num_particles; p++)
  {
#ifdef EXCLUSION
    particle_check = particle_exclusion[p];

    if (particle_check > 0)
      continue;
#endif

#ifdef WAYPOINTS
    particle_check = 1;
    for (uint w = 0; w < num_waypts; w++)
    {
      particle_check *= particle_waypoints[p*num_waypts + w];
    }

    if (particle_check == 0)
      continue;
#endif

    particle_check = particles_done[p];

    if (particle_done > 0)
    {
      particle_entry = particle_pdfs[p * entries_per_particle + entry_num];
    }
    else
    {
      continue;
    }

    for (b = 0; b < 32; b++)
    {
     vertex_num = root_vertex + b;

      to_total = (particle_entry >> (31 - b)) & 0x00000001;
      running_total[b] += to_total;
    }
  }

  for (b = 0; b < 32; b++)
  {
    vertex_num = root_vertex + b;
    total_pdf[vertex_num] = running_total[b];
  }
}
