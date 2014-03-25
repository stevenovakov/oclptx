/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */
__kernel void PdfSum( __global uint* total_pdf,
                      __global uint* particle_pdfs,
                      __global uint* particles_done,
                      uint num_particles,
                      uint entries_per_particle,
                      uint nx,
                      uint ny,
                      uint nz
)
{
  uint entry_num = get_global_id(0);

  uint shift_num;
  uint vertex_num;
  uint root_vertex = entry_num * 32;
  uint b, p;
  uint particle_done =0;

  uint ny_nz = ny*nz;
  uint total_dim = nx*ny*nz;

  uint particle_entry = 0;

  __local uint running_total[32];
  uint to_total;

  for ( p = 0; p < num_particles; p++)
  {
    particle_done = particles_done[p];

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

      if (vertex_num > total_dim)
      {
        break;
      }

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
