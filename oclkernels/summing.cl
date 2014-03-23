/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */
__kernel void PdfSum( __global uint* total_pdf,
                      __global uint* particle_pdfs,
                      __global uint* particles_done,
                      uint num_particles,
                      uint entries_per_particle
)
{
  uint glidx = get_global_id(0);
  uint glidy = get_global_id(1);
  uint glidz = get_global_id(2);

  uint entry_num;
  uint shift_num;
  uint p;

  uint vertex_num = glidx*glidy*glidz;
  uint particle_entry;
  uint particle_done;

  uint running_total = 0;

  for ( p = 0; p < num_particles; p++)
  {
    particle_done = particles_done[p];
    if (particle_done > 0)
    {
      entry_num = vertex_num / 32;
      shift_num = (vertex_num % 32) - 1;

      particle_entry = particle_pdfs[p * entries_per_particle + entry_num];
      running_total += ((particle_entry >> shift_num) & 0x00000001);
    }
  }

  total_pdf[vertex_num] = running_total;
}
