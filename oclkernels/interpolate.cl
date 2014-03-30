/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */
 
#include "rng.cl"

//*********************************************************************
//
// Main Kernel
//
//*********************************************************************
//
// sample data
// Access x, y, z vertex:
//    index = x*(ny*nz*ns*ndir) + y*(nz*ns*ndir) + z*(ns*ndir) + s*ndir
//    here ndir = 1, and is not included
//

// TODO(jeff):
//  - Double check that completion logic is similar to collatz, abstract if
//  needed.
//  - Check that we add to the particle path in the right place.

struct particle_data
{
  rng_t rng; //RW
  float4 position;
} __attribute__((aligned(64)));

struct particle_attrs
{
  int steps_per_kernel;
  int max_steps;
  int particles_per_side;
  uint sample_nx;
  uint sample_ny;
  uint sample_nz;
  uint num_samples;
  float curvature_threshold;
  uint n_waypoint_masks;
} __attribute__((aligned(8)));

__kernel void OclPtxInterpolate(
  struct particle_attrs attrs,  /* RO */
  __global struct particle_data *state,  /* RW */

  // Debugging info
  __global float4* particle_paths, //RW
  __global ushort* particle_steps, //RW

  // Output
  __global ushort* particle_done, //RW
  __global uint* particle_pdfs, //RW
  __global ushort* particle_waypoints, //W
  __global ushort* particle_exclusion, //W

  // Global Data
  __global float* f_samples, //R
  __global float* phi_samples, //R
  __global float* theta_samples, //R
  __global ushort* brain_mask, //R
  __global ushort* waypoint_masks,  //R
  __global ushort* termination_mask,  //R
  __global ushort* exclusion_mask //R
)
{
  uint glid = get_global_id(0);
  
  uint path_index;

#ifdef WAYPOINTS
  uint mask_size = attrs.sample_nx * attrs.sample_ny * attrs.sample_nz;
#endif
    
  uint step;

  uint pdf_entries_per_particle = (attrs.sample_nx*attrs.sample_ny*attrs.sample_nz / 32) + 1;
  
  uint3 current_select_vertex;

  float3 volume_fraction;
  
  uint diffusion_index;
  uint sample;
  
  // last location of particle
  float4 particle_pos = state[glid].position;
  particle_pos.s3 = 0.0;

  float4 temp_pos = (float4) (0.0f); //dx, dy, dz
  float4 dr = (float4) (0.0f);
  float4 last_dr = (float4) (0.0f);
  
  float xmin, xmax, ymin, ymax, zmin, zmax;
  xmin = 0.0; ymin = 0.0; zmin = 0.0;
  xmax = attrs.sample_nx*1.0; ymax = attrs.sample_ny*1.0; zmax = attrs.sample_nz*1.0;
  
  float f, phi, theta;
  float jump_dot;

  ulong rng_output;
  float vol_frac;

  uint mask_index;
  //unsigned int termination_mask_index;
  ushort bounds_test;

  uint vertex_num;
  uint entry_num;
  uint shift_num;
  uint particle_entry;

  // No new valid data.  Likely the host is out of data.  We need to avoid
  // screwing it up.
  if (particle_done[glid])
    particle_steps[glid] = 0;

  for (step = 0; step < attrs.steps_per_kernel; ++step)
  {
    // calculate current index in diffusion space
    current_select_vertex.s0 = floor(particle_pos.s0);
    current_select_vertex.s1 = floor(particle_pos.s1);
    current_select_vertex.s2 = floor(particle_pos.s2);

    volume_fraction.s0 = particle_pos.s0 - (float) current_select_vertex.s0;
    volume_fraction.s1 = particle_pos.s1 - (float) current_select_vertex.s1;
    volume_fraction.s2 = particle_pos.s2 - (float) current_select_vertex.s2;

#if EULER_STREAMLINE
    // not yet!
    //
    // TODO @STEVE
    // Maybe Implement a function that encapsulates "stepping" so that
    // we don't have to duplicate code
    //
#else
    // Pick Sample
    //
    // TODO @STEVE
    // Implement multiple direction selection if more than one direction of
    // bedpost data being used
    //
    rng_output = Rand(&(state[glid].rng));
    sample = rng_output % attrs.num_samples;
    // Volume Fraction Selection
    //
    // Steve; Should I convert rng_output to float before compare? Ask jeff...
    //
    rng_output = Rand(&(state[glid].rng));
    vol_frac = volume_fraction.s0 * kRandMax;

    if (rng_output > vol_frac)
    {
      current_select_vertex.s0 += 1;
    }

    rng_output = Rand(&(state[glid].rng));
    vol_frac = volume_fraction.s1 * kRandMax;

    if (rng_output > vol_frac)
    {
      current_select_vertex.s1 += 1;
    }

    rng_output = Rand(&(state[glid].rng));
    vol_frac = volume_fraction.s2 * kRandMax;

    if (rng_output > vol_frac)
    {
      current_select_vertex.s2 += 1;
    }
    // pick flow vertex
    diffusion_index =
      sample*(attrs.sample_nz*attrs.sample_ny*attrs.sample_nx)+
      current_select_vertex.s0*(attrs.sample_nz*attrs.sample_ny) +
      current_select_vertex.s1*(attrs.sample_nz) +
      current_select_vertex.s2;
    
    // find next step location
    f = f_samples[diffusion_index];
    theta = theta_samples[diffusion_index];
    phi = phi_samples[diffusion_index];
    
    dr.s0 = cos( phi ) * sin( theta );
    dr.s1 = sin( phi ) * sin( theta );
    dr.s2 = cos( theta );
    
    // alternates initial direction of particle set
    if( particle_steps[glid] == 0 && glid%2 == 0)
      dr = dr*-1.0;

    //
    // jump (aligns direction to prevent zig-zagging)
    //
    
    jump_dot = dr.s0*last_dr.s0 + dr.s1*last_dr.s1 +
      dr.s2*last_dr.s2;

    if (jump_dot < 0.0 )
    {
      dr = dr*-1.0;
      jump_dot = dr.s0*last_dr.s0 + dr.s1*last_dr.s1 +
      dr.s2*last_dr.s2;
    }
    
    //
    // Curvature Threshold
    //
    
    if (particle_steps[glid] > 1 && jump_dot < attrs.curvature_threshold)
    {
      particle_done[glid] = 1;
      if (0 == step)
        particle_steps[glid] = 0;
    }

    // update last flow vector
    last_dr = dr;

    dr = dr*0.25; // implement actual step length argument later
#endif
    // update particle position
    temp_pos = particle_pos + dr;

    //
    // Complete out of bounds test (just in case)
    //
    if (temp_pos.s0 > xmax || xmin > temp_pos.s0 ||
      temp_pos.s1 > ymax || ymin > temp_pos.s1 ||
        temp_pos.s2 > zmax || zmin > temp_pos.s2)
    {
      particle_done[glid] = 1;
      if (0 == step)
        particle_steps[glid] = 0;
    }
    //
    // Brain Mask Test - Checks NEAREST vertex.
    //
    mask_index =
      round(temp_pos.s0)*(attrs.sample_nz*attrs.sample_ny) +
        round(temp_pos.s1)*(attrs.sample_nz) + round(temp_pos.s2);

    bounds_test = brain_mask[mask_index];
    if (bounds_test == 0)
    {
      particle_done[glid] = 1;
      if (0 == step)
        particle_steps[glid] = 0;
    }

#ifdef TERMINATION
    bounds_test = termination_mask[mask_index];
    if (bounds_test == 0)
    {
      particle_done[glid] = 1;
      if (0 == step)
        particle_steps[glid] = 0;
      break;
    }
#endif  // TERMINATION

#ifdef EXCLUSION
    bounds_test = exclusion_mask[mask_index];
    if (bounds_test == 0)
    {
      particle_exclusion[glid] = 1;
      particle_done[glid] = 1;
      if (0 == step)
        particle_steps[glid] = 0;
      break;
    }
#endif  // EXCLUSION

#ifdef WAYPOINTS
    for (uint w = 0; w < attrs.n_waypoint_masks; w++)
    {
      bounds_test = waypoint_masks[w*mask_size + mask_index];
      if (bounds_test > 0)
        particle_waypoints[glid*attrs.n_waypoint_masks + w] |= 1;
    }
#endif  // WAYPOINTS

    // update current location
    particle_pos = temp_pos;

    // add to particle paths
    path_index = glid * attrs.steps_per_kernel + step;
    particle_paths[path_index] = particle_pos;
  
    // update particle pdf
    vertex_num =
      round(particle_pos.s0)*round(particle_pos.s1)*round(particle_pos.s2);
    entry_num = vertex_num / 32;
    shift_num = 31 - (vertex_num % 32);

//    particle_entry =
//      particle_pdfs[glid * pdf_entries_per_particle + entry_num];
//    particle_pdfs[glid * pdf_entries_per_particle + entry_num] =
//      particle_entry | (0x00000001 << shift_num);
    
    if (particle_steps[glid] + 1 == attrs.max_steps){
      particle_done[glid] = 1;
      if (0 == step)
        particle_steps[glid] = 0;
    }

    // update step location
    if (!particle_done[glid])
    {
      particle_steps[glid] += 1;
    }
  }
}


//EOF
