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
  float4 dr;
} __attribute__((aligned(64)));

struct particle_attrs
{
  float4 brain_mask_dim;
  int steps_per_kernel;
  int max_steps;
  int particles_per_side;
  uint pdf_mask_entries;
  uint sample_nx;
  uint sample_ny;
  uint sample_nz;
  uint num_samples;
  float curvature_threshold;
  uint n_waypoint_masks;
  float step_length;  // TODO(steve) figure out differences in steplength (why div 2)
  uint lx;  // Loopcheck sizes
  uint ly;
  uint lz;
} __attribute__((aligned(16)));

__kernel void OclPtxInterpolate(
  struct particle_attrs attrs,  /* RO */
  __global struct particle_data *state,  /* RW */

  // Debugging info
  __global float4 *particle_paths, //RW
  __global ushort *particle_steps, //RW

  // Output
  __global ushort *particle_done, //RW
  __global uint *particle_pdfs, //RW
  __global ushort *particle_waypoints, //W
  __global ushort *particle_exclusion, //W
  __global float4 *particle_loopcheck_lastdir, //RW

  // Global Data
  __global float *f_samples, //R
  __global float *phi_samples, //R
  __global float *theta_samples, //R
  __global ushort *brain_mask, //R
  __global ushort *waypoint_masks,  //R
  __global ushort *termination_mask,  //R
  __global ushort *exclusion_mask //R
)
{
  uint glid = get_global_id(0);
  
  uint path_index;

#ifdef WAYPOINTS
  uint mask_size = attrs.sample_nx * attrs.sample_ny * attrs.sample_nz;
#endif
    
  uint step;
  
  uint3 current_select_vertex;
  float3 volume_fraction;
  
  uint diffusion_index;
  uint sample;
  
  // last location of particle
  float4 particle_pos;
  particle_pos.s3 = 0.0;

  float4 temp_pos = state[glid].position; //dx, dy, dz
  float4 new_dr = (float4) (0.0f);

#ifdef EULER_STREAMLINE
  float4 dr2 = (float4) (0.0f);
#endif
  
  float xmin, xmax, ymin, ymax, zmin, zmax;
  xmin = 0.0; ymin = 0.0; zmin = 0.0;
  xmax = attrs.sample_nx*1.0; ymax = attrs.sample_ny*1.0; zmax = attrs.sample_nz*1.0;
  
  float f, phi, theta;
  float jump_dot;

#ifdef PRNG
  ulong rng_output;
  float vol_frac;
#endif
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

  uint3 loopcheck_voxel;

  uint loopcheck_dir_size = attrs.lx * attrs.ly * attrs.lz;

  uint loopcheck_index;
  float4 last_loopcheck_dr;
  float loopcheck_product;

  for (step = 0; step < attrs.steps_per_kernel; ++step)
  {
    particle_pos = state[glid].position;
    // calculate current index in diffusion space
    current_select_vertex.s0 = floor(particle_pos.s0);
    current_select_vertex.s1 = floor(particle_pos.s1);
    current_select_vertex.s2 = floor(particle_pos.s2);

    volume_fraction.s0 = particle_pos.s0 - (float) current_select_vertex.s0;
    volume_fraction.s1 = particle_pos.s1 - (float) current_select_vertex.s1;
    volume_fraction.s2 = particle_pos.s2 - (float) current_select_vertex.s2;
    //
    // Pick Sample
    //

    // TODO @STEVE
    // Implement multiple direction selection if more than one direction of
    // bedpost data being used
    //
    rng_output = Rand(&(state[glid].rng));
    sample = rng_output % attrs.num_samples;

    // Volume Fraction Selection

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
    
    new_dr.s0 = cos( phi ) * sin( theta );
    new_dr.s1 = sin( phi ) * sin( theta );
    new_dr.s2 = cos( theta );
    
    // jump (aligns direction to prevent zig-zagging)
    jump_dot = new_dr.s0 * state[glid].dr.s0
             + new_dr.s1 * state[glid].dr.s1
             + new_dr.s2 * state[glid].dr.s2;

    if (jump_dot < 0.0 )
    {
      new_dr = new_dr*-1.0;
    }

    new_dr = new_dr * attrs.step_length / attrs.brain_mask_dim;

    // update particle position
    temp_pos += new_dr;

#ifdef EULER_STREAMLINE
    current_select_vertex.s0 = floor(temp_pos.s0);
    current_select_vertex.s1 = floor(temp_pos.s1);
    current_select_vertex.s2 = floor(temp_pos.s2);

    volume_fraction.s0 = temp_pos.s0 - (float) current_select_vertex.s0;
    volume_fraction.s1 = temp_pos.s1 - (float) current_select_vertex.s1;
    volume_fraction.s2 = temp_pos.s2 - (float) current_select_vertex.s2;
    //
    // Pick Sample
    //
    rng_output = Rand(&(state[glid].rng));
    sample = rng_output % attrs.num_samples;

    // Volume Fraction Selection
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
    
    dr2.s0 = cos( phi ) * sin( theta );
    dr2.s1 = sin( phi ) * sin( theta );
    dr2.s2 = cos( theta );

    //
    // jump (aligns direction to prevent zig-zagging)
    //
    jump_dot = dr2.s0*state[glid].dr.s0 +
                dr2.s1*state[glid].dr.s1 +
                  dr2.s2*state[glid].dr.s2;

    if (jump_dot < 0.0 )
    {
      dr2 = dr2*-1.0;
    }

    dr2 = dr2*attrs.step_length / attrs.brain_mask_dim;

    new_dr = 0.5*(new_dr + dr2);
#endif
    // update particle position
    temp_pos = particle_pos + new_dr;

    //
    // Curvature Threshold
    //

    // normalize for curvature threshold
    new_dr = new_dr /
             (new_dr.s0 * new_dr.s0
            + new_dr.s1 * new_dr.s1
            + new_dr.s2 * new_dr.s2);
    
    jump_dot = new_dr.s0 * state[glid].dr.s0
             + new_dr.s1 * state[glid].dr.s1
             + new_dr.s2 * state[glid].dr.s2;

    if (particle_steps[glid] > 1 && jump_dot < attrs.curvature_threshold)
    {
      particle_done[glid] = 2;
      if (0 == step)
        particle_steps[glid] = 0;
    }

    //
    // Complete out of bounds test (just in case)
    //
    if (temp_pos.s0 > xmax || xmin > temp_pos.s0 ||
      temp_pos.s1 > ymax || ymin > temp_pos.s1 ||
        temp_pos.s2 > zmax || zmin > temp_pos.s2)
    {
      particle_done[glid] = 4;
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
      particle_done[glid] = 5;
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

#ifdef LOOPCHECK
  loopcheck_voxel.s0 = round(temp_pos.s0)/5;
  loopcheck_voxel.s1 = round(temp_pos.s1)/5;
  loopcheck_voxel.s2 = round(temp_pos.s2)/5;

  loopcheck_index = loopcheck_voxel.s0*(attrs.ly*attrs.lz) +
    loopcheck_voxel.s1*attrs.lz + loopcheck_voxel.s2;

  last_loopcheck_dr =
      particle_loopcheck_lastdir[glid*loopcheck_dir_size +
        loopcheck_index];

  loopcheck_product = last_loopcheck_dr.s0*new_dr.s0 +
    last_loopcheck_dr.s1*new_dr.s1 + last_loopcheck_dr.s2*new_dr.s2;

  if (loopcheck_product < 0) // loopcheck break
  {
    particle_done[glid] = 3;
    if (0 == step)
      particle_steps[glid] = 0;
    break;
  }

  particle_loopcheck_lastdir[glid*loopcheck_dir_size +
    loopcheck_index] = new_dr;

#endif  // LOOPCHECK

    // update last flow vector
    state[glid].dr = new_dr;

    // add to particle paths
    path_index = glid * attrs.steps_per_kernel + step;
    particle_paths[path_index] = temp_pos;
  
    // update particle pdf
    vertex_num =
      round(temp_pos.s0)*round(temp_pos.s1)*round(temp_pos.s2);
    entry_num = vertex_num / 32;
    shift_num = 31 - (vertex_num % 32);

   particle_entry =
     particle_pdfs[glid * attrs.pdf_mask_entries + entry_num];
   particle_pdfs[glid * attrs.pdf_mask_entries + entry_num] =
     particle_entry | (0x00000001 << shift_num);
    
    if (particle_steps[glid] + 1 == attrs.max_steps){
      particle_done[glid] = 1;
      if (0 == step)
        particle_steps[glid] = 0;
    }

    state[glid].position = temp_pos;
    // update step location
    if (!particle_done[glid])
    {
      particle_steps[glid] += 1;
    }
  }
}
