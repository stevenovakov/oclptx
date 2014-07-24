// Copyright (C) 2014
//  Afshin Haidari
//  Steve Novakov
//  Jeff Taylor
//
//  This kernel traces out the paths taken by particles.  Each thread acts as a
//  single particle.
 
#include "attrs.h"
#include "rng.h"

// sample data
// Access x, y, z vertex:
//    index = x*(ny*nz*ns*ndir) + y*(nz*ns*ndir) + z*(ns*ndir) + s*ndir
//    here ndir = 1, and is not included
//

__kernel void OclPtxInterpolate(
  struct particle_attrs attrs,  /* RO */
  __global struct particle_data *state,  /* RW */

  // Debugging info
  __global float3 *particle_paths, //RW
  __global ushort *particle_steps, //RW

  // Output
  __global ushort *particle_done, //RW
  __global uint *particle_pdfs, //RW
  __global ushort *particle_waypoints, //W
  __global ushort *particle_exclusion, //W
  __global float3 *particle_loopcheck_lastdir, //RW

  // Global Data
  __global float *f_samples, //R
  __global float *phi_samples, //R
  __global float *theta_samples, //R
  __global float *f_samples_2,  //R
  __global float *phi_samples_2,  //R
  __global float *theta_samples_2,  //R
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
  float3 particle_pos;

  float3 temp_pos = state[glid].position; //dx, dy, dz
  float3 new_dr = (float3) (0.0f);

#ifdef EULER_STREAMLINE
  float3 dr2 = (float3) (0.0f);
#endif
  
  float3 min = (float3) (0.0f);
  float3 max = (float3) (attrs.sample_nx * 1.0,
                         attrs.sample_ny * 1.0,
                         attrs.sample_nz * 1.0);
  
  float f, phi, theta;

  ulong3 rng_output;
  float3 vol_frac;
  uint mask_index;
  //unsigned int termination_mask_index;
  ushort bounds_test;

  uint vertex_num;
  uint entry_num;
  uint shift_num;

  // No new valid data.  Likely the host is out of data.  We need to avoid
  // screwing it up.

  if (particle_done[glid])
  {
    particle_done[glid] = STILL_FINISHED;
    particle_steps[glid] = 0;
    return;
  }

#ifdef LOOPCHECK
  uint3 loopcheck_voxel;

  uint loopcheck_dir_size = attrs.lx * attrs.ly * attrs.lz;

  uint loopcheck_index;
  float3 last_loopcheck_dr;
  float loopcheck_product;
#endif // LOOPCHECK

  global float* target_f = f_samples;
  global float* target_theta = theta_samples;
  global float* target_phi = phi_samples;

  for (step = 0; step < attrs.steps_per_kernel; ++step)
  {
    particle_pos = state[glid].position;
    // calculate current index in diffusion space
    current_select_vertex = convert_uint3(floor(particle_pos));

    volume_fraction = particle_pos - convert_float3(current_select_vertex);

    // Pick Sample
    sample = Rand(&(state[glid].rng)) % attrs.num_samples;

    // Volume Fraction Selection
    rng_output = (ulong3) (Rand(&(state[glid].rng)),
                           Rand(&(state[glid].rng)),
                           Rand(&(state[glid].rng)));

    vol_frac = volume_fraction * kRandMax;

    current_select_vertex += (convert_float3(rng_output) > vol_frac)? 1: 0;

    // pick flow vertex
    diffusion_index =
      sample*(attrs.sample_nz*attrs.sample_ny*attrs.sample_nx)+
      current_select_vertex.s0*(attrs.sample_nz*attrs.sample_ny) +
      current_select_vertex.s1*(attrs.sample_nz) +
      current_select_vertex.s2;

#ifdef ANISOTROPIC
    f = target_f[diffusion_index];
    rng_output = Rand(&(state[glid].rng));
    if (f*kRandMax < rng_output)
    {
      particle_done[glid] = ANISO_BREAK;
      if (0 == step)
        particle_steps[glid] = 0;
      break;
    }
#endif // ANISOTROPIC

    // find next step location
    theta = target_theta[diffusion_index];
    phi = target_phi[diffusion_index];
    
    new_dr.s0 = cos( phi ) * sin( theta );
    new_dr.s1 = sin( phi ) * sin( theta );
    new_dr.s2 = cos( theta );
    
    // jump (aligns direction to prevent zig-zagging)
    if (dot(new_dr, state[glid].dr) < 0.0 )
      new_dr *= -1.;

    new_dr = new_dr / attrs.brain_mask_dim;
    new_dr = new_dr * attrs.step_length;

#ifdef EULER_STREAMLINE
    // update particle position
    temp_pos += new_dr;

    current_select_vertex = convert_uint3(floor(temp_pos));
    volume_fraction = temp_pos - convert_float3(current_select_vertex);
    // Pick Sample
    sample = Rand(&(state[glid].rng)) % attrs.num_samples;

    // Volume Fraction Selection
    rng_output = (ulong3) (Rand(&(state[glid].rng)),
                           Rand(&(state[glid].rng)),
                           Rand(&(state[glid].rng)));

    vol_frac = volume_fraction * kRandMax;

    current_select_vertex += (convert_float3(rng_output) > vol_frac)? 1: 0;

    // pick flow vertex
    diffusion_index =
      sample*(attrs.sample_nz*attrs.sample_ny*attrs.sample_nx)+
      current_select_vertex.s0*(attrs.sample_nz*attrs.sample_ny) +
      current_select_vertex.s1*(attrs.sample_nz) +
      current_select_vertex.s2;

#ifdef ANISOTROPIC
    f = target_f[diffusion_index];
    if (f * kRandMax < Rand(&(state[glid].rng)))
    {
      particle_done[glid] = ANISO_BREAK;
      if (0 == step)
        particle_steps[glid] = 0;
      break;
    }
#endif // ANISOTROPIC

    // find next step location
    theta = target_theta[diffusion_index];
    phi = target_phi[diffusion_index];
    
    dr2.s0 = cos( phi ) * sin( theta );
    dr2.s1 = sin( phi ) * sin( theta );
    dr2.s2 = cos( theta );

    // jump (aligns direction to prevent zig-zagging)
    if (dot(dr2, state[glid].dr) < 0.0 )
      dr2 *= -1.;

    dr2 = dr2 / attrs.brain_mask_dim;
    dr2 = dr2 * attrs.step_length;

    new_dr = 0.5*(new_dr + dr2);
#endif  /* EULER_STREAMLINE */
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

    if (particle_steps[glid] > 1 
      && dot(new_dr, state[glid].dr) < attrs.curvature_threshold)
    {
      particle_done[glid] = BREAK_CURV;
      if (0 == step)
        particle_steps[glid] = 0;
      break;
    }

    // Out of bounds?
    if (any(temp_pos > max || min > temp_pos))
    {
      particle_done[glid] = BREAK_INVALID;
      if (0 == step)
        particle_steps[glid] = 0;
      break;
    }

    // Brain Mask Test - Checks NEAREST vertex.
    mask_index =
      round(temp_pos.s0)*(attrs.sample_nz*attrs.sample_ny) +
        round(temp_pos.s1)*(attrs.sample_nz) + round(temp_pos.s2);

    bounds_test = brain_mask[mask_index];
    if (bounds_test == 0)
    {
      particle_done[glid] = BREAK_BRAIN_MASK;
      if (0 == step)
        particle_steps[glid] = 0;
      break;
    }

#ifdef TERMINATION
    bounds_test = termination_mask[mask_index];
    if (bounds_test == 1)
    {
      particle_done[glid] = BREAK_TERM;
      if (0 == step)
        particle_steps[glid] = 0;
      break;
    }
#endif  // TERMINATION

#ifdef EXCLUSION
    bounds_test = exclusion_mask[mask_index];
    if (bounds_test == 1)
    {
      particle_exclusion[glid] = 1;
      particle_done[glid] = BREAK_EXCLUSION;
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
    particle_done[glid] = BREAK_LOOPCHECK;
    if (0 == step)
      particle_steps[glid] = 0;
    break;
  }

  particle_loopcheck_lastdir[glid*loopcheck_dir_size +
    loopcheck_index] = new_dr;

#endif  // LOOPCHECK

    // update step location
    state[glid].position = temp_pos;

    // update last flow vector
    state[glid].dr = new_dr;

    // add to particle paths
    path_index = glid * attrs.steps_per_kernel + step;

    if (particle_paths)
      particle_paths[path_index] = temp_pos;
  
    // update particle pdf
    vertex_num = floor(temp_pos.s0) * attrs.sample_ny * attrs.sample_nz
               + floor(temp_pos.s1) * attrs.sample_nz
               + floor(temp_pos.s2);

    entry_num = vertex_num / 32;
    shift_num = (vertex_num % 32);

    uint entries_per_particle =
      (attrs.sample_nx * attrs.sample_ny * attrs.sample_nz / 32) + 1;

    particle_pdfs[glid*entries_per_particle + entry_num] |= (1 << shift_num);
    
    if (particle_steps[glid] + 1 == attrs.max_steps){
      particle_done[glid] = BREAK_MAXSTEPS;
      if (0 == step)
        particle_steps[glid] = 0;
      break;
    }

    if (!particle_done[glid])
    {
      particle_steps[glid] += 1;
    }
  }
}
