/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */
 
/* interpolate.cl
 *
 *
 * Part of
 *    oclptx
 * OpenCL-based, GPU accelerated probtrackx algorithm module, to be used
 * with FSL - FMRIB's Software Library
 *
 * This file is part of oclptx.
 *
 * oclptx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * oclptx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with oclptx.  If not, see <http://www.gnu.org/licenses/>.
 *
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

__kernel void OclPtxInterpolate(
  __global uint* particle_indeces, //R
  __global float4* particle_paths, //RW
  __global uint* particle_pdfs, //RW
  __global uint* particle_steps_taken, //RW
  __global uint* particle_done, //RW
  __global rng_t* rng, //RW
  __global float* f_samples, //R
  __global float* phi_samples, //R
  __global float* theta_samples, //R
  __global ushort* brain_mask, //R
  uint max_steps,
  uint n_particles,
  uint sample_nx,
  uint sample_ny,
  uint sample_nz,
  uint sample_ns,
  uint interval_steps,
  float curvature_threshold
#ifdef WAYPOINTS
  , __global ushort* waypoint_masks, //R //unsire if want double ptr
  __global ushort* particle_waypoints, //W
  uint n_waypoint_masks
#endif
#ifdef EXCLUSION
  , __global ushort* exclusion_mask, //R
  __global ushort* particle_exclusion //W
#endif
#ifdef TERMINATION
  , __global ushort* termination_mask //R
#endif
)
{
  uint glid = get_global_id(0);
  
  uint particle_index = particle_indeces[glid];
  uint steps_taken = particle_steps_taken[particle_index];
  uint current_path_index =
    particle_index*(max_steps + 1) + steps_taken;
    
  uint interval_steps_taken;
  
  uint3 current_select_vertex;

  float3 volume_fraction;
  
  uint diffusion_index;
  uint sample;
  
  // last location of particle
  float4 particle_pos = particle_paths[current_path_index];
  particle_pos.s3 = 0.0;

  float4 temp_pos = (float4) (0.0f); //dx, dy, dz
  float4 dr = (float4) (0.0f);
  float4 last_dr = (float4) (0.0f);
  
  float xmin, xmax, ymin, ymax, zmin, zmax;
  xmin = 0.0; ymin = 0.0; zmin = 0.0;
  xmax = sample_nx*1.0; ymax = sample_ny*1.0; zmax = sample_nz*1.0;
  
  float f, phi, theta;
  float jump_dot;

  ulong rng_output;
  float vol_frac;

  float rand_max = 18446744073709551616.; //0xFFFFFFFFFFFFFFFF;
        
  uint mask_index;
  //unsigned int termination_mask_index;
  ushort bounds_test;

  uint vertex_num;
  uint entry_num;
  uint shift_num;
  uint particle_entry;
  
  for (interval_steps_taken = 0; interval_steps_taken < interval_steps;
    interval_steps_taken++)
  {
    // calculate current index in diffusion space
    current_select_vertex.s0 = floor(particle_pos.s0);
    current_select_vertex.s1 = floor(particle_pos.s1);
    current_select_vertex.s2 = floor(particle_pos.s2);

    volume_fraction.s0 = particle_pos.s0 - (float) current_select_vertex.s0;
    volume_fraction.s1 = particle_pos.s1 - (float) current_select_vertex.s1;
    volume_fraction.s2 = particle_pos.s2 - (float) current_select_vertex.s2;

#ifdef EULER_STREAMLINE
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
#ifdef PRNG
    rng_output = Rand(rng  + particle_index);
    sample = rng_output % sample_ns;
#else
    sample = 0;
#endif
    // Volume Fraction Selection
#ifdef PRNG
    rng_output = Rand(rng  + particle_index);
    vol_frac = volume_fraction.s0 * rand_max;

    if (rng_output > vol_frac)
    {
      current_select_vertex.s0 += 1;
    }

    rng_output = Rand(rng  + particle_index);
    vol_frac = volume_fraction.s1 * rand_max;

    if (rng_output > vol_frac)
    {
      current_select_vertex.s1 += 1;
    }

    rng_output = Rand(rng  + particle_index);
    vol_frac = volume_fraction.s2 * rand_max;

    if (rng_output > vol_frac)
    {
      current_select_vertex.s2 += 1;
    }
#endif
    // pick flow vertex
    diffusion_index =
      sample*(sample_nz*sample_ny*sample_nx)+
      current_select_vertex.s0*(sample_nz*sample_ny) +
      current_select_vertex.s1*(sample_nz) +
      current_select_vertex.s2;
    
    // find next step location
    f = f_samples[diffusion_index];
    theta = theta_samples[diffusion_index];
    phi = phi_samples[diffusion_index];
    
    dr.s0 = cos( phi ) * sin( theta );
    dr.s1 = sin( phi ) * sin( theta );
    dr.s2 = cos( theta );
    
    // alternates initial direction of particle set
    if( steps_taken == 0 && glid%2 == 0)
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
    
    if (steps_taken > 1 && jump_dot < curvature_threshold)
    {
      particle_done[particle_index] = 1;
      break;
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
      particle_done[particle_index] = 1;
      break;
    }
    //
    // Brain Mask Test - Checks NEAREST vertex.
    //
    mask_index =
      round(temp_pos.s0)*(sample_nz*sample_ny) +
        round(temp_pos.s1)*(sample_nz) + round(temp_pos.s2);

    bounds_test = brain_mask[mask_index];

    if (bounds_test == 0)
    {
      particle_done[particle_index] = 1;
      break;
    }
#ifdef TERMINATION
#endif
#ifdef EXCLUSION
#endif
#ifdef WAYPOINTS
#endif
    // update current location
    particle_pos = temp_pos;

    // add to particle paths
    current_path_index = current_path_index + 1;
    particle_paths[current_path_index] = particle_pos;
  
    // update steps taken
    steps_taken = steps_taken + 1;
    // update step location
    particle_steps_taken[particle_index] = steps_taken;

    // update particle pdf
    vertex_num =
      round(particle_pos.s0)*round(particle_pos.s1)*round(particle_pos.s2);
    entry_num = vertex_num / 32;
    shift_num = (vertex_num % 32) - 1;

    particle_entry = particle_pdfs[particle_index * n_particles + entry_num];
    particle_pdfs[particle_index * n_particles + entry_num] = particle_entry | (0x00000001 << shift_num);
    
    if (steps_taken == max_steps){
      particle_done[particle_index] = 1;
      break;
    }
  }
}


//EOF