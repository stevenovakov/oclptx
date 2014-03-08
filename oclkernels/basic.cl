/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

/* basic.cl
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

/*
 * OCL KERNEL COMPILATION - APPEND ORDER
 *
 * Append parsed files in this order in one container
 * before compiling kernel for runtime:
 *
 *      prngmethods.cl
 *      basic.cl
 *
 */

// sample data
// Access x, y, z vertex:
//    index = x*(ny*nz*ns*ndir) + y*(nz*ns*ndir) + z*(ns*ndir) + s*ndir
//    here ndir = 1, and is not included

__kernel void BasicInterpolate(
  __global unsigned int* particle_indeces, //R
  __global float4* particle_paths, //R
  __global unsigned int* particle_steps_taken, //RW
  __global unsigned int* particle_done, //RW
  __global float* f_samples, //R
  __global float* phi_samples, //R
  __global float* theta_samples, //R
  __global unsigned short int* brain_mask, //R
  unsigned int section_size, // dont think we need this...remove later
  unsigned int max_steps,
  unsigned int sample_nx,
  unsigned int sample_ny,
  unsigned int sample_nz,
  unsigned int sample_ns,
  unsigned int interval_steps
)
{
  unsigned int glid = get_global_id(0);
  
  unsigned int particle_index = particle_indeces[glid];
  unsigned int steps_taken = particle_steps_taken[particle_index];
  unsigned int current_path_index =
    particle_index*(max_steps + 1) + steps_taken;
    
  unsigned int interval_steps_taken;
  
  int3 current_root_vertex;
  
  unsigned int diffusion_index;
  unsigned int sample;
  
  // last location of particle
  float4 particle_pos = particle_paths[current_path_index];
  particle_pos.s3 = 0.0;

  float4 temp_pos = (float4) (0.0f); //dx, dy, dz
  float4 xyz= (float4) (0.0f);
  
  float xmin, xmax, ymin, ymax, zmin, zmax;
  float f, phi, theta;
  
  unsigned int brain_mask_index;
  //unsigned int termination_mask_index;
  unsigned short int bounds_test;
  
  for (interval_steps_taken = 0; interval_steps_taken < interval_steps;
    interval_steps_taken++)
  {
    // calculate current index in diffusion space
    current_root_vertex.s0 = floor(particle_pos.s0);
    current_root_vertex.s1 = floor(particle_pos.s1);
    current_root_vertex.s2 = floor(particle_pos.s2);
    
    diffusion_index = 
      current_root_vertex.s0*(sample_nz*sample_ny*sample_ns) +
      current_root_vertex.s1*(sample_nz*sample_ns) +
      current_root_vertex.s2*(sample_ns);
    
    // pick sample
    sample = 0; // fixed, for now
    
    xmin = (float) current_root_vertex.s0;
    ymin = (float) current_root_vertex.s1;
    zmin = (float) current_root_vertex.s2;
    xmax = xmin + 1.0;
    ymax = ymin + 1.0;
    zmax = zmin + 1.0;
    
    // find next step location
    f = f_samples[diffusion_index + sample];
    theta = theta_samples[diffusion_index + sample];
    phi = phi_samples[diffusion_index + sample];
    
    xyz.s0 = 0.25 * cos( phi ) * sin( theta );
    xyz.s1 = 0.25 * sin( phi ) * sin( theta );
    xyz.s2 = 0.25 * cos( theta );

    temp_pos = particle_pos + xyz;
    
    //
    // Brain Mask Test - Checks NEAREST vertex.
    //
    brain_mask_index = 
      floor(temp_pos.s0)*(sample_nz*sample_ny) +
        floor(temp_pos.s1)*(sample_ny) + floor(temp_pos.s2);

    bounds_test = brain_mask[brain_mask_index];

    if (bounds_test == 0)
    {
      particle_done[particle_index] = 1;
      //break;
    }

    // update current location
    particle_pos = temp_pos;
    // add to particle paths
    current_path_index = current_path_index + 1;
    particle_paths[current_path_index] = (float4)( 1.0*bounds_test,
      1.0*current_root_vertex.s1, 1.0*current_root_vertex.s2, 0.0); //particle_pos;
    
    // update steps taken
    steps_taken = steps_taken + 1;
    // update step location
    particle_steps_taken[particle_index] = steps_taken;
    
    if (steps_taken == max_steps){
      particle_done[particle_index] = 1;
      break;  
    }
  }
}


//EOF