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
  __global int4* particle_elem, //RW
  __global unsigned int* particle_done, //RW
  __global float4* f_samples, //R
  __global float4* phi_samples, //R
  __global float4* theta_samples, //R
  __global unsigned int* brain_mask, //R
  unsigned int section_size, // dont think we need this...remove later
  unsigned int max_steps,
  unsigned int sample_nx,
  unsigned int sample_ny,
  unsigned int sample_nz,
  unsigned int sample_ns,
  unsigned int num_steps
)
{
  unsigned int glid = get_global_id(0);
  
  unsigned int particle_index = particle_indeces[glid];
  unsigned int current_path_index =
    particle_index*max_steps + particle_steps_taken[particle_index];
  
  int4 current_root_vertex = particle_elem[particle_index];
  int4 next_root_vertex = current_root_vertex;
  
  int diffusion_index;
  unsigned int sample;
  
  // last location of particle
  float4 particle_pos = particle_paths[current_path_index];
  particle_pos.s3 = 0.0;

  float4 temp_pos = (float4) (0.0f); //dx, dy, dz
  float4 xyz= (float4) (0.0f);
  
  float xmin, xmax, ymin, ymax, zmin, zmax;
  float f, phi, theta;
  
  unsigned int total_steps = num_steps;
  unsigned int current_step = 0;
  unsigned int steps_taken = particle_steps_taken[particle_index];
  
  uint8 box_vertices_index;
  
  int d_elem_x, d_elem_y, d_elem_z;
  unsigned int d_vert_x, d_vert_y, d_vert_z;

  unsigned int bounds_test;
  
  while(current_step < total_steps)
  {
    // calculate current index in diffusion space
    diffusion_index = (int) (
      current_root_vertex.s0*(sample_nz*sample_ny*sample_ns) +
      current_root_vertex.s1*(sample_nz*sample_ns) +
      current_root_vertex.s2*(sample_ns)
    );
    
    // pick sample
    sample = 0; // fixed, for now
    
    xmin = (float) current_root_vertex.s0;
    ymin = (float) current_root_vertex.s1;
    zmin = (float) current_root_vertex.s2;
    xmax = xmin + 1;
    ymax = ymin + 1;
    zmax = zmin + 1;
    
    // find next step location
    f = f_samples[diffusion_index + sample];
    theta = theta_samples[diffusion_index + sample];
    phi = phi_samples[diffusion_index + sample];
    
    xyz.s0 = f * cos( phi ) * sin( theta );
    xyz.s1 = f * sin( phi ) * sin( theta );
    xyz.s2 = f * cos( theta );

    temp_pos = particle_pos + xyz;
    
    // find root elem of next step location
    d_elem_x = 0;
    d_elem_y = 0;
    d_elem_z = 0;

    if( xmin - temp_pos.x > 0)
      d_elem_x = -1;
    else if( temp_pos.x - xmax > 0)
      d_elem_x = 1;

    if( ymin - temp_pos.y > 0)
      d_elem_y = -1;
    else if( temp_pos.y - ymax > 0)
      d_elem_y = 1;

    if( zmin - temp_pos.z > 0)
      d_elem_z = -1;
    else if( temp_pos.z - zmax > 0)
      d_elem_z = 1;
    
    current_root_vertex = current_root_vertex +
      (int4) ( d_elem_x, d_elem_y, d_elem_z, 0);
    
    // are we outside of brain mask?
    // check all elements, multiply, if zero just stop
    bounds_test = 1;
    
    for (unsigned int i = 0; i < 8; i++)
    {
      d_vert_x = (i & 0x00000001);
      d_vert_y = ((i & 0x00000002) >> 1);
      d_vert_z = ((i & 0x00000004) >> 2);
      
      // because this is a coordinate space, no sample factor here
      box_vertices_index[i] =
        (current_root_vertex.s0 + d_vert_x)*
          (sample_nz*sample_ny) + 
            (current_root_vertex.s1 + d_vert_y)*(sample_nz) +
              (current_root_vertex.s2 + d_vert_z);
              
      bounds_test = bounds_test * brain_mask[box_vertices_index[i]];
    }
    
    // if element is not surrounded by ones, then we are "outside"
    // if so, write to particle done, and exit
    if (bounds_test < 1)
    {
      particle_done[particle_index] = 1;
      break;
    }
    
    // update current location
    particle_pos = temp_pos;
    // add to particle paths
    current_path_index = current_path_index + 1;
    particle_paths[current_path_index] = particle_pos;
    // update steps taken
    steps_taken = steps_taken + 1;
    particle_steps_taken[particle_index] = steps_taken;
    // update elem
    particle_elem[particle_index] = current_root_vertex;
    
    if (steps_taken == max_steps)
      break;  
  }
}


//EOF