/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */
 
/* interptest.cl
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
 *      interptest.cl
 *
 */

// maybe 'better' to declare inline function instead of macro
//#define CIRCTOCART(invec, xyz){\
  //xyz.s1 = sincos(invec.s1, &xyz.s0 ); \
  //xyz.s2 = 0.f;\
  //xyz = xyz* invec.s0 * sin( invec.s2 );\
  //xyz.s2 = invec.s0 * cos(invec.s2);\
////}

//inline void CircToCart( float4 circ, float4 * xyz)
//{
  
  //float4 temp = (float4)(0.0f);
  
  //temp.s1 = sincos(circ.s1, &temp.s0 ); 
  //temp.s2 = 0.f;
  //temp = temp * circ.s0 * sin( circ.s2 );
  //temp.s2 = circ.s0 * cos(circ.s2);
  
  //*xyz = temp;
  
  //return;
  
//}

// both macro and inline function generating SIGSEV upon cl program build

 
// Flow List, Vertex List
// Access x, y, z vertex:
//                var_name.vol[z*(ny*nx*8) + y*nx*8 + x*8 + v]
 
__kernel void InterpolateTestKernel(
                                  __global float4* vertex_list, //R
                                  __global float4* flow_list, //R 
                                  __global float4* seed_list, //R
                                  __global unsigned int* seed_elem,//RW
                                  unsigned int nx, 
                                  unsigned int ny, 
                                  unsigned int nz,
                                  float4 min_bounds,
                                  float4 max_bounds,
                                  unsigned int n_steps,
                                  __global float4* path_storage //W
                                )
{
  
  unsigned int glIDx = get_global_id(0);
  
  int current_elem = (int) seed_elem[glIDx];
  
  float4 max_vert, min_vert;
  
  float4 particle_pos = seed_list[glIDx];
  
  float4 flow_dir = (float4) (0.0f); //dr, phi, theta
  float4 temp_pos = (float4) (0.0f); //dx, dy, dz
  float4 xyz= (float4) (0.0f);
  
  int d_elem_x;
  int d_elem_y;
  int d_elem_z;
  
  float xmin, xmax, ymin, ymax, zmin, zmax;
  
  unsigned int current_step = 0;
  
  path_storage[glIDx*n_steps] = particle_pos;
  
  while(current_step < n_steps)
  {
    //
    // Current elem limits
    //
    
    min_vert = vertex_list[8*current_elem];
    max_vert = vertex_list[8*current_elem+6];
    
    xmin = min_vert.s0;
    xmax = max_vert.s0;
    
    ymin = min_vert.s1;
    ymax = max_vert.s1;
    
    zmin = min_vert.s2;
    zmax = max_vert.s2;
    
    // pick vertex
    // just gonna pick the lowest one (v0)

    // generate next step
    flow_dir = flow_list[8*current_elem+4]; //+0 (x,y,z+1)
    
    //
    // condititions to proceed
    //
    //  is step out of total volume (first, break)
    //  are we transitioning to a new elem
    // 
     
    //CircToCart(flow_dir, &xyz); 
    // hmm compiler has a problem with sincos
    
    xyz.s0 = flow_dir.s0 * cos( flow_dir.s1 ) * sin( flow_dir.s2 );
    xyz.s1 = flow_dir.s0 * sin( flow_dir.s1 ) * sin( flow_dir.s2 );
    xyz.s2 = flow_dir.s0 * cos( flow_dir.s2 );
    
    temp_pos = particle_pos + xyz;

    if( temp_pos.x - max_bounds.s0 > 0.0 ||
      min_bounds.s0 - temp_pos.x > 0.0 ||
        temp_pos.y - max_bounds.s1 > 0.0 ||
          min_bounds.s1 - temp_pos.y > 0.0 ||
            temp_pos.z - max_bounds.s2 > 0.0 ||
              min_bounds.s2 - temp_pos.z > 0.0 )
    {
      break;
    }
    
    d_elem_x = 0;
    d_elem_y = 0;
    d_elem_z = 0;
    
    //
    // If statements suck. Find some way to do this with projections?
    ////    
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
    
    current_elem = current_elem + ((d_elem_z*(ny*nx)) +
        (d_elem_y*nx) + d_elem_x);    
    
    // add to path_storage
    
    particle_pos = temp_pos;
    path_storage[glIDx*n_steps + current_step] = particle_pos;

    // proceed
    
    current_step = current_step + 1;
  }
  
  // Idle Loop
  // Re-write last viable position.
  
  while(current_step < n_steps)
  {
    path_storage[glIDx*n_steps + current_step]= temp_pos;
    current_step = current_step + 1;
  }
  
  // All Done
}


//EOF