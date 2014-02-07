/*  Copyright (C) 2004
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */
 
/* interptest.h
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
 
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <time.h>
#include <cstdlib>

#include "customtypes.h" 
 
//
// Declarations
// 
FloatVolume CreateVoxelSpace( unsigned int NX, unsigned int NY,
  unsigned int NZ, float4 min_bounds, float4 max_bounds);
FloatVolume CreateFlowSpace( FloatVolume voxel_space,
  float dr, float3 setpoints );                     
float4 FlowFunction( float4 coords, float dr, float3 setpoints);
std::vector<float4> RandSeedPoints( int n, 
  IntVolume vvol, std::vector<unsigned int> seed_elem );
std::vector<unsigned int> RandSeedElem( int n,
  float3 mins, float3 maxs, IntVolume vvol );

void VolumeToFile( IntVolume ivol, FloatVolume fvol );
void PathsToFile( std::vector<float4> path_vector,
  unsigned int n_seeds, unsigned int n_steps);
 
//
// Simple integer voxel space, from 0,0,0 to N,N,N
// 
FloatVolume CreateVoxelSpace( unsigned int NX, unsigned int NY, 
  unsigned int NZ, float4 min_bounds, float4 max_bounds)
{
  
  FloatVolume voxel_space;
  
  voxel_space.nx = NX;
  voxel_space.ny = NY;
  voxel_space.nz = NZ;
  
  float dx = (max_bounds.x - min_bounds.x)/(float)NX;
  float dy = (max_bounds.y - min_bounds.y)/(float)NY;
  float dz = (max_bounds.z - min_bounds.z)/(float)NZ;  
  
  float4 temp;
  temp.t = 0.0;
  
  for(unsigned int k = 0; k < NZ; k ++)
  {
        for(unsigned int j = 0; j < NY; j++)
        {
            for(unsigned int i = 0; i < NX; i++)
            {
              
              temp.x=i*dx + min_bounds.x;
              temp.y=j*dy + min_bounds.y;
              temp.z=k*dz + min_bounds.z;
              voxel_space.vol.push_back(temp);
              temp.x += dx;
              voxel_space.vol.push_back(temp);
              temp.y += dy;
              voxel_space.vol.push_back(temp);
              temp.x -= dx;
              voxel_space.vol.push_back(temp);
              temp.y -= dy;
              temp.z += dz;
              voxel_space.vol.push_back(temp);
              temp.x += dx;
              voxel_space.vol.push_back(temp);
              temp.y += dy;
              voxel_space.vol.push_back(temp);
              temp.x -= dx;
              voxel_space.vol.push_back(temp);
              
            }
        }
  }
  
  return voxel_space;  
}

FloatVolume CreateFlowSpace(  FloatVolume voxel_space,
  float dr, float3 setpoints )
{
  
  FloatVolume flow_space;
  flow_space.nx = voxel_space.nx;
  flow_space.ny = voxel_space.ny;
  flow_space.nz = voxel_space.nz;
  
  std::vector<float4>::iterator vit;

  for(vit = voxel_space.vol.begin(); 
    vit != voxel_space.vol.end(); ++vit)
  {
      flow_space.vol.push_back( FlowFunction( *vit,
                                              dr,
                                              setpoints
                                            )
                              );
  }

  
  
  return flow_space;  
}

float4 FlowFunction( float4 coords, float dr, float3 setpoints)
{
  //
  // Right now its just like a "tree" from center of space to top
  //    
  float r = dr;
  float theta = std::atan(1.0) * coords.z/ setpoints.z;
  float phi = 
    std::atan2((float) (coords.y - setpoints.y),
      (float) (coords.x - setpoints.x));
    
  float4 ret;
  ret.x = r;
  ret.y = phi;
  ret.z = theta;
  ret.t = 0;
  
  return ret;
}

std::vector<unsigned int> RandSeedElem( unsigned int n, float3 mins,
  float3 maxs, FloatVolume vvol )
{
  std::vector<unsigned int> seed_elem;
  int temp_elem;
  
  unsigned int x, y, z;
  
  srand( time(NULL) ); 
  
  for(unsigned int i = 0; i < n; i++)
  {
    x = (unsigned int) (rand()%1000*(maxs.x - mins.x))/1000.0 + mins.x;
    y = (unsigned int) (rand()%1000*(maxs.y - mins.y))/1000.0 + mins.y;
    z = (unsigned int) (rand()%1000*(maxs.z - mins.z))/1000.0 + mins.z;

    temp_elem = z*(8*vvol.nx*vvol.ny) + y*8*vvol.nx + 8*x;
    // "elem #" is START of vertex list 0, 1, 2, ...., 6, 7
    
    seed_elem.push_back((unsigned int) temp_elem);
    //std::cout<<temp_elem<<"\n";
  }
  
  return seed_elem;
  
}

std::vector<float4> RandSeedPoints( int n, 
  FloatVolume vvol, std::vector<unsigned int> seed_elem )
{
  std::vector<float4> seed_set;
  float4 temp_point;
  temp_point.t = 0;
  
  int maxx, minx, maxy, miny, maxz, minz;
  
  srand( time(NULL) );
  
  for(unsigned int i = 0; i < seed_elem.size(); i++)
  {

    minx = vvol.vol.at(seed_elem.at(i)).x;
    maxx = vvol.vol.at(seed_elem.at(i) + 1).x;
    
    miny = vvol.vol.at(seed_elem.at(i)).y;
    maxy = vvol.vol.at(seed_elem.at(i)+2).y;
    
    minz = vvol.vol.at(seed_elem.at(i)).z;
    maxz = vvol.vol.at(seed_elem.at(i)+4).z;
    
    temp_point.x = rand()%1000*(maxx - minx)/1000.0 + minx;
    temp_point.y = rand()%1000*(maxy - miny)/1000.0 + miny;
    temp_point.z = rand()%1000*(maxz - minz)/1000.0 + minz;

    seed_set.push_back(temp_point);
  }

  return seed_set;
}


void VolumeToFile( FloatVolume vvol, FloatVolume fvol )
{
  
  int vsize = vvol.vol.size()/8;
  int fsize = fvol.vol.size()/8;
  
  std::ostringstream convert (std::ostringstream::ate);
  
  std::string volume_filename;
  std::string flow_filename;
    
  
  time_t t = time(0);
  struct tm * now = localtime(&t);
  
  convert << "Test Data/"<< now->tm_yday << "-" <<
    ((int) now->tm_year) + 1900 << "_"<< now->tm_hour << ":" <<
      now->tm_min << ":" << now->tm_sec;
    
  volume_filename = convert.str() + "_VOL.dat";
  flow_filename = convert.str() + "_FLOW.dat";
  
  std::fstream volume_file;
  volume_file.open(volume_filename.c_str(), std::ios::app|std::ios::out);
  
  
  for(int i = 0; i < vsize; i ++)
  {
    for( int j = 0; j < 8; j++)
    {
      volume_file<< vvol.vol.at(j + i).x <<"," <<
        vvol.vol.at(j + i).y << "," << vvol.vol.at(j + i).z;
      
      if( j < 7 )
        volume_file<<",";
    }
        
    if( i == vsize/8 - 1)
      break;
    else
      volume_file <<"\n";
  }
  
  volume_file.close();
  
  
  std::fstream flow_file;
  flow_file.open(flow_filename.c_str(), std::ios::app|std::ios::out);
  
  for(int i = 0; i < fsize; i ++)
  {
      
    for( int j = 0; j < 8; j++)
    {
      flow_file<< fvol.vol.at(j + i).x <<"," << fvol.vol.at(j + i).y <<
        "," << fvol.vol.at(j + i).z;
      
      if( j < 7 )
        flow_file<<",";
    }
        
    if( i == fsize - 1)
      break;
    else
      flow_file <<"\n";
  }
  
  flow_file.close();  
  
}


void PathsToFile( std::vector<float4> path_vector,
  unsigned int n_seeds, unsigned int n_steps)
{
    
  std::ostringstream convert (std::ostringstream::ate);
  
  std::string path_filename;
  
  std::vector<float> temp_x;
  std::vector<float> temp_y;
  std::vector<float> temp_z;
  
  time_t t = time(0);
  struct tm * now = localtime(&t);
  
  convert << "Test Data/"<< now->tm_yday << "-" <<
    ((int) now->tm_year) + 1900 << "_"<< now->tm_hour << ":" <<
      now->tm_min << ":" << now->tm_sec;
    
  path_filename = convert.str() + "_PATH.dat";
  std::cout<<"Writing to " << path_filename <<"\n";
   
  std::fstream path_file;
  path_file.open(path_filename.c_str(), std::ios::app|std::ios::out);
  
  for( unsigned int n = 0; n < n_seeds;n ++ )
  {
    for( unsigned int s = 0; s < n_steps; s++ )
    {
      temp_x.push_back( path_vector.at(n*n_steps + s).x );
      temp_y.push_back( path_vector.at(n*n_steps + s).y );
      temp_z.push_back( path_vector.at(n*n_steps + s).z );    
    }
    
    for(unsigned int i = 0; i < (unsigned int) n_steps; i++)
    {
      path_file<<temp_x.at(i);
      
      if(i < (unsigned int) n_steps - 1)
        path_file<<",";
      else
        path_file<<"\n";   
    }
    
    for(unsigned int i = 0; i < (unsigned int) n_steps; i++)
    {
      path_file<<temp_y.at(i);
      
      if(i < (unsigned int) n_steps - 1)
        path_file<<",";
      else
        path_file<<"\n";    
    }
    
    for(unsigned int i = 0; i < (unsigned int) n_steps; i++)
    {
      path_file<<temp_z.at(i);
      
      if(i < (unsigned int) n_steps - 1)
        path_file<<",";
      else
        path_file<<"\n";    
    }
    
    //std::cout<< temp_x.size() <<","<<temp_y.size()<<","<<
    //  temp_z.size()<<"\n";
    temp_x.clear();
    temp_y.clear();
    temp_z.clear();
    
    //if( n < n_seeds - 1)
      //path_file<<"\n";
    
  }
  
  path_file.close();
  
}