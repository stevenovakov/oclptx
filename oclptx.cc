/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

/* oclptx.cc
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
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
// adds exception support from CL libraries
// define before CL headers inclusion

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "oclptx.h"

//*********************************************************************
//
// Assorted Function Declerations
//
//*********************************************************************

std::string GenerateFile(std::string suffix);
void WriteToPdf( std::vector<uint32_t>* global_pdf, uint32_t* device_pdf);
void PdfToFile(std::string pdf_filename, std::vector<uint32_t>* global_pdf,
  uint32_t nx, uint32_t ny, uint32_t nz);

//*********************************************************************
//
// Main
//
//*********************************************************************

int main(int argc, char *argv[] )
{
  // stuff set by s_manager and environment

  // Test Routine
  //OclEnv environment("interptest");
  //
  // OclEnv should only ever be declared once (can rewrite as singleton
  // class later). Recompile programs with ->SetOclRoutine(...)
  //
  //SimpleInterpolationTest(environment.GetContext(),
                          //environment.GetCq(0),
                          //environment.GetKernel(0));

  // Sample Manager
  auto t_end = std::chrono::high_resolution_clock::now();
  auto t_start = std::chrono::high_resolution_clock::now();

  SampleManager& s_manager = SampleManager::GetInstance();
  if(&s_manager == NULL)
  {
     std::cout<<"\n Null value"<<std::endl;
  }
  else
  {
    s_manager.ParseCommandLine(argc, argv);

    const BedpostXData* f_data = s_manager.GetFDataPtr();
    const BedpostXData* theta_data = s_manager.GetThetaDataPtr();
    const BedpostXData* phi_data = s_manager.GetPhiDataPtr();

    unsigned int n_particles = s_manager.GetNumParticles();
    unsigned int max_steps = s_manager.GetNumMaxSteps();

    // do not calld delete[] on
    const float4* initial_positions =
      s_manager.GetSeedParticles()->data();
      
    const unsigned short int* brain_mask =
      s_manager.GetBrainMaskToArray();
    
    // TODO get curv thresh from sample manager
    float curvature_threshold = 0.2;
    float mem_frac = 1.0;
      
    OclEnv environment("standard");
    unsigned int n_devices = environment.HowManyDevices();

    int enough_mem;
    enough_mem = environment.AvailableGPUMem(mem_frac); // will pass args later

    if (enough_mem < 0)
    {
      printf("Insufficient GPU Memory: Terminating Program\n\n");
    }
    else
    {
      //
      // and then (this is a naive, "serial" implementation;
      //

      environment.AllocateSamples(  f_data,
                                  phi_data,
                                  theta_data,
                                  brain_mask);
      std::cout<<"\tsamples done\n";

      std::cout<<"Using " << n_devices << " Devices\n";

      std::vector<OclPtxHandler*> handlers;

      for (unsigned int d = 0; d < n_devices; d++)
      {
        std::cout<<"Device " << d <<"\n";

        handlers.push_back(
          new OclPtxHandler(environment.GetContext(),
                            environment.GetCq(d),
                            environment.GetKernel(d),
                            environment.GetSumKernel(d),
                            curvature_threshold,
                            environment.GetEnvData()));
        std::cout<<"\tinit done\n";
        handlers.back()->WriteInitialPosToDevice(
                                          initial_positions,
                                          n_particles,
                                          max_steps,
                                          n_devices,
                                          d);
        std::cout<<"\tpos done\n";
        handlers.back()->PrngInit();
        std::cout<<"\tPrng Done\n";
        handlers.back()->SingleBufferInit();
        //handler.DoubleBufferInit( n_particles/2, max_steps);
        std::cout<<"\tdbuff done\n";
      }

      for (unsigned int d = 0; d < n_devices; d++)
      {
        handlers.at(d)->BlockCq();
      }

      for (unsigned int d = 0; d < n_devices; d++)
      {
        std::cout<<"Device " << d << ", Total GPU Memory Allocated (MB): "<<
          handlers.at(d)->GpuMemUsed()/1e6 << "\n";
      }

      std::cout<<"Press Any Button To Continue...\n";
      std::cin.get();

      t_start = std::chrono::high_resolution_clock::now();

      for (unsigned int d = 0; d < n_devices; d++)
      {
        handlers.at(d)->Interpolate();
        std::cout<<"Device " << d << ", interp done\n";
        //t_end = std::chrono::high_resolution_clock::now();
        handlers.at(d)->PdfSum();
        std::cout<<"Device " << d << ", pdf done\n";
      }

      t_end = std::chrono::high_resolution_clock::now();
      std::cout<< "Interpolation Test Time:" <<
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_end-t_start).count() << std::endl;

      std::string path_filename = GenerateFile("_PATHS.dat");
      std::string pdf_filename = GenerateFile("_PDF.dat");

      FILE * path_file;
      path_file = fopen(path_filename.c_str(), "wb");
      fprintf(path_file, "[");
      fclose(path_file);

      for (unsigned int d = 0; d < n_devices; d++)
      {
        handlers.at(d)->ParticlePathsToFile(path_filename);

        if( d < n_devices - 1)
        {
          path_file = fopen(path_filename.c_str(), "ab");
          fprintf(path_file, ",\n");
          fclose(path_file);
        }
      }

      path_file = fopen(path_filename.c_str(), "ab");
      fprintf(path_file, "]");
      fclose(path_file);

      EnvironmentData * env_data = environment.GetEnvData();
      uint32_t pdf_size = env_data->nx * env_data->ny * env_data->nz;

      std::vector<uint32_t> global_pdf(pdf_size, 0);
      uint32_t* device_pdf = new uint32_t[pdf_size];

      printf("Writing PDF Data to %s\n", pdf_filename.c_str());
      for (unsigned int d = 0; d < n_devices; d++)
      {
        handlers.at(d)->GetPdfData(device_pdf);
        WriteToPdf(&global_pdf, device_pdf);
      }

      delete[] device_pdf;
      PdfToFile(pdf_filename, &global_pdf,
        env_data->nx, env_data->ny, env_data->nz);

      for (unsigned int d = 0; d < n_devices; d++)
      {
        delete handlers.at(d);
      }
    }

    delete[] brain_mask;
  }

  std::cout<<"\n\nExiting...\n\n";
  return 0;
}

//*********************************************************************
//
// Assorted Functions
//
//*********************************************************************
std::string GenerateFile(std::string suffix)
{
  std::ostringstream convert(std::ostringstream::ate);
  std::string path_filename;

  time_t t = time(0);
  struct tm * now = localtime(&t);

  convert << "OclPtx Results/"<< now->tm_yday << "-" <<
    static_cast<int>(now->tm_year) + 1900 << "_"<< now->tm_hour <<
      ":" << now->tm_min << ":" << now->tm_sec;

  path_filename = convert.str() + suffix;

  return path_filename;
}

void WriteToPdf( std::vector<uint32_t>* global_pdf, uint32_t* device_pdf)
{
  for (uint32_t i = 0; i < global_pdf->size(); i++)
    global_pdf->at(i) += device_pdf[i];
}

void PdfToFile(std::string pdf_filename, std::vector<uint32_t>*global_pdf,
  uint32_t nx, uint32_t ny, uint32_t nz)
{
  FILE * pdf_file;
  pdf_file = fopen(pdf_filename.c_str(), "wb");
  fprintf(pdf_file, "[");

  uint32_t index = 0;

  for (uint32_t k = 0; k < nz; k++)
  {
    fprintf(pdf_file, "[");
    for (uint32_t j = 0; j < ny; j++)
    {
      fprintf(pdf_file, "[");
      for (uint32_t i = 0; i < nx; i++)
      {
        index = i*(ny*nz) + j*(nz) + k;
        fprintf(pdf_file, "%u", global_pdf->at(index));

        if (i < nx - 1)
          fprintf(pdf_file, ",");
      }
      fprintf(pdf_file, "]");

      if (j < ny - 1)
        fprintf(pdf_file, ",");
    }
    fprintf(pdf_file, "]");

    if (k < nz - 1)
      fprintf(pdf_file, ",");
  }

  fprintf(pdf_file, "]");
  fclose(pdf_file);
}

//EOF