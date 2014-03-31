/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

/* samplemanager.cc
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
#include <cmath>

#include "samplemanager.h"
#include "oclptxOptions.h"

//
// Assorted Functions Declerations
//


uint64_t rand_64()
{
  // Assumption: rand() gives at least 16 bits.  It gives 31 on my system.
  assert(RAND_MAX >= (1<<16));

  uint64_t a,b,c,d;
  a = rand() & ((1<<16)-1);
  b = rand() & ((1<<16)-1);
  c = rand() & ((1<<16)-1);
  d = rand() & ((1<<16)-1);

  uint64_t r = ((a << (16*3)) | (b << (16*2)) | (c << (16)) | d);

  return r;
}

cl_ulong8 SampleManager::NewRng()
{
  cl_ulong8 rng = {0,};
  fprintf(stderr, "Rng: ");
  for (int i = 0; i < 5; i++)
  {
    rng.s[i] = rand_64();
    fprintf(stderr, "%lx,", rng.s[i]);
  }
  fprintf(stderr, "\n");

  return rng;
}

std::string SampleManager::IntTostring(const int& value)
{
    std::stringstream s;
    s << value;
    return s.str();
}

//Private method: Used for loading data directly into member containers
void SampleManager::LoadBedpostDataHelper(
  const std::string& aThetaSampleName,
  const std::string& aPhiSampleName,
  const std::string& afSampleName,
  const NEWIMAGE::volume<float>& aMask,
  const int aFiberNum  )
{
  NEWIMAGE::volume4D<float> loadedVolume4DTheta;
  NEWIMAGE::volume4D<float> loadedVolume4DPhi;
  NEWIMAGE::volume4D<float> loadedVolume4Df;

  //Load Theta/Phi/f samples
  NEWIMAGE::read_volume4D(loadedVolume4DTheta, aThetaSampleName);
  NEWIMAGE::read_volume4D(loadedVolume4DPhi, aPhiSampleName);
  NEWIMAGE::read_volume4D(loadedVolume4Df, afSampleName);


  PopulateTHETA(loadedVolume4DTheta,
    _thetaData, loadedVolume4DTheta[0], aFiberNum, false);
  PopulatePHI(loadedVolume4DPhi,
    _phiData, loadedVolume4DPhi[0], aFiberNum, false);
  PopulateF(loadedVolume4Df,
    _fData, loadedVolume4Df[0], aFiberNum, false);
}


void SampleManager::PopulateF(
  const NEWIMAGE::volume4D<float> aLoadedData,
  BedpostXData& aTargetContainer,
  const NEWIMAGE::volume<float> aMaskParams,
  const int aFiberNum,
  bool _16bit)
{

  const int ns = aLoadedData.tsize();
  const int nx = aLoadedData.xsize();
  const int ny = aLoadedData.ysize();
  const int nz = aLoadedData.zsize();

  aTargetContainer.data.push_back( new float[ns*nx*ny*nz] );
  aTargetContainer.nx = nx;
  aTargetContainer.ny = ny;
  aTargetContainer.nz = nz;
  aTargetContainer.ns = ns;

  int xoff = aLoadedData[0].minx() - aMaskParams.minx();
  int yoff = aLoadedData[0].miny() - aMaskParams.miny();
  int zoff = aLoadedData[0].minz() - aMaskParams.minz();

  for (int z = aMaskParams.minz(); z <= aMaskParams.maxz(); z++)
  {
    for (int y = aMaskParams.miny(); y <= aMaskParams.maxy(); y++)
    {
      for (int x = aMaskParams.minx(); x <= aMaskParams.maxx(); x++)
      {
        for (int t = aLoadedData.mint();
          t <= aLoadedData.maxt(); t++)
        {
            aTargetContainer.data.at(aFiberNum)[t*nx*ny*nz +
              x*nz*ny + y*nz + z] =
                aLoadedData[t](x+xoff,y+yoff,z+zoff);
        }
      }
    }
  }
}

void SampleManager::PopulatePHI(
  const NEWIMAGE::volume4D<float> aLoadedData,
  BedpostXData& aTargetContainer,
  const NEWIMAGE::volume<float> aMaskParams,
  const int aFiberNum,
  bool _16bit)
{
  const int ns = aLoadedData.tsize();
  const int nx = aLoadedData.xsize();
  const int ny = aLoadedData.ysize();
  const int nz = aLoadedData.zsize();

  aTargetContainer.data.push_back( new float[ns*nx*ny*nz] );
  aTargetContainer.nx = nx;
  aTargetContainer.ny = ny;
  aTargetContainer.nz = nz;
  aTargetContainer.ns = ns;

  int xoff = aLoadedData[0].minx() - aMaskParams.minx();
  int yoff = aLoadedData[0].miny() - aMaskParams.miny();
  int zoff = aLoadedData[0].minz() - aMaskParams.minz();

  float angle;

  for (int z = aMaskParams.minz(); z <= aMaskParams.maxz(); z++)
  {
    for (int y = aMaskParams.miny(); y <= aMaskParams.maxy(); y++)
    {
      for (int x = aMaskParams.minx(); x <= aMaskParams.maxx(); x++)
      {
        for (int t = aLoadedData.mint();
          t <= aLoadedData.maxt(); t++)
        {
            angle = aLoadedData[t](x+xoff,y+yoff,z+zoff);
            angle = atan2(sin(angle), cos(angle));

            if (angle < 0.)
              angle = 2*M_PI + angle;
            
            aTargetContainer.data.at(aFiberNum)[t*nx*ny*nz +
              x*nz*ny + y*nz + z] = angle;
        }
      }
    }
  }
}

void SampleManager::PopulateTHETA(
  const NEWIMAGE::volume4D<float> aLoadedData,
  BedpostXData& aTargetContainer,
  const NEWIMAGE::volume<float> aMaskParams,
  const int aFiberNum,
  bool _16bit)
{
  const int ns = aLoadedData.tsize();
  const int nx = aLoadedData.xsize();
  const int ny = aLoadedData.ysize();
  const int nz = aLoadedData.zsize();

  aTargetContainer.data.push_back( new float[ns*nx*ny*nz] );
  aTargetContainer.nx = nx;
  aTargetContainer.ny = ny;
  aTargetContainer.nz = nz;
  aTargetContainer.ns = ns;

  int xoff = aLoadedData[0].minx() - aMaskParams.minx();
  int yoff = aLoadedData[0].miny() - aMaskParams.miny();
  int zoff = aLoadedData[0].minz() - aMaskParams.minz();

  float angle;

  for (int z = aMaskParams.minz(); z <= aMaskParams.maxz(); z++)
  {
    for (int y = aMaskParams.miny(); y <= aMaskParams.maxy(); y++)
    {
      for (int x = aMaskParams.minx(); x <= aMaskParams.maxx(); x++)
      {
        for (int t = aLoadedData.mint();
          t <= aLoadedData.maxt(); t++)
        {
            angle = aLoadedData[t](x+xoff,y+yoff,z+zoff);
            angle = acos(cos(angle));

            aTargetContainer.data.at(aFiberNum)[t*nx*ny*nz +
              x*nz*ny + y*nz + z] = angle;
        }
      }
    }
  }
}

void SampleManager::LoadBedpostData(const std::string& aBasename)
{
  std::cout<<"Loading Bedpost samples....."<<std::endl;
  if(aBasename == "")
  {
    std::cout<< "Bad File Name"<<std::endl;
    return;
  }

  //Set Particle Number and Max Steps
  _nParticles = _oclptxOptions.nparticles.value();
  _nMaxSteps = _oclptxOptions.nsteps.value();

  //Load Sample Data
  std::string thetaSampleNames;
  std::string phiSampleNames;
  std::string fSampleNames;

  //Single Fiber Case.
  if(NEWIMAGE::fsl_imageexists(aBasename+"_thsamples"))
  {
    thetaSampleNames = aBasename+"_thsamples";
    phiSampleNames = aBasename+"_phisamples";
    fSampleNames = aBasename+"_fsamples";
    LoadBedpostDataHelper(
      thetaSampleNames,phiSampleNames,fSampleNames);
  }
  //Multiple Fiber Case.
  else
  {
    int fiberNum = 1;
    std::string fiberNumAsstring = IntTostring(fiberNum);
    thetaSampleNames = aBasename+"_th"+fiberNumAsstring+"samples";
    bool doesFiberExist = NEWIMAGE::fsl_imageexists(thetaSampleNames);
    while(doesFiberExist)
    {
      phiSampleNames = aBasename+"_ph"+fiberNumAsstring+"samples";
      fSampleNames = aBasename+"_f"+fiberNumAsstring+"samples";

      LoadBedpostDataHelper(
       thetaSampleNames,phiSampleNames,fSampleNames);

      fiberNum++;
      fiberNumAsstring = IntTostring(fiberNum);
      thetaSampleNames = aBasename+"_th"+fiberNumAsstring+"samples";
      doesFiberExist = NEWIMAGE::fsl_imageexists(thetaSampleNames);
    }
    if(fiberNum == 1)
    {
      std::cout<<
       "Could not find samples. Exiting Program..."<<std::endl;
      exit(1);
    }
    std::cout<<"Finished Loading Samples from Bedpost"<<std::endl;
  }
}

void SampleManager::ParseCommandLine(int argc, char** argv)
{
  _oclptxOptions.parse_command_line(argc, argv);

  if (_oclptxOptions.verbose.value()>0)
  {
    _oclptxOptions.status();
  }

  if (_oclptxOptions.simple.value())
  {
      if (_oclptxOptions.matrix1out.value() ||
        _oclptxOptions.matrix3out.value())
      {
        std::cout<<
         "Error: cannot use matrix1 and matrix3 in simple mode"<<
            std::endl;
        exit(1);
      }
      std::cout<<"Running in simple mode"<<std::endl;
      this->LoadBedpostData(_oclptxOptions.basename.value());
      if(_oclptxOptions.seedref.value() == "")
      {
        NEWIMAGE::read_volume(_brainMask,
          _oclptxOptions.maskfile.value());
      }
      else
      {
        NEWIMAGE::read_volume(_brainMask,
          _oclptxOptions.seedref.value());
      }
      if(_oclptxOptions.rubbishfile.value() != "")
      {
         NEWIMAGE::read_volume(_exclusionMask,
         _oclptxOptions.rubbishfile.value());
         std::cout<<"Successfully loaded Exclusion Mask"<<endl;
      }
      if(_oclptxOptions.stopfile.value() != "")
      {
        NEWIMAGE::read_volume(_terminationMask,
        _oclptxOptions.stopfile.value());
        std::cout<<"Successfully loaded Termination Mask"<<endl;
      }
      if(_oclptxOptions.waypoints.set())
      {
        std::string waypoints = _oclptxOptions.waypoints.value();
        std::istringstream ss(waypoints);
        std::string wayMaskLocation;
        while(std::getline(ss,wayMaskLocation,','))
        {
          NEWIMAGE::volume<short int> vol;
          NEWIMAGE::read_volume(vol, wayMaskLocation);
          _wayMasks.push_back(vol);
        }
        cout<<"Successfully loaded " << _wayMasks.size() << " WayMasks"<<endl;
      }
      this->GenerateSeedParticles(_oclptxOptions.sampvox.value());
  }
  else if (_oclptxOptions.network.value())
  {
    std::cout<<"Network mode unsupported. Terminating Application"<<std::endl;
    exit(1);
  }
  else
  {
    std::cout<<"Running in seedmask mode"<<std::endl;
  }
}

void SampleManager::GenerateSeedParticles(float aSampleVoxel)
{
  using namespace std;

  Matrix seeds =
  MISCMATHS::read_ascii_matrix(_oclptxOptions.seedfile.value());

  srand(time(NULL));
  srand(_oclptxOptions.rseed.value());

  cl_float4 seed;
  struct OclPtxHandler::particle_data *particle;

  // If there is no seed file given with the -x CLI parameter,
  // we use the middle of the mask as the seed.
  if (seeds.Nrows() == 0)
  {
    seed.s[0] = floor((_brainMask.xsize())/2.0);
    seed.s[1] = floor((_brainMask.ysize())/2.0);
    seed.s[2] = floor((_brainMask.zsize())/2.0);
    GenerateSeedParticlesHelper(seed,aSampleVoxel);
  }
  else
  {
    this->_nParticles = seeds.Nrows();
    _seedParticles = new Fifo<struct OclPtxHandler::particle_data>(_nParticles);
    if (seeds.Ncols()!=3 && seeds.Nrows()==3)
    {
       seeds=seeds.t();
    }
    for (int t = 1; t<=seeds.Nrows(); t++)
    {
      seed.s[0] = seeds(t,1);
      seed.s[1] = seeds(t,2);
      seed.s[2] = seeds(t,3);
      particle = new OclPtxHandler::particle_data;
      *particle = {NewRng(), seed};
      _seedParticles.PushOrDie(seed);
    // TODO @STEVE
    // Implement Jittering on seed file.
    // GenerateSeedParticlesHelper(seed, aSampleVoxel);
    }
  }
}

void SampleManager::GenerateSeedParticlesHelper(
  cl_float4 aSeed, float aSampleVoxel)
{
 struct OclPtxHandler::particle_data *particle;
 _seedParticles = new Fifo<struct OclPtxHandler::particle_data>(_nParticles);
 for (int p = 0; p<_nParticles; p++)
 {
  cl_float4 randomParticle = aSeed;
  if(aSampleVoxel > 0)
  {
    float dx,dy,dz;
    float radSq = aSampleVoxel*aSampleVoxel;
    
    while(true)
    {
      dx=2.0*aSampleVoxel*((float)rand()/float(RAND_MAX)-.5);
      dy=2.0*aSampleVoxel*((float)rand()/float(RAND_MAX)-.5);
      dz=2.0*aSampleVoxel*((float)rand()/float(RAND_MAX)-.5);
      if(dx*dx + dy*dy + dz*dz <= radSq)
      {
         break;
      }
    }
    randomParticle.s[0] += dx / _brainMask.xdim();
    randomParticle.s[1] += dy / _brainMask.ydim();
    randomParticle.s[2] += dz / _brainMask.zdim();
  }

  particle = new OclPtxHandler::particle_data;
  *particle = {NewRng(), randomParticle};
  _seedParticles.PushOrDie(randomParticle);
 }
}

const unsigned short int* SampleManager::GetBrainMaskToArray()
{
  return GetMaskToArray(_brainMask);
}

const unsigned short int* SampleManager::GetExclusionMaskToArray()
{
  if(_oclptxOptions.rubbishfile.value() != "")
    return GetMaskToArray(_exclusionMask);
  else
    return NULL;
}

const unsigned short int* SampleManager::GetTerminationMaskToArray()
{
  if(_oclptxOptions.stopfile.value() != "")
    return GetMaskToArray(_terminationMask);
  else
    return NULL;
}

const std::vector<unsigned short int*> SampleManager::GetWayMasksToVector()
{
  vector<unsigned short int*> waymasks;
  for (unsigned int i = 0; i < _wayMasks.size(); i++)
  {
    waymasks.push_back(GetMaskToArray(_wayMasks.at(i)));
  }
  return waymasks;
}


unsigned short int* SampleManager::GetMaskToArray(
  NEWIMAGE::volume<short int> aMask)
{
  const int maxZ = aMask.maxz();
  const int maxY = aMask.maxy();
  const int maxX = aMask.maxx();
  const int minZ = aMask.minz();
  const int minY = aMask.miny();
  const int minX = aMask.minx();
  const int sizeX = aMask.xsize();
  const int sizeY = aMask.ysize();
  const int sizeZ = aMask.zsize();

  unsigned short int* target =
    new unsigned short int[sizeX * sizeY * sizeZ];

  for (int z = minZ; z <= maxZ; z++)
  {
    for (int y = minY; y <= maxY; y++)
    {
      for (int x = minX; x <= maxX; x++)
      {
            target[x*sizeY*sizeZ + y*sizeZ + z] = aMask(x,y,z);
      }
    }
  }
  return target;
}

float const SampleManager::GetThetaData(
  int aFiberNum, int aSamp, int aX, int aY, int aZ)
{
  if(_thetaData.data.size()>0)
  {
    int nx = _thetaData.nx;
    int ny = _thetaData.ny;
    int nz = _thetaData.nz;
    return _thetaData.data.at(aFiberNum)[(aSamp)*(nx*ny*nz) +
      (aX)*(nz*ny) + (aY)*nz + (aZ)];
  }
  return 0.0f;
}

float const SampleManager::GetPhiData(
  int aFiberNum, int aSamp, int aX, int aY, int aZ)
{
  if(_phiData.data.size()>0)
  {
    int nx = _phiData.nx;
    int ny = _phiData.ny;
    int nz = _phiData.nz;
    return _phiData.data.at(aFiberNum)[(aSamp)*(nx*ny*nz) +
      (aX)*(nz*ny) + (aY)*nz + (aZ)];
  }
  return 0.0f;
}

float const SampleManager::GetfData(int aFiberNum,
  int aSamp, int aX, int aY, int aZ)
{
  if(_fData.data.size()>0)
  {
    int nx = _fData.nx;
    int ny = _fData.ny;
    int nz = _fData.nz;
    return _fData.data.at(aFiberNum)[(aSamp)*(nx*ny*nz) +
      (aX)*(nz*ny) + (aY)*nz + (aZ)];
  }
  return 0.0f;
}

const BedpostXData* SampleManager::GetThetaDataPtr()
{
  if(_thetaData.data.size() > 0)
  {
    return &_thetaData;
  }
  return NULL;
}

const BedpostXData* SampleManager::GetPhiDataPtr()
{
  if(_phiData.data.size() > 0)
  {
    return &_phiData;
  }
  return NULL;
}

const BedpostXData* SampleManager::GetFDataPtr()
{
  if(_fData.data.size() > 0)
  {
    return &_fData;
  }
  return NULL;
}
//*********************************************************************
// samplemanager Constructors/Destructors/Initializers
//*********************************************************************

//Use this to get the only instance of SampleManager in the program.

SampleManager& SampleManager::GetInstance()
{
    if(_manager == NULL)
    {
        _manager = new SampleManager();
    }
    return *_manager;
}SampleManager* SampleManager::_manager;

//Private Constructor.
SampleManager::SampleManager():_oclptxOptions(
  oclptxOptions::getInstance()),
  _seedParticles(NULL)
{}

SampleManager::~SampleManager()
{
  if (NULL != _seedParticles)
    delete _seedParticles;

  for (unsigned int i = 0; i < _thetaData.data.size(); i++)
  {
    delete[] _thetaData.data.at(i);
  }

  for (unsigned int i = 0; i < _phiData.data.size(); i++)
  {
    delete[] _phiData.data.at(i);
  }

  for (unsigned int i = 0; i < _fData.data.size(); i++)
  {
    delete[] _fData.data.at(i);
  }

  delete _manager;
}
