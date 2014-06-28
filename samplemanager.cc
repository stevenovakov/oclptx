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

#include "fifo.h"
#include "oclptxhandler.h"
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

void SampleManager::AddSeedParticle(
    float x, float y, float z, float xdim, float ydim, float zdim)
{
  oclptxOptions& opts = oclptxOptions::getInstance();

  float sampvox = opts.sampvox.value();
  struct OclPtxHandler::particle_data *particle;

  cl_float4 forward = {{ 1.0, 0., 0., 0.}};
  cl_float4 reverse = {{-1.0, 0., 0., 0.}};
  cl_float4 pos = {{x, y, z, 0.}};

  for (int p = 0; p < opts.nparticles.value(); p++)
  {
    pos.s[0] = x;
    pos.s[1] = y;
    pos.s[2] = z;
    // random jitter of seed point inside a sphere
    if (sampvox > 0.)
    {
      bool rej = true;
      float dx, dy, dz;
      float r2 = sampvox * sampvox;
      while(rej)
      {
        dx = 2.0 * sampvox * ((float)rand()/float(RAND_MAX)-.5);
        dy = 2.0 * sampvox * ((float)rand()/float(RAND_MAX)-.5);
        dz = 2.0 * sampvox * ((float)rand()/float(RAND_MAX)-.5);
        if( dx * dx + dy * dy + dz * dz <= r2 )
          rej=false;
      }
      pos.s[0] += dx / xdim;
      pos.s[1] += dy / ydim;
      pos.s[2] += dz / zdim;
    }

  
    particle = new OclPtxHandler::particle_data;
    particle->rng = NewRng();
    particle->position = pos;
    particle->dr = forward;
    _seedParticles->Push(particle);

    particle = new OclPtxHandler::particle_data;
    particle->rng = NewRng();
    particle->position = pos;
    particle->dr = reverse;
    _seedParticles->Push(particle);
  }
}

void SampleManager::GenerateSimpleSeeds()
{
  oclptxOptions& opts = oclptxOptions::getInstance();
  NEWIMAGE::volume<short int> seedref;

  if (opts.seedref.value() != "")
  {
    read_volume(seedref,opts.seedref.value());
  }
  else
  {
    read_volume(seedref,opts.maskfile.value());
  }

  NEWMAT::Matrix Seeds = read_ascii_matrix(opts.seedfile.value());
  if (Seeds.Ncols() != 3 && Seeds.Nrows() == 3)
    Seeds = Seeds.t();

  // convert coordinates from nifti (external) to newimage (internal)
  //   conventions - Note: for radiological files this should do nothing
  NEWMAT::Matrix newSeeds(Seeds.Nrows(), 3);
  for (int n = 1; n<=Seeds.Nrows(); n++)
  {
    NEWMAT::ColumnVector v(4);
    v << Seeds(n,1) << Seeds(n,2) << Seeds(n,3) << 1.0;
    v = seedref.niftivox2newimagevox_mat() * v;
    newSeeds.Row(n) << v(1) << v(2) << v(3);
  }

  int count = opts.nparticles.value() * newSeeds.Nrows();
  _seedParticles =
      new Fifo<struct OclPtxHandler::particle_data>(2 * count);

  for (int SN = 1; SN <= newSeeds.Nrows(); SN++)
  {
    float xst = newSeeds(SN, 1);
    float yst = newSeeds(SN, 2);
    float zst = newSeeds(SN, 3);
    AddSeedParticle(xst, yst, zst,
      seedref.xdim(), seedref.ydim(), seedref.zdim());
  }
  _seedParticles->Finish();
}

void SampleManager::GenerateMaskSeeds()
{
// TODO(jeff): CSV causes us to enter dependency hell.  Try to make do without
// it.
#if 0
  oclptxOptions& opts =oclptxOptions::getInstance();
  
  // we need a reference volume for CSV
  // (in case seeds are a list of surfaces)
  NEWIMAGE::volume<short int> refvol;
  if(opts.seedref.value()!="")
    read_volume(refvol,opts.seedref.value());
  else
    read_volume(refvol,opts.maskfile.value());
  
  CSV seeds(refvol);
  seeds.set_convention(opts.meshspace.value());
  seeds.load_rois(opts.seedfile.value());
  puts("done");

  if (seeds.nSurfs() > 0) {
    puts("OclPtx doesn't support surface seedmasks.  Sorry.");
    exit(1);
  }

  if (seeds.nVols() == 0 && opts.seedref.value() == "")
  {
    printf("Warning: need to set a reference volume when defining a "
           "surface-based seed\n");
  }

  // seed from volume-like ROIs
  if (0 == seeds.nVols())
  {
    puts("No volumes specified.");
    exit(1);
  }

  // Figure out how many particles we'll need to allocate our fifo.
  int count = 0;
  for (int roi = 1; roi <= seeds.nVols(); roi++)
    for (int z = 0; z < seeds.zsize(); z++)
      for (int y = 0; y < seeds.ysize(); y++)
        for (int x = 0; x < seeds.xsize(); x++)
          if(seeds.isInRoi(x,y,z,roi))
            count += opts.nparticles.value();

  _seedParticles =
      new Fifo<struct OclPtxHandler::particle_data>(2 * count);

  for (int roi = 1; roi <= seeds.nVols(); roi++) {
    printf("Parsing volume %i\n", roi);

    for (int z = 0; z < seeds.zsize(); z++) {
      for (int y = 0; y < seeds.ysize(); y++) {
        for (int x = 0; x < seeds.xsize(); x++) {
          if(seeds.isInRoi(x,y,z,roi))
            AddSeedParticle(x, y, z, seeds.xdim(), seeds.ydim(), seeds.zdim());
        }
      }
    }
  }
#endif
}

void SampleManager::GenerateSeedParticles()
{
  oclptxOptions& opts =oclptxOptions::getInstance();
  if (opts.simple.value())
  {
    if (opts.matrix1out.value() || opts.matrix3out.value())
    {
      puts("Error: cannot use matrix1 and matrix3 in simple mode");
      exit(1);
    }
    puts("Running in simple mode");
    GenerateSimpleSeeds();
  }
  else if(opts.network.value())
  {
    puts("OclPtx doesn't support network mode.  Sorry");
    exit(1);
  }
  else
  {
    puts("Running in seedmask mode");
    GenerateMaskSeeds();
  }
}

cl_ulong8 SampleManager::NewRng()
{
  cl_ulong8 rng = {{0,}};
  for (int i = 0; i < 5; i++)
  {
    rng.s[i] = rand_64();
  }

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
       thetaSampleNames,phiSampleNames,fSampleNames,
       NEWIMAGE::volume<float>(), fiberNum-1);

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
  this->GenerateSeedParticles();
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

vector<unsigned short int*>* SampleManager::GetWayMasksToVector()
{
  vector<unsigned short int*>* waymasks = 
    new vector<unsigned short int*>;
  for (unsigned int i = 0; i < _wayMasks.size(); i++)
  {
    waymasks->push_back(GetMaskToArray(_wayMasks.at(i)));
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

cl_float4 SampleManager::brain_mask_dim()
{
  return cl_float4{{_brainMask.xdim(),
                    _brainMask.ydim(),
                    _brainMask.zdim(),
                    1.}};
}

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
}
