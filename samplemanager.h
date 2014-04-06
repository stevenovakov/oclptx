/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

/* samplemanager.h
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

#ifndef  SAMPLEMANAGER_H_
#define  SAMPLEMANAGER_H_

#include <iostream>
#include <vector>
#include <string>

#include "fifo.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "oclptxhandler.h"
#include "oclptxOptions.h"
#include "customtypes.h"

class SampleManager
{
  public:
    static SampleManager& GetInstance();
    ~SampleManager();

    // CLI Example: ./oclptx -s bedpostXdata/merged --simple
    //  --sampvox=2 -m bedpostXdata/nodif_brain_mask.nii.gz
    //    -x bedpostXdata/seedFile
    //
    // --simple = loading basic data form. Other types have not been
    // implemented (MANDATORY)
    // --sampvox = Sample random points within x mm sphere seed
    // voxels (MANDATORY)
    // -m = Brain Mask file (MANDATORY)
    // -x = Seedmask file (OPTIONAL). If no seedmask file is
    // specified, the program will seed using a single point:
    // the midpoint of the brainmask data.
    // TODO: Update with documentation required for Waymasks
    void ParseCommandLine(int argc, char** argv);

    //Getters: Data
    float const GetThetaData(int aFiberNum,
      int aSamp, int aX, int aY, int aZ);
    float const GetPhiData(int aFiberNum,
      int aSamp, int aX, int aY, int aZ);
    float const GetfData(int aFiberNum,
      int aSamp, int aX, int aY, int aZ);

    const unsigned short int* GetBrainMaskToArray();
    const unsigned short int* GetExclusionMaskToArray();
    const unsigned short int* GetTerminationMaskToArray();
    const std::vector<unsigned short int*> GetWayMasksToVector();

    // Getters:
    // Counts (Particles Default = 5000, Steps Default = 2000)
    int const GetNumParticles() {return _nParticles;}
    int const GetNumMaxSteps() {return _nMaxSteps;}

    // Getters: Randomly seeded particles (uses midpoint
    //  of _brainMask if no seedfile is specified)
    Fifo<struct OclPtxHandler::particle_data> *GetSeedParticles()
    {
      return _seedParticles;
    }

    // If you use these getters, you must access data from
    // the BedpostXData vector as follows:
    // Ex Theta: thetaData.data.at(aFiberNum)[(aSamp)*(nx*ny*nz) +
    // (aZ)*(nx*ny) + (aY)*nx + (aX)]
    // Where aFiberNum = the Fiber (0 or 1), aSamp = SampleNumber,
    //  nx ny nz = spacial dimensions stored in BedpostXData,
    // and aY aX, aZ = inputed spacial coordinates.
    // See definition of GetThetaData(...) above for example.
    const BedpostXData* GetThetaDataPtr();
    const BedpostXData* GetPhiDataPtr();
    const BedpostXData* GetFDataPtr();
    
    //OclptxOptions and custom options
    const oclptxOptions& GetOclptxOptions(){return _oclptxOptions;}

    // Get scaling factors for brain mask.
    cl_float4 brain_mask_dim();

  private:
    SampleManager();
    void LoadBedpostData(const std::string& aBasename);
    void LoadBedpostDataHelper(
      const std::string& aThetaSampleName,
      const std::string& aPhiSampleName,
      const std::string& afSampleName,
      const NEWIMAGE::volume<float>& aMask =
        NEWIMAGE::volume<float>(),
      const int aFiberNum = 0);
    void PopulateF(
      const NEWIMAGE::volume4D<float> aLoadedData,
      BedpostXData& aTargetContainer,
      const NEWIMAGE::volume<float> aMaskParams,
      const int aFiberNum,
      bool _16bit);
    void PopulatePHI(
      const NEWIMAGE::volume4D<float> aLoadedData,
      BedpostXData& aTargetContainer,
      const NEWIMAGE::volume<float> aMaskParams,
      const int aFiberNum,
      bool _16bit);
    void PopulateTHETA(
      const NEWIMAGE::volume4D<float> aLoadedData,
      BedpostXData& aTargetContainer,
      const NEWIMAGE::volume<float> aMaskParams,
      const int aFiberNum,
      bool _16bit);
    void GenerateSeedParticles();
    std::string IntTostring(const int& value);
    unsigned short int* GetMaskToArray(NEWIMAGE::volume<short int> aMask);
    cl_ulong8 NewRng();
    void AddSeedParticle(float x, float y, float z, float xdim, float ydim, float zdim);
    void GenerateSimpleSeeds();
    void GenerateMaskSeeds();

    //Statics
    static SampleManager* _manager;
    oclptxOptions& _oclptxOptions;
    //Seed Particles
    Fifo<struct OclPtxHandler::particle_data> *_seedParticles;
    //BedpostData
    BedpostXData _thetaData;
    BedpostXData _phiData;
    BedpostXData _fData;
    NEWIMAGE::volume<short int> _brainMask;
    NEWIMAGE::volume<short int> _exclusionMask;
    bool exclude;
    NEWIMAGE::volume<short int> _terminationMask;
    bool terminate;
    std::vector<NEWIMAGE::volume<short int>> _wayMasks;
    bool way;
    //Path Logging
    bool _showPaths;
    //Input Constants
    int _nParticles; //Default 5000
    int _nMaxSteps; //Default 2000
};

#endif  // SAMPLEMANAGER_H_
