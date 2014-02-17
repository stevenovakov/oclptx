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



#include <fstream>
#include <sstream>


#define __CL_ENABLE_EXCEPTIONS
// adds exception support from CL libraries
// define before CL headers inclusion

#include "samplemanager.h"
#include "oclptxOptions.h"

//
// Assorted Functions Declerations
//

std::string SampleManager::IntTostring(const int& value)
{
   std::stringstream s;
   s << value;
   return s.str();
}

//Private method: Used for loading data directly into member containers
void SampleManager::LoadBedpostDataToMatrix(
  const std::string& aThetaSampleName,
  const std::string& aPhiSampleName,
  const std::string& afSampleName,
  const volume<float>& aMask  )
{
   volume4D<float> loadedVolume4DTheta;
   volume4D<float> loadedVolume4DPhi;
   volume4D<float> loadedVolume4Df;

   //Load Theta/Phi/f samples
   NEWIMAGE::read_volume4D(loadedVolume4DTheta, aThetaSampleName);
   NEWIMAGE::read_volume4D(loadedVolume4DPhi, aPhiSampleName);
   NEWIMAGE::read_volume4D(loadedVolume4Df, afSampleName);

   if(aMask.xsize() > 0)
   {
      _thetaSamples.push_back(loadedVolume4DTheta.matrix(aMask));
      _phiSamples.push_back(loadedVolume4DPhi.matrix(aMask));
      _fSamples.push_back(loadedVolume4Df.matrix(aMask));
   }
   else
   {
      _thetaSamples.push_back(loadedVolume4DTheta.matrix());
      _phiSamples.push_back(loadedVolume4DPhi.matrix());
      _fSamples.push_back(loadedVolume4Df.matrix());
   }
}

//Loading BedpostData: No Masks.
void SampleManager::LoadBedpostData(const std::string& aBasename)
{
   std::cout<<"Loading Bedpost samples....."<<std::endl;
   if(aBasename == "")
   {
      std::cout<< "Bad File Name"<<std::endl;
      return;
   }

   //volume4D<float> loadedVolume4D;
   std::string thetaSampleNames;
   std::string phiSampleNames;
   std::string fSampleNames;

   //Single Fiber Case.
   if(NEWIMAGE::fsl_imageexists(aBasename+"_thsamples"))
   {
      thetaSampleNames = aBasename+"_thsamples";
      phiSampleNames = aBasename+"_phisamples";
      fSampleNames = aBasename+"_fsamples";
      LoadBedpostDataToMatrix(
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

         LoadBedpostDataToMatrix(
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
   }
   else if (_oclptxOptions.network.value())
   {
      std::cout<<"Running in network mode"<<std::endl;
   }
   else
   {
      std::cout<<"Running in seedmask mode"<<std::endl;
   }
}

//AFSHIN TODO: Implement Masks

const vector<Matrix>* SampleManager::GetThetaSamples()
{
   if(_thetaSamples.size() > 0)
   {
      return &_thetaSamples;
   }
   return NULL;
}

const vector<Matrix>* SampleManager::GetPhiSamples()
{
   if(_phiSamples.size() > 0)
   {
      return &_phiSamples;
   }
   return NULL;;
}

const vector<Matrix>* SampleManager::GetFSamples()
{
   if(_fSamples.size() > 0)
   {
      return &_fSamples;
   }
   return NULL;;
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
  oclptxOptions::getInstance()){}

SampleManager::~SampleManager()
{
   delete _manager;
}


//EOF
