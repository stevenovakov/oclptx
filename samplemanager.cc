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
#include <vector>


#define __CL_ENABLE_EXCEPTIONS 
// adds exception support from CL libraries
// define before CL headers inclusion

#include "samplemanager.h"

//
// Assorted Functions Declerations
//

string SampleManager::IntToString(const int& value)
{
   stringstream s;
   s << value;
   return s.str();
}

//Private method: Used for loading data directly into member containers
void SampleManager::LoadBedpostDataToMatrix(const string& aThetaSampleName, const string& aPhiSampleName, const string& afSampleName, const volume<float>& aMask)
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
void SampleManager::LoadBedpostData(const string& aBasename)
{
   cout<<"Loading Bedpost samples....."<<endl;
   if(aBasename == "")
   {
      cout<< "Bad File Name"<<endl;
      return;
   }
   
   //volume4D<float> loadedVolume4D;
   string thetaSampleNames;
   string phiSampleNames; 
   string fSampleNames;
   
   //Single Fiber Case.
   if(NEWIMAGE::fsl_imageexists(aBasename+"_thsamples"))
   {
      thetaSampleNames = aBasename+"_thsamples";
      phiSampleNames = aBasename+"_phisamples";
      fSampleNames = aBasename+"_fsamples";
      LoadBedpostDataToMatrix(thetaSampleNames,phiSampleNames,fSampleNames);
   }
   //Multiple Fiber Case.
   else
   {
      int fiberNum = 1;       
      string fiberNumAsString = IntToString(fiberNum);
      thetaSampleNames = aBasename+"_th"+fiberNumAsString+"samples";
      bool doesFiberExist = NEWIMAGE::fsl_imageexists(thetaSampleNames);
      while(doesFiberExist)
      {
         phiSampleNames = aBasename+"_ph"+fiberNumAsString+"samples";
         fSampleNames = aBasename+"_f"+fiberNumAsString+"samples";
         
         LoadBedpostDataToMatrix(thetaSampleNames,phiSampleNames,fSampleNames);

         fiberNum++;
         fiberNumAsString = IntToString(fiberNum);
         thetaSampleNames = aBasename+"_th"+fiberNumAsString+"samples";
         doesFiberExist = NEWIMAGE::fsl_imageexists(thetaSampleNames);
      }
      if(fiberNum == 1)
      {
         cout<<"Could not find samples"<<endl;
         exit(1);
      }
      cout<<"Finished Loading Samples from Bedpost"<<endl;
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
}

//Private Constructor.
SampleManager::SampleManager(){}

SampleManager::~SampleManager()
{
   delete _manager;
}


//EOF
