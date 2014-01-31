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

#ifndef  OCLPTX_SAMPLEMANAGER_H_
#define  OCLPTX_SAMPLEMANAGER_H_

#include "newimage/newimageall.h"
#include <iostream>
#include <vector>
#include <string>
using namespace std;
using namespace NEWIMAGE;

#define __CL_ENABLE_EXCEPTIONS
// adds exception support from CL libraries
// define before CL headers inclusion
 
class SampleManager{
	public:
      static SampleManager& GetInstance();
      ~SampleManager();
      
      void LoadBedpostData(const string& aBasename);
      //LoadBedpostData(const string& aBasename, const volume<float>& aMask);
      
      //Getters
      const vector<Matrix>* GetThetaSamples();
      const vector<Matrix>* GetPhiSamples();
      const vector<Matrix>* GetFSamples();
   
	private:
      SampleManager();
      
      void LoadBedpostDataToMatrix(const string& aThetaSampleName, const string& aPhiSampleName, const string& afSampleName, const volume<float>& aMask = volume<float>());
      string IntToString(const int& value);
      //Members
      static SampleManager* _manager;
      vector<Matrix> _thetaSamples;
      vector<Matrix> _phiSamples;
      vector<Matrix> _fSamples; 
        
};
 
#endif

//EOF
