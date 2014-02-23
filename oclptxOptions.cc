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

#define WANT_STREAM
#define WANT_MATH

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "oclptxOptions.h"
#include "utils/options.h"
//#include "newmat.h"
using namespace Utilities;

oclptxOptions* oclptxOptions::gopt = NULL;

void oclptxOptions::parse_command_line(int argc, char** argv)
{
  //Do the parsing;
  try
  {
    for(int a = options.parse_command_line(argc, argv); a < argc; a++) ;
    if(help.value() || ! options.check_compulsory_arguments())
    {
       options.usage();
       exit(2);
    }
 //AFSHIN: Add logging if needed later.

    //else{
////modecheck(); // check all the correct options are set for this mode.
//if(forcedir.value())
  //logger.setthenmakeDir(logdir.value(),"probtrackx.log");
//else
  //logger.makeDir(logdir.value(),"probtrackx.log");

//cout << "Log directory is: " << logger.getDir() << std::endl;

//// do again so that options are logged
//for(int a = 0; a < argc; a++)
  //logger.str() << argv[a] << " ";
//logger.str() << std::endl << "---------------------------------------------" << std::endl << std::endl;


  }
  catch(X_OptionError& e)
  {
    std::cerr<<e.what()<<std::endl;
    std::cerr<<"try: oclptx --help"<<std::endl;
    exit(2);
  }
}

 oclptxOptions& oclptxOptions::getInstance()
 {
    if(gopt == NULL)
       gopt = new oclptxOptions();
    return *gopt;
 }


void oclptxOptions::modecheck()
{
//     bool check=true;
//     std::string mesg="";
//     if(mode.value()=="simple"){
//       if(outfile.value()==""){
//  mesg+="You must set an output name in simple mode: -o\n";
//  check=false;
//       }
//     }


//     std::cerr<<mesg;
//     exit(2);
}



void oclptxOptions::status()
{
    cout<<"basename   "<<basename.value()<<std::endl;
    cout<<"maskfile   "<<maskfile.value()<<std::endl;
    cout<<"seeds      "<<seedfile.value()<<std::endl;
    cout<<"output     "<<outfile.value()<<std::endl;
    cout<<"verbose    "<<verbose.value()<<std::endl;
    cout<<"nparticles "<<nparticles.value()<<std::endl;
    cout<<"nsteps     "<<nsteps.value()<<std::endl;
    cout<<"usef       "<<usef.value()<<std::endl;
    cout<<"rseed      "<<rseed.value()<<std::endl;
    cout<<"randfib    "<<randfib.value()<<std::endl;
    cout<<"fibst      "<<fibst.value()<<std::endl;
}











