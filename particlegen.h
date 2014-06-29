/* Copyright (C) 2014
 *  Afshin Haidari
 *  Steve Novakov
 *  Jeff Taylor
 *
 * Manages particle generation.  Init returns a FIFO (ie a stream) which can
 * continuously Pop() particles from multiple threads.
 */

#include "fifo.h"
#include "newimage/newimageall.h"
#include "oclptxhandler.h"

#include <thread>

class ParticleGenerator
{
 public:
  ParticleGenerator();
  ~ParticleGenerator();
  Fifo<struct OclPtxHandler::particle_data> *Init();
 private:
  Fifo<struct OclPtxHandler::particle_data> *particle_fifo_;
  std::thread *particlegen_thread_;
  char *coords_filename_;

  void AddParticles(float* newSeeds, int count, float xdim, float ydim, float zdim);
  void AddSeedParticle(float x, float y, float z,
    float xdim, float ydim, float zdim);
};

