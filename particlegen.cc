/* Copyright (C) 2014
 *  Afshin Haidari
 *  Steve Novakov
 *  Jeff Taylor
 */

#include "particlegen.h"

#include "fifo.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "oclptxhandler.h"
#include "oclptxOptions.h"
#include "customtypes.h"

#include <thread>

uint64_t Rand64()
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

cl_ulong8 NewRng()
{
  cl_ulong8 rng = {{0,}};
  for (int i = 0; i < 5; i++)
  {
    rng.s[i] = Rand64();
  }

  return rng;
}

ParticleGenerator::ParticleGenerator():
  particle_fifo_(NULL),
  particlegen_thread_(NULL)
{}

ParticleGenerator::~ParticleGenerator()
{
  if (particlegen_thread_)
    particlegen_thread_->join();
  delete particlegen_thread_;
  delete particle_fifo_;
}

Fifo<struct OclPtxHandler::particle_data> *ParticleGenerator::Init()
{
  oclptxOptions& opts = oclptxOptions::getInstance();
  NEWIMAGE::volume<short int> seedref;

  read_volume(seedref,opts.seedref.value());

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
  particle_fifo_ =
      new Fifo<struct OclPtxHandler::particle_data>(2 * count);

  for (int SN = 1; SN <= newSeeds.Nrows(); SN++)
  {
    float xst = newSeeds(SN, 1);
    float yst = newSeeds(SN, 2);
    float zst = newSeeds(SN, 3);
    AddSeedParticle(xst, yst, zst,
      seedref.xdim(), seedref.ydim(), seedref.zdim());
  }
  particle_fifo_->Finish();

  return particle_fifo_;
}

void ParticleGenerator::AddSeedParticle(
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
    particle_fifo_->Push(particle);

    particle = new OclPtxHandler::particle_data;
    particle->rng = NewRng();
    particle->position = pos;
    particle->dr = reverse;
    particle_fifo_->Push(particle);
  }
}

