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

Fifo<struct OclPtxHandler::particle_data> *ParticleGenerator::Init(int fifo_size)
{
  oclptxOptions& opts = oclptxOptions::getInstance();
  NEWIMAGE::volume<short int> seedref;

  read_volume(seedref,opts.seedref.value());

  NEWMAT::Matrix Seeds = read_ascii_matrix(opts.seedfile.value());
  if (Seeds.Ncols() != 3 && Seeds.Nrows() == 3)
    Seeds = Seeds.t();

  total_particles_ = 2 * opts.nparticles.value() * Seeds.Nrows();

  float *newSeeds = new float[Seeds.Nrows() * 3];

  // convert coordinates from nifti (external) to newimage (internal)
  //   conventions - Note: for radiological files this should do nothing
  for (int n = 0; n < Seeds.Nrows(); n++)
  {
    NEWMAT::ColumnVector v(4);
    v << Seeds(1+n,1) << Seeds(1+n,2) << Seeds(1+n,3) << 1.0;
    v = seedref.niftivox2newimagevox_mat() * v;

    newSeeds[3*n]   = v(1);
    newSeeds[3*n+1] = v(2);
    newSeeds[3*n+2] = v(3);
  }

  particle_fifo_ =
      new Fifo<struct OclPtxHandler::particle_data>(fifo_size);

  // TODO(jeff): First off: what are Seeds/seedref?  Second off: Copying like
  // this might be bad.  Figure out the values we need explicitly, then copy
  // those in.
  particlegen_thread_ = new std::thread([=]
    { AddParticles(newSeeds, Seeds.Nrows(),
                   seedref.xdim(), seedref.ydim(), seedref.zdim()); });

  return particle_fifo_;
}

int64_t ParticleGenerator::total_particles()
{
  return total_particles_;
}

void ParticleGenerator::AddParticles(float* newSeeds, int count,
                                     float xdim, float ydim, float zdim)
{
  float x, y, z;
  for (int i = 0; i < count; ++i)
  {
    x = newSeeds[3*i];
    y = newSeeds[3*i+1];
    z = newSeeds[3*i+2];
    AddSeedParticle(x, y, z, xdim, ydim, zdim);
  }
  particle_fifo_->Finish();
  delete[] newSeeds;
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

