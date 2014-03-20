// Copyright 2014 Jeff Taylor

#include <unistd.h>
#include <cassert>
#include <thread>

#include "fifo.h"
#include "oclenv.h"
#include "oclptxhandler.h"
#include "threading.h"
#include "collatz_particle.h"

// log base 2
int lb(int x)
{
  int r = 0;
  x>>=1;
  while (x)
  {
    r++;
    x>>=1;
  }
  return r;
}

// 54 72 36 12 17 42 53 16 873 14 423
int main(int argc, char **argv)
{
  const int kStepsPerKernel = 3;
  const int kNumReducers = 2;

  if (argc < 2)
  {
    fprintf(stderr, "Usage: %s <collatz numbers>.\n", argv[0]);
    return -1;
  }

  // Add the particles to our list
  Fifo<particle::particle_data> particles(lb(argc-1)+1);
  struct particle::particle_data *data;
  for (int i = 1; i < argc; ++i)
  {
    data = new particle::particle_data;
    *data = {(uint64_t) atoi(argv[i])};
    particles.PushOrDie(data);
  }

  // Create our oclenv
  OclEnv env("collatz");

  struct particle_attrs attrs = {kStepsPerKernel, 0};

  // Create some particle buffers
  struct particles *gpu_particles = NewParticles(&env, &attrs);

  // Create a new oclptxhandler.
  OclPtxHandler handler(environment.GetContext(),
                        environment.GetCq(0),
                        environment.GetKernel(0));

  // Start up the threads.
  std::thread gpu_manager(threading::RunThreads, &gpu_particles, &handler, &particles, kNumReducers);
  gpu_manager.join();

  return 0;
}
