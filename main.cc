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
  const int kNumReducers = 1;

  if (argc < 2)
  {
    fprintf(stderr, "Usage: %s <collatz numbers>.\n", argv[0]);
    return -1;
  }

  // Add the particles to our list
  Fifo<struct OclPtxHandler::particle_data> particles_fifo(lb(argc-1)+1);
  struct OclPtxHandler::particle_data *data;
  for (int i = 1; i < argc; ++i)
  {
    data = new OclPtxHandler::particle_data;
    *data = {(uint64_t) atoi(argv[i])};
    particles_fifo.PushOrDie(data);
  }

  // Create our oclenv
  OclEnv env("collatz");
  env.Init();

  struct OclPtxHandler::particle_attrs attrs = {kStepsPerKernel, 0};

  // Create a new oclptxhandler.
  OclPtxHandler handler;
  handler.Init(env.GetContext(),
               env.GetCq(0),
               env.GetKernel(0),
               NULL, NULL, NULL, 0, NULL, // 5 bedpost data values.
               &attrs);

  // Start up the threads.
  std::thread gpu_manager(threading::RunThreads, &handler, &particles_fifo, kNumReducers);
  gpu_manager.join();

  return 0;
}
