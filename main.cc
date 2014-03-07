// Copyright 2014 Jeff Taylor
//
// Initialization routines for multi-thread testing.  Goal of this project:
//  - Check that my mult-CPU reduction model is functional, and optimal.
//  - Create a testbed for working on alternative ideas.  For example, my
//    current model does a decent amount of copying, maybe something else
//    could work?

#include <unistd.h>
#include <cassert>
#include <thread>

#include "oclptx/gpu.h"
#include "oclptx/threading.h"

// Basic log base 2
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
  const int kParticlesPerSide = 2;
  const int kStepsPerKernel = 3;
  const int kNumReducers = 2;

  if (argc < 2)
  {
    fprintf(stderr, "Usage: %s <collatz numbers>.\n", argv[0]);
    return -1;
  }

  Fifo<threading::collatz_data> particles(lb(argc-1)+1);
  struct threading::collatz_data *data;
  for (int i = 1; i < argc; ++i)
  {
    data = new threading::collatz_data;
    *data = {(uint64_t) atoi(argv[i]), 0, 0};
    particles.PushOrDie(data);
  }

  Gpu gpu(kParticlesPerSide, kStepsPerKernel);
  char kick = 0;

  // Start up the threads.
  std::thread gpu_manager(threading::RunThreads, &gpu, &particles, kNumReducers, &kick);

  // TODO(jeff): finish properly.  In the future I should be able to join() the
  // above threads.
  usleep(100 * 1000);
  kick = 2;

  gpu_manager.join();

  return 0;
}
