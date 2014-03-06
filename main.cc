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

int main(int argc, char **argv)
{
  const int kParticlesPerSide = 1;
  const int kStepsPerKernel = 3;
  const int kNumReducers = 1;

  uint64_t particle[] = {54, 72, 36, 12, 17, 42, 53, 16, 873, 14, 423};
  const int kTotNumParticles = 11;

  // Put the particles in the particles fifo
  Fifo<threading::collatz_data> particles(4);  // 2**4-1 = 15 > 11
  struct threading::collatz_data *data;
  for (int i = 0; i < kTotNumParticles; ++i)
  {
    data = new threading::collatz_data;
    *data = {particle[i], 0, 0};
    particles.PushOrDie(data);
  }

  Gpu gpu(kParticlesPerSide, kStepsPerKernel);
  char kick = 0;

  // Start up the threads.
  std::thread gpu_manager(threading::RunThreads, &gpu, &particles, kNumReducers, &kick);

  // TODO(jeff): finish properly.  In the future I should be able to join() the
  // above threads.
  sleep(1);
  kick = 2;

  gpu_manager.join();

  return 0;
}
