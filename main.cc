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
  const int kChunkSize = kParticlesPerSide/kNumReducers;  // verify rounds up.

  uint64_t particle[] = {54, 72, 36, 12, 17, 42, 53, 16, 873, 14, 423};
  const int kTotNumParticles = 11;

  // TODO(jeff) make this fifo the proper size
  Fifo<threading::collatz_data> particles(5);

  // TODO(jeff) kick is still a hack
  char kick = 0;

  Gpu gpu(kParticlesPerSide, kStepsPerKernel);

  // Create nRed chunks, each one is particles_per_side / nRed in length
  struct threading::shared_data sdata[kNumReducers];
  struct threading::collatz_data *data;

  for (int i = 0; i < kNumReducers; ++i)
  {
    sdata[i].kick = &kick;
    sdata[i].data_ready = false;
    sdata[i].reduction_complete = false;
  }

  // Fill the GPU with valid particles.
  int current_particle = 0;
  int current_chunk = 0;

  while (current_particle < kParticlesPerSide)
  {
    assert(current_chunk < kNumReducers);

    data = new threading::collatz_data[kChunkSize];

    int chunk_size = 0;
    for (int j = 0; j < kChunkSize; ++j)
    {
      // Add a particle to the chunk
      if (current_particle >= kParticlesPerSide)
        break;
      data[j] = {particle[current_particle], current_particle, 0};
      ++current_particle;
      ++chunk_size;
    }
    sdata[current_chunk].chunk = {data, 0, chunk_size, kChunkSize};
    sdata[current_chunk].reduction_complete = true;
    ++current_chunk;
  }

  // Put the remainer of the particles in the particles fifo
  for (int i = current_particle; i < kTotNumParticles; ++i)
  {
    data = new threading::collatz_data;
    *data = {particle[i], 0, 0};
    particles.PushOrDie(data);
  }

  // Start our threads
  std::thread worker(threading::Worker, sdata, &gpu, &kick);
  std::thread *reducers[kNumReducers];
  for (int i = 0; i < kNumReducers; ++i)
  {
    reducers[i] = new std::thread(threading::Reducer, &sdata[i], &particles);
  }

  // TODO(jeff): finish properly
  // Hack: Assume everything will be done in one second, and kill the threads.
  sleep(1);
  kick = 2;

  worker.join();
  for (int i = 0; i < kNumReducers; ++i)
  {
    reducers[i]->join();
    delete reducers[i];
  }

  return 0;
}
