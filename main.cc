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
  const int kNumChunks = kNumReducers;
  const int kChunkSize = kParticlesPerSide/kNumReducers;  // verify this rounds up.

  uint64_t particle[] = {54, 72, 36, 12, 17, 42, 53, 16, 873, 14, 423};
  const int kTotNumParticles = 11;

  // TODO(jeff) make these fifos the proper size
  Fifo<threading::collatz_data_chunk> processed(5);
  Fifo<threading::collatz_data_chunk> dirty(5);
  Fifo<threading::collatz_data> particles(5);

  struct threading::global_fifos fifos = {&processed, &dirty, &particles};

  char kick = 0;

  // Test routine for threading.
  Gpu gpu(kParticlesPerSide, kStepsPerKernel);

  // Create nRed * 2 chunks, each one is particles_per_side / nRed in length
  // TODO(jeff): This line is most likely going to cause cache issues.
  struct threading::collatz_data_chunk chunks[kNumChunks];
  struct threading::collatz_data *data;

  // Fill the GPU with valid particles.
  int current_particle = 0;
  int current_chunk = 0;

  while (current_particle < kParticlesPerSide)
  {
    assert(current_chunk < kNumChunks);

    data = new threading::collatz_data[kChunkSize];

    int chunk_size = 0;
    for (int j = 0; j < kChunkSize; ++j)
    {
      // Add a particle to the chunk
      if (current_particle >= kParticlesPerSide)
        break;
      data[j] = {particle[current_particle], current_particle, 0};  // value, offset, complete
      ++current_particle;
      ++chunk_size;
    }
    chunks[current_chunk] = {data, 0, chunk_size, kChunkSize};
    // The chunks are all adjacent to each other in memory, which means they
    // likely share a cache line.  That might be bad.
    fifos.dirty->PushOrDie(&chunks[current_chunk]);
    ++current_chunk;
  }

  // Put the remainer of the particles in the particles fifo
  for (int i = current_particle; i < kTotNumParticles; ++i)
  {
    data = new threading::collatz_data;
    *data = {particle[i], 0, 0};
    fifos.particles->PushOrDie(data);
  }

  // Start our threads
  std::thread worker(threading::Worker, &fifos, &gpu, &kick);
  std::thread *reducers[kNumReducers];
  for (int i = 0; i < kNumReducers; ++i)
  {
    reducers[i] = new std::thread(threading::Reducer, &fifos, &kick);
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
