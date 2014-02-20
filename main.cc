// Copyright 2014 Jeff Taylor
//
// Initialization routines for multi-thread testing.  Goal of this project:
//  - Check that my mult-CPU reduction model is functional, and optimal.
//  - Create a testbed for working on alternative ideas.  For example, my
//    current model does a decent amount of copying, maybe something else
//    could work?

#include <unistd.h>
#include <thread>

#include "oclptx/gpu.h"
#include "oclptx/threading.h"

int main(int argc, char **argv)
{
  const int kParticlesPerSide = 1;
  const int kStepsPerKernel = 3;
  const int kNumReducers = 1;
  const int kNumChunks = kNumReducers * 2;
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

  // Give the GPU valid particles.
  for (int i = 0; i < kNumChunks; ++i)
  {
    // TODO(jeff): handle when num particles small
    //             handle when gpu size not divisible by particle count.
    // Better way to do this is counting *down* remaining space on GPU and
    // number of particles left.
    data = new threading::collatz_data[kChunkSize];
    for (int j = 0; j < kChunkSize; ++j)
    {
      *data = {particle[j], j % kParticlesPerSide, 0};  // value, offset, complete
    }
    chunks[i] = {data, 0, kChunkSize, kChunkSize};

    // TODO: Bad Jeff.  Passing identical pointers onto a FIFO.  Major bad.
    fifos.dirty->PushOrDie(&chunks[i]);
  }

  // Put the remainer of the particles in the particles fifo
  for (int i = kNumChunks; i < kTotNumParticles; ++i)
  {
    data = new threading::collatz_data;
    *data = {particle[i], 0, 0};
    fifos.particles->PushOrDie(data);
  }

  std::thread worker(threading::Worker, &fifos, &gpu, &kick);
  std::thread *reducers[kNumReducers];
  for (int i = 0; i < kNumReducers; ++i)
  {
    reducers[i] = new std::thread(threading::Reducer, &fifos, &kick);
  }

  sleep(1); // Assumption: data will be complete by then.
  kick = 2;  // Kill the thread, which should start polling this when it's done.

  worker.join();
  for (int i = 0; i < kNumReducers; ++i)
  {
    reducers[i]->join();
  }

  // Expected values.
  // 32 -> 16 -> 8  -> 4
  // 17 -> 52 -> 26 -> 13
  return 0;
}
