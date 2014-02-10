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
  // Test routine for threading.
  Gpu gpu(1, 3);

  struct threading::collatz_data data[2] = {{17, 0}, {32, 0}};
  struct threading::collatz_data_chunk chunk = {data, 2, 2};

  struct threading::collatz_data data2[2];
  struct threading::collatz_data_chunk free_chunk = {data2, 0, 2};

  Fifo<threading::collatz_data_chunk> to_process(2);  // 3 items
  Fifo<threading::collatz_data_chunk> leftover(2);
  Fifo<threading::collatz_data_chunk> processed(2);
  Fifo<threading::collatz_data_chunk> free(2);

  struct threading::global_fifos fifos = {&to_process, &processed, &leftover, &free};

  char kick = 0;

  to_process.PushOrDie(&chunk);
  free.PushOrDie(&free_chunk);

  std::thread worker(threading::Worker, &fifos, &gpu, &kick);

  sleep(1); // Assumption: data will be complete by then.
  kick = 2;  // Kill the thread, which should start polling this when it's done.

  worker.join();

  return 0;

  // Set up my FIFOs:
  //  - Reduced: Length >= max(2*nGPUs,2*nRedThreads)
  //  - Leftover: Length >= nGPUs
  //  - Processed: Length >= 2*nGPUs
  //  - Free: >= Buffercount

  // Create some sample data, load it into "reduced" fifo.  Needs to be split
  // into 2*nGPUs chunks.  Keep track of how many seeds that *actually* is---
  // that's going to change as we run.  Maybe we'll exponentially drop the
  // number we queue onto the GPU?  Needs thought.
  // And some empty buffers for us to use later.

  // Kick off some threads.  nGPU worker threads and nRed reducers.

  // This thread turns into a watchdog.  We need to kill the threads if they
  // hang, after, say, 1 second.
  threading::Watchdog();

  return 0;
}
