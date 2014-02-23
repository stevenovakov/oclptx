// Copyright 2014 Jeff Taylor
//
// Various thread functions

#include <unistd.h>

#include "oclptx/gpu.h"

#define REDUCERS_PER_GPU 1  // Duplicate in main.cc

namespace threading
{

// Threads checking in.  Kicks watchdog and lets know if program is done.  This
// function is really inefficient, so don't call it too often (cache)
bool CheckIn(char *kick)
{
  // TODO(jeff): fix this hack.
  // Note: Thread mustn't be killed until Watchdog thread notices that the
  // *all* data is complete.
  if (2 == *kick)
    return true;

  *kick = 1;
  return false;
}

// Worker thread.  Controls the GPU.
void Worker(struct global_fifos *fifos, Gpu *gpu, char *kick)
{
  // Note, there are two "sides" of GPU memory.  At all times, a kernel must
  // only access the one side.  We must only copy data to and from the
  // non-running side.
  int inactive_side = 0;
  struct collatz_data_chunk *chunk[REDUCERS_PER_GPU];

  while (1)
  {
    for (int i = 0; i < REDUCERS_PER_GPU; ++i)
    {
      chunk[i] = fifos->dirty->Pop();
      while (!chunk[i])
      {
        usleep(1000);  // TODO(jeff) don't poll
        if (CheckIn(kick))
          return;
        chunk[i] = fifos->dirty->Pop();
      }
      gpu->WriteParticles(chunk[i]);
    }

    gpu->WaitForKernel();

    gpu->RunKernelAsync(inactive_side);

    // Inactive side is now active
    inactive_side = (0 == inactive_side)? 1: 0;

    // Split the particles between threads evenly.
    int leftover_particles = gpu->particles_per_side_ % REDUCERS_PER_GPU;
    int offset = gpu->particles_per_side_ * inactive_side;
    int count;
    for (int i = 0; i < REDUCERS_PER_GPU; ++i)
    {
      count = gpu->particles_per_side_ / REDUCERS_PER_GPU;
      if (leftover_particles)
      {
        count++;
        leftover_particles--;
      }
      gpu->ReadParticles(chunk[i], offset, count);
      fifos->processed->PushOrDie(chunk[i]);
      offset += count;
    }
  }
}

// Reducer thread.
void Reducer(struct global_fifos *fifos, char *kick)
{
  struct collatz_data_chunk *chunk;
  struct collatz_data *particle;

  int finished_particles = 0;
  while (1)
  {
    chunk = fifos->processed->Pop();
    while (!chunk)
    {
      usleep(1000);  // TODO(jeff) don't poll
      if (CheckIn(kick))
        return;
      chunk = fifos->processed->Pop();
    }

    int reduced_count = 0;
    for (int i = 0; i < chunk->last; i++)
    {
      if (chunk->v[i].complete)
      {
        // Do something with the finished particle here, if we so desire.
        // It's "chunk->v[i]"
        ++finished_particles;

        particle = fifos->particles->Pop();
        if (!particle)
          break;  // No particles left.

        chunk->v[reduced_count] = *particle;
        chunk->v[reduced_count].offset = chunk->offset + i;
        ++reduced_count;

        delete particle;
      }
    }
    chunk->last = reduced_count;

    fifos->dirty->PushOrDie(chunk);
  }
}

// Watchdog thread.  Watches other threads for activity.
int Watchdog()
{
  // time_t workers_last_kick[n_gpus];
  // time_t reducers_last_kick[n_reds];
  return 0;
}

}  // namespace threading
