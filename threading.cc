// Copyright 2014 Jeff Taylor
//
// Various thread functions
//

#include <unistd.h>

#include "oclptx/gpu.h"

#define REDUCERS_PER_GPU 1  // Duplicate in main.cc

namespace threading
{

// Threads checking in.  Kicks watchdog and lets know if program is done.  This
// function is really inefficient, so don't call it too often (cache)
// Return true: we're all done.
bool CheckIn(char *kick)
{
  if (2 == *kick)
    return true;

  *kick = 1;
  return false;
}

// Worker thread.  Controls the GPU.
void Worker(struct shared_data *p, Gpu *gpu, char *kick)
{
  // Note, there are two "sides" of GPU memory.  At all times, a kernel must
  // only access the one side.  We must only copy data to and from the
  // non-running side.
  int inactive_side = 0;
  std::unique_lock<std::mutex> *lk[REDUCERS_PER_GPU];

  while (1)
  {
    for (int i = 0; i < REDUCERS_PER_GPU; ++i)
    {
      lk[i] = new std::unique_lock<std::mutex>(p[i].data_lock);
      while (!p[i].reduction_complete)
      {
        p[i].reduction_complete_cv.wait_for(*lk[i],
                                            std::chrono::milliseconds(100));
        if (CheckIn(kick))
          return;
      }
      p[i].reduction_complete = false;

      gpu->WriteParticles(&p[i].chunk);
    }

    gpu->WaitForKernel();

    gpu->RunKernelAsync(inactive_side);

    // Inactive side is now active
    inactive_side = (0 == inactive_side)? 1: 0;

    // Split the particles between threads evenly.
    // This is a work-queue style of operation, which is relatively
    // standard, but that's not obvious.  Making it more obvious would greatly
    // improve readability.
    int leftover_particles = gpu->particles_per_side_ % REDUCERS_PER_GPU;
    int offset = gpu->particles_per_side_ * inactive_side;
    int count;
    for (int i = 0; i < REDUCERS_PER_GPU; ++i)
    {
      // We still have data lock i
      count = gpu->particles_per_side_ / REDUCERS_PER_GPU;
      if (leftover_particles)
      {
        count++;
        leftover_particles--;
      }
      gpu->ReadParticles(&p[i].chunk, offset, count);
      offset += count;

      p[i].data_ready = true;
      p[i].data_ready_cv.notify_one();
      delete lk[i];
    }
  }
}

// Reducer thread.
// TODO(jeff) here: separate IN and OUT.  Having them shared is fairly ugly.
// I'm really close to having this fixed.
void Reducer(struct shared_data *p, Fifo<threading::collatz_data> *particles)
{
  struct collatz_data *particle;
  int reduced_count;

  while (1)
  {
    // Wait for data to be ready.
    std::unique_lock<std::mutex> lk(p->data_lock);
    while (!p->data_ready)
    {
      p->data_ready_cv.wait_for(lk, std::chrono::milliseconds(100));
      if (CheckIn(p->kick))
        return;
    }
    p->data_ready = false;

    // Do the actual reduction
    reduced_count = 0;
    for (int i = 0; i < p->chunk.last; i++)
    {
      if (p->chunk.v[i].complete)
      {
        // Do something with the finished particle here, if we so desire.
        // It's "chunk.v[i]"

        // New particle.
        particle = particles->Pop();
        if (!particle)
          break;  // No particles left.

        p->chunk.v[reduced_count] = *particle;
        p->chunk.v[reduced_count].offset = p->chunk.offset + i;
        ++reduced_count;

        delete particle;
      }
    }
    p->chunk.last = reduced_count;

    p->reduction_complete = true;
    p->reduction_complete_cv.notify_one();
    lk.unlock();
  }
}

}  // namespace threading
