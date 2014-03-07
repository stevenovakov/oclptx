// Copyright 2014 Jeff Taylor
//
// Various thread functions
//

#include <unistd.h>

#include "oclptx/gpu.h"

namespace threading
{

struct shared_data {
  struct collatz_data_chunk chunk;  // TODO(jeff) split two directions
  std::mutex data_lock;

  bool data_ready;
  std::condition_variable data_ready_cv;

  bool reduction_complete;
  std::condition_variable reduction_complete_cv;

  bool done;
  bool has_data;
};

// Worker thread.  Controls the GPU.
void Worker(struct shared_data *p, Gpu *gpu, int num_reducers)
{
  // Note, there are two "sides" of GPU memory.  At all times, a kernel must
  // only access the one side.  We must only copy data to and from the
  // non-running side.
  int inactive_side = 0;
  bool has_data_side[2] = {true,true};
  std::unique_lock<std::mutex> *lk[num_reducers];

  while (1)
  {
    // If no data on either side, we're done!
    if (!has_data_side[0] && !has_data_side[1])
    {
      for (int i = 0; i < num_reducers; ++i)
      {
        // Wake up reducers and have them exit.
        std::unique_lock<std::mutex> lock(p[i].data_lock);
        p[i].done = true;
        p[i].data_ready_cv.notify_one();
        lock.unlock();
      }
      return;
    }

    has_data_side[inactive_side] = false;
    for (int i = 0; i < num_reducers; ++i)
    {
      lk[i] = new std::unique_lock<std::mutex>(p[i].data_lock);
      while (!p[i].reduction_complete)
      {
        p[i].reduction_complete_cv.wait_for(*lk[i],
                                            std::chrono::milliseconds(100));
      }
      p[i].reduction_complete = false;

      if (p[i].has_data)
      {
        has_data_side[inactive_side] = true;
        gpu->WriteParticles(&p[i].chunk);
      }
    }

    gpu->WaitForKernel();

    gpu->RunKernelAsync(inactive_side);

    // Inactive side is now active
    inactive_side = (0 == inactive_side)? 1: 0;

    // Split the particles between threads evenly.
    // This is a work-queue style of operation, which is relatively
    // standard, but that's not obvious.  Making it more obvious would greatly
    // improve readability.
    int leftover_particles = gpu->particles_per_side_ % num_reducers;
    int offset = gpu->particles_per_side_ * inactive_side;
    int count;
    for (int i = 0; i < num_reducers; ++i)
    {
      // We still have data lock i
      count = gpu->particles_per_side_ / num_reducers;
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
void Reducer(struct shared_data *p, Fifo<collatz_data> *particles)
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
      if (p->done)
        return;
    }
    p->data_ready = false;

    // Do the actual reduction
    reduced_count = 0;
    p->has_data = false;
    for (int i = 0; i < p->chunk.last; i++)
    {
      if (p->chunk.v[i].complete)
      {
        // Do something with the finished particle here, if we so desire.
        // It's "chunk.v[i]".

        // New particle.
        particle = particles->Pop();
        if (!particle)
          break;  // No particles left.

        p->chunk.v[reduced_count] = *particle;
        p->chunk.v[reduced_count].offset = p->chunk.offset + i;
        ++reduced_count;
        p->has_data = true;

        delete particle;
      }
      else
        p->has_data = true;
    }
    p->chunk.last = reduced_count;

    p->reduction_complete = true;
    p->reduction_complete_cv.notify_one();
    lk.unlock();
  }
}

void RunThreads(Gpu *gpu, Fifo<collatz_data> *particles, int num_reducers)
{
  // Push blank data with complete=1 to reducer.  It will fill it in with
  // particles.
  int leftover_particles = gpu->particles_per_side_ % num_reducers;
  int chunk_size = gpu->particles_per_side_ / num_reducers + 1;

  int offset = 0;
  int count;
  struct threading::shared_data sdata[num_reducers];
  struct threading::collatz_data *data;
  for (int i = 0; i < num_reducers; ++i)
  {
    data = new threading::collatz_data[chunk_size];
    for (int j = 0; j < chunk_size; ++j)
      data[j] = {0, 0, 1}; // No data, complete = 1

    count = gpu->particles_per_side_ / num_reducers;
    if (leftover_particles)
    {
      count++;
      leftover_particles--;
    }

    sdata[i].chunk = {data, offset, count, chunk_size};
    sdata[i].data_ready = true;

    sdata[i].reduction_complete = false;
    sdata[i].done = false;
    sdata[i].has_data = true;

    offset += count;
  }

  // Start our threads
  std::thread *reducers[num_reducers];
  for (int i = 0; i < num_reducers; ++i)
  {
    reducers[i] = new std::thread(threading::Reducer, &sdata[i], particles);
  }
  Worker(sdata, gpu, num_reducers);


  for (int i = 0; i < num_reducers; ++i)
  {
    reducers[i]->join();
    delete reducers[i];
    delete sdata[i].chunk.v;
  }
}

}  // namespace threading
