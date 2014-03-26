// Copyright 2014 Jeff Taylor
//
// Various thread functions
//

#include <fcntl.h>
#include <unistd.h>
#include <condition_variable>

#include "fifo.h"
#include "oclptxhandler.h"

namespace threading
{

FILE *global_fd;

struct shared_data {
  int chunk_size;  // Amount of space allocated
  int count;  // Number of occupied elements

  struct OclPtxHandler::particle_data *chunk;
  cl_ushort *complete;
  int *particle_offset;
  int chunk_offset;
  std::mutex data_lock;


  bool data_ready;
  std::condition_variable data_ready_cv;

  bool reduction_complete;
  std::condition_variable reduction_complete_cv;

  bool done;
  bool has_data;
};

// Worker thread.  Controls the GPU.
void Worker(struct shared_data *sdata, OclPtxHandler *handler, int num_reducers)
{
  // Note, there are two "sides" of GPU memory.  At all times, a kernel must
  // only access the one side.  We must only copy data to and from the
  // non-running side.
  int inactive_side = 0;
  bool has_data_side[2] = {true, true};
  std::unique_lock<std::mutex> *lk[num_reducers];

  while (1)
  {
    // If no data on either side, we're done!
    if (!has_data_side[0] && !has_data_side[1])
    {
      for (int i = 0; i < num_reducers; ++i)
      {
        // Wake up reducers and have them exit.
        std::unique_lock<std::mutex> lock(sdata[i].data_lock);
        sdata[i].done = true;
        sdata[i].data_ready_cv.notify_one();
        lock.unlock();
      }
      return;
    }

    has_data_side[inactive_side] = false;
    for (int i = 0; i < num_reducers; ++i)
    {
      lk[i] = new std::unique_lock<std::mutex>(sdata[i].data_lock);
      while (!sdata[i].reduction_complete)
      {
        sdata[i].reduction_complete_cv.wait_for(*lk[i],
                                            std::chrono::milliseconds(100));
      }
      sdata[i].reduction_complete = false;

      if (sdata[i].has_data)
      {
        has_data_side[inactive_side] = true;
        for (int j = 0; j < sdata[i].count; ++j)
          handler->WriteParticle(
              (sdata[i].chunk + j),
              sdata[i].particle_offset[j]);
      }
    }

    handler->RunKernel(inactive_side);

    // Inactive side is now active
    inactive_side = (0 == inactive_side)? 1: 0;

    // Split the particles between threads evenly.
    // This is a work-queue style of operation, which is relatively
    // standard, but that's not obvious.  Making it more obvious would greatly
    // improve readability.
    int leftover_particles = handler->particles_per_side() % num_reducers;
    int offset = handler->particles_per_side() * inactive_side;
    int count;
    for (int i = 0; i < num_reducers; ++i)
    {
      // We still have data lock i
      count = handler->particles_per_side() / num_reducers;
      if (leftover_particles)
      {
        count++;
        leftover_particles--;
      }
      handler->ReadStatus(offset, count, sdata[i].complete);

      sdata[i].data_ready = true;
      sdata[i].count = count;
      sdata[i].chunk_offset = offset;
      sdata[i].data_ready_cv.notify_one();

      offset += count;

      delete lk[i];
    }

    // Dump all paths
    handler->DumpPath(inactive_side * handler->particles_per_side(),
                      handler->particles_per_side(),
                      global_fd);
  }
}

void Reducer(
    struct shared_data *sdata,
    Fifo<OclPtxHandler::particle_data> *particles)
{
  struct OclPtxHandler::particle_data *particle;
  int reduced_count;

  while (1)
  {
    // Wait for data to be ready.
    std::unique_lock<std::mutex> lk(sdata->data_lock);
    while (!sdata->data_ready)
    {
      sdata->data_ready_cv.wait_for(lk, std::chrono::milliseconds(100));
      if (sdata->done)
        return;
    }
    sdata->data_ready = false;

    // Do the actual reduction
    reduced_count = 0;
    sdata->has_data = false;
    for (int i = 0; i < sdata->count; i++)
    {
      if (sdata->complete[i])
      {
        // Do something with the finished particle here, if we so desire.
        // It's "chunk.v[i]".  Note: the first few particles will be
        // unprocessed garbage.  I might need to add an extra state to the
        // status buffer.

        // New particle.
        particle = particles->Pop();
        if (!particle)
          break;  // No particles left.

        sdata->chunk[reduced_count] = *particle;
        sdata->particle_offset[reduced_count] = sdata->chunk_offset + i;
        ++reduced_count;

        delete particle;
      }

      sdata->has_data = true;
    }
    sdata->count = reduced_count;

    sdata->reduction_complete = true;
    sdata->reduction_complete_cv.notify_one();
    lk.unlock();
  }
}

void RunThreads(
    OclPtxHandler *handler,
    Fifo<OclPtxHandler::particle_data> *particles,
    int num_reducers)
{
  // Push blank data with complete=1 to reducer.  It will fill it in with
  // particles.
  int leftover_particles = handler->particles_per_side() % num_reducers;
  int chunk_size = handler->particles_per_side() / num_reducers + 1;

  int offset = 0;
  int count;
  struct shared_data sdata[num_reducers];
  struct OclPtxHandler::particle_data *data;
  cl_ushort *status;
  int *particle_offset;

  global_fd = fopen("./path_output", "w");
  if (NULL == global_fd)
  {
    perror("Couldn't open file");
    exit(1);
  }

  for (int i = 0; i < num_reducers; ++i)
  {
    count = handler->particles_per_side() / num_reducers;
    if (leftover_particles)
    {
      count++;
      leftover_particles--;
    }

    data = new OclPtxHandler::particle_data[chunk_size];
    status = new cl_ushort[chunk_size];
    particle_offset = new int[chunk_size];

    for (int j = 0; j < chunk_size; ++j)
      status[j] = true;  // complete = 1

    sdata[i].chunk = data;
    sdata[i].chunk_offset = offset;
    sdata[i].complete = status;
    sdata[i].particle_offset = particle_offset;
    sdata[i].count = count;
    sdata[i].chunk_size = chunk_size;

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
    reducers[i] = new std::thread(Reducer, &sdata[i], particles);
  }
  Worker(sdata, handler, num_reducers);

  // Clean everything up.
  for (int i = 0; i < num_reducers; ++i)
  {
    reducers[i]->join();
    delete reducers[i];
    delete[] sdata[i].particle_offset;
    delete[] sdata[i].complete;
    delete[] sdata[i].chunk;
  }

  fclose(global_fd);
}

}  // namespace threading
