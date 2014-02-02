// Copyright 2014 Jeff Taylor
//
// Various thread functions

#include <unistd.h>

#include "oclptx/gpu.h"

namespace threading
{

// Threads checking in.  Kicks watchdog and lets know if program is done.  This
// function is really inefficient, so don't call it too often (cache)
bool CheckIn(char *kick)
{
  // TODO(jeff): fix this hack.
  if (2 == *kick)
    return true;

  *kick = 1;
  return false;
}

// Worker thread.  Simulates the action of the GPU
void Worker(struct global_fifos *fifos, Gpu *gpu, char *kick)
{
  struct collatz_data_chunk *chunk;

  while (1)
  {
    // TODO(jeff): How do I figure out whether we're done?  Maybe include a
    // status tracker in kick_dog to keep a running count of how many paths
    // have finished, and exit if we're done.
    if (CheckIn(kick))
      return;

    // Is there data in "leftover"?
    chunk = fifos->leftover->Pop();
    if (NULL != chunk)
    {
      gpu->WriteChunk(chunk);

      if (chunk->last)
        fifos->leftover->PushOrDie(chunk);
      else
        fifos->free->PushOrDie(chunk);
    }

    while (gpu->SpaceRemaining())
    {
      // Is there data in "reduced"?
      chunk = fifos->reduced->Pop();

      if (NULL == chunk)
        break;

      gpu->WriteChunk(chunk);

      if (chunk->last)
        fifos->leftover->PushOrDie(chunk);
      else
        fifos->free->PushOrDie(chunk);
    }

    if (gpu->SpaceUsed())
    {
      gpu->RunKernel();

      do
      {
        // Get Free chunk (or die)
        chunk = fifos->free->Pop();
        if (NULL == chunk)
        {
          puts("Error: Ran out of free buffers.");
          abort();
        }

        gpu->ReadChunk(chunk);

        fifos->processed->PushOrDie(chunk);
      } while (gpu->SpaceUsed());
    } else {
      // No data available.

      // Maybe a "wakeup sleeping workers" would be nice.
      // For now polling will have to do.
      usleep(100*1000);
    }
  }
}

// Reducer thread.
int Reducer(struct global_fifos *fifos, char *kick)
{
}

// Watchdog thread.  Watches other threads for activity.
int Watchdog()
{
  // time_t workers_last_kick[n_gpus];
  // time_t reducers_last_kick[n_reds];
}

}  // namespace threading
