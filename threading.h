// Copyright 2014 Jeff Taylor
//
// Various thread functions

#ifndef THREADING_H_
#define THREADING_H_

#include <stdint.h>
#include <condition_variable>
#include "oclptx/fifo.h"

class Gpu;

namespace threading
{
// TODO(jeff): I'm using the same struct to represent both the "dirty" particle
// indicator, and the whole GPU memory copy.  This is pretty sloppy and makes,
// the code needlessly confusing, but simplifies memory management.  I'll need
// to think of a way to clear this up, likely by keeping free buffer lists for
// the two new types.

// Data contained in a single instance of the collatz problem
struct collatz_data
{
  uint64_t value;  // Dirty list
  int offset;  // Dirty list: This parameter specifically can NOT exist on GPU.
  int complete;  // GPU mirror
};

// List of the above, with supplementary context
struct collatz_data_chunk
{
  struct collatz_data *v;
  int offset;  // For GPU mirror
  int last;  // For Both
  int size;  // For Both.  Mainly a sanity check.
};

void RunThreads(Gpu *gpu, Fifo<collatz_data> *particles, int num_reducers);

}  // namespace threading

#endif  // THREADING_H_
