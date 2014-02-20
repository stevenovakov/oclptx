// Copyright 2014 Jeff Taylor

#ifndef STRUCTS_H_
#define STRUCTS_H_

#include <stdint.h>
#include "oclptx/fifo.h"

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
  uint64_t value; // Dirty list
  size_t offset; // Dirty list
  int complete; // GPU mirror
};

// List of the above, with supplementary context
struct collatz_data_chunk
{
  struct collatz_data *v;
  size_t offset;  // For GPU mirror
  size_t last;  // For Both
  size_t size;  // For Both.  Mainly a sanity check.
};

// Global FIFOs.  These are used in a number of places, and contain
// threadsafe methods.
struct global_fifos
{
  class Fifo<struct collatz_data_chunk> *dirty;
  class Fifo<struct collatz_data_chunk> *processed;
  class Fifo<struct collatz_data> *particles;
};

// The kicker is used to inform the watchdog that this thread is alive and
// well, and hasn't hung.
struct global_kicker
{
  bool *worker_kick;
  bool *reducer_kick;
};

}  // namespace threading

#endif  // STRUCTS_H_
