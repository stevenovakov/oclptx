// Copyright 2014 Jeff Taylor

#ifndef STRUCTS_H_
#define STRUCTS_H_

#include <stdint.h>
#include "oclptx/fifo.h"

namespace threading
{

// Data contained in a single instance of the collatz problem
struct collatz_data
{
  uint64_t value;
  bool complete;
};

struct collatz_data_chunk
{
  struct collatz_data *v;
  int last;
  int size;
};

// Global FIFOs.  These are used in a number of places, and contain
// threadsafe methods.
struct global_fifos
{
  class Fifo<struct collatz_data_chunk> *reduced;
  class Fifo<struct collatz_data_chunk> *processed;
  class Fifo<struct collatz_data_chunk> *leftover;
  class Fifo<struct collatz_data_chunk> *free;
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
