// Copyright 2014 Jeff Taylor
//
// GPU imitator.
// Pretends that it's a GPU, including the ability to copy data and run
// "kernels".

#ifndef GPU_H_
#define GPU_H_

#include "structs.h"

class Gpu
{
 public:
  Gpu(int particles_per_chunk, int steps_per_kernel);
  ~Gpu();
  void WriteChunk(struct threading::collatz_data_chunk *chunk);
  void ReadChunk(struct threading::collatz_data_chunk *chunk);
  void RunKernel();
  int SpaceRemaining();
  int SpaceUsed();
 private:
  struct threading::collatz_data_chunk data_;
  int particles_per_chunk_ = 1;
  int steps_per_kernel_ = 3;
};

#endif  // GPU_H_
