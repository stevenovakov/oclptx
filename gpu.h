// Copyright 2014 Jeff Taylor
//
// GPU imitator.
// Pretends that it's a GPU, including the ability to copy data and run
// "kernels".

#ifndef GPU_H_
#define GPU_H_

#include <thread>

#include "oclptx/structs.h"

class Gpu
{
 public:
  Gpu(int particles_per_side, int steps_per_kernel);
  ~Gpu();
  void WriteParticles(struct threading::collatz_data_chunk *chunk);
  void ReadParticles(struct threading::collatz_data_chunk *chunk,
                     int offset,
                     int count);
  void RunKernel(int side);
  void RunKernelAsync(int side);
  void WaitForKernel();

  int particles_per_side_;
  int steps_per_kernel_;
 private:
  std::thread *kernel_;
  struct threading::collatz_data_chunk data_;
};

#endif  // GPU_H_
