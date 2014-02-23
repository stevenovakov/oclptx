// Copyright 2014 Jeff Taylor

#include <cassert>
#include <cstring>

#include "oclptx/gpu.h"
#include "oclptx/structs.h"

Gpu::Gpu(int particles_per_side, int steps_per_kernel):
  particles_per_side_(particles_per_side),
  steps_per_kernel_(steps_per_kernel),
  kernel_(NULL)
{
  data_.last = 0;
  data_.size = 2 * particles_per_side;
  data_.v = new threading::collatz_data[2 * particles_per_side];
}

Gpu::~Gpu()
{
  delete data_.v;
  if (kernel_)
  {
    kernel_->join();
    delete kernel_;
  }
}

void Gpu::WriteParticles(struct threading::collatz_data_chunk *chunk)
{
  size_t off;
  for (int i = 0; i < chunk->last; i++)
  {
    off = chunk->v[i].offset;
    data_.v[off] = chunk->v[i];
  }
}

void Gpu::ReadParticles(struct threading::collatz_data_chunk *chunk,
                        size_t offset,
                        size_t count)
{
  assert(count <= chunk->size);

  void *src = reinterpret_cast<void*>(data_.v + offset);
  memcpy(reinterpret_cast<void*>(chunk->v),
          src,
          sizeof(threading::collatz_data) * count);

  chunk->last = count;
}

void Gpu::RunKernelAsync(int side)
{
  assert(NULL == kernel_);

  // kludge(jeff): Is there anyway to start the thread after the thread object
  // is created?
  kernel_ = new std::thread(&Gpu::RunKernel, this, side);
}

void Gpu::WaitForKernel()
{
  if (NULL == kernel_)
    return;  // No kernel to wait for.

  // TODO(jeff): Might be nice to have a timer run here.
  kernel_->join();
  delete kernel_;
  kernel_ = NULL;
}

void Gpu::RunKernel(int side)
{
  for (int i = 0; i < particles_per_side_; i++)
  {
    size_t off = i + particles_per_side_ * side;
    for (int j = 0; j < steps_per_kernel_; j++)
    {
      // Collatz conjecture.
      if (data_.v[off].value & 1)
        data_.v[off].value = data_.v[off].value * 3 + 1;
      else
        data_.v[off].value = data_.v[off].value >> 1;

      // End of the path?
      if (1 == data_.v[off].value)
        data_.v[off].complete = 1;
    }
  }
}
