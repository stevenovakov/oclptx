// Copyright 2014 Jeff Taylor

#include "oclptx/gpu.h"

Gpu::Gpu(int particles_per_chunk, int steps_per_kernel)
{
  particles_per_chunk_ = particles_per_chunk;
  steps_per_kernel_ = steps_per_kernel;
  data_.v = new threading::collatz_data[1];
  data_.last = 0;
  data_.size = 1;
}

Gpu::~Gpu()
{
  delete data_.v;
}

void Gpu::WriteChunk(struct threading::collatz_data_chunk *chunk)
{
  size_t space_remaining = data_.size - data_.last;

  // TODO(jeff): Cleanup this copy to look like a proper mem copy.
  // This actually reverses the data a la Towers of Hanoi, which I'd like to
  // avoid.  It's not "incorrect" persay, but makes debugging way harder.
  while ((data_.size != data_.last) && (0 != chunk->last))
  {
    data_.v[data_.last] = chunk->v[chunk->last - 1];

    chunk->last--;
    data_.last++;
  }
}

void Gpu::ReadChunk(struct threading::collatz_data_chunk *chunk)
{
  size_t space_remaining = chunk->size - chunk->last;

  // TODO(jeff): See above comment.
  while ((0 != data_.last) && (chunk->size != chunk->last))
  {
    chunk->v[chunk->last] = data_.v[data_.last - 1];

    chunk->last++;
    data_.last--;
  }
}

int Gpu::SpaceRemaining()
{
  return data_.size - data_.last;
}

void Gpu::RunKernel()
{
  for (int i = 0; i < particles_per_chunk_; i++)
  {
    for (int j = 0; j < steps_per_kernel_; j++)
    {
      // Collatz conjecture.
      if (data_.v[i].value & 1)
        data_.v[i].value = data_.v[i].value * 3 + 1;
      else
        data_.v[i].value = data_.v[i].value >> 1;

      // TODO(jeff): Check for completion.
    }
  }
}

int Gpu::SpaceUsed()
{
  return data_.last;
}
