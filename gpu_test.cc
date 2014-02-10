// Copyright 2014 Jeff Taylor
//
// Test our GPU class.

#include <cassert>
#include "oclptx/gpu.h"
#include "oclptx/structs.h"

int main(int argc, char **argv)
{
  // Create a GPU
  Gpu gpu(1, 3);

  assert(1 == gpu.SpaceRemaining());
  assert(0 == gpu.SpaceUsed());

  // Create two particles, check that Gpu only accepts one.
  struct threading::collatz_data data[2] = {{17, 0}, {32, 0}};
  struct threading::collatz_data_chunk chunk = {data, 2, 2};

  gpu.WriteChunk(&chunk);

  assert(1 == chunk.last);
  assert(1 == gpu.SpaceUsed());
  assert(0 == gpu.SpaceRemaining());

  // Run the kernel.
  gpu.RunKernel();

  // Now read the data into a new chunk.
  struct threading::collatz_data data2[2] = {{0, 0}, {0, 0}};
  struct threading::collatz_data_chunk chunk2 = {data2, 0, 2};

  gpu.ReadChunk(&chunk2);

  assert(1 == chunk2.last);
  assert(0 == gpu.SpaceUsed());
  assert(1 == gpu.SpaceRemaining());

  // And check that the result is correct.
  // 32 -> 16 -> 8  -> 4
  // 17 -> 52 -> 26 -> 13
  assert(4 == data2[0].value);

  return 0;
}
