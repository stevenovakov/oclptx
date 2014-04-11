// This code is a mess.  Please don't judge
// Create 100 RNGs, run them on the GPU, pull back their data, and verify it.

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include <CL/cl.hpp>
#include <CL/cl_platform.h>

#include "oclenv.h"

typedef cl_ulong8 rng_t;

uint64_t rand_64()
{
  // Assumption: rand() gives at least 16 bits.  It gives 31 on my system.
  assert(RAND_MAX >= (1<<16));

  uint64_t a,b,c,d;
  a = rand() & ((1<<16)-1);
  b = rand() & ((1<<16)-1);
  c = rand() & ((1<<16)-1);
  d = rand() & ((1<<16)-1);

  uint64_t r = ((a << (16*3)) | (b << (16*2)) | (c << (16)) | d);

  return r;
}

int init_rng(rng_t* rng, int seed, int count)
{
  if (RAND_MAX < (1<<16)) {
    puts("RAND_MAX is too small");
    return -1;
  }

  srand(seed);

  uint64_t init;
  for (int i = 0; i < count; ++i)
    for (int j = 0; j < 5; j++)
    {
      init = rand_64();
      rng[i].s[j] = init;
      fprintf(stderr, "Rng %i: z%i = %lx\n",  i, j, init);
    }

  return 0;
}

int main(int argc, char **argv)
{
  if (argc < 4) {
    printf("Usage: %s <random seed> <num_rngs> <num_steps>\n", argv[0]);
    return -1;
  }

  int seed = atoi(argv[1]);
  int num_rngs = atoi(argv[2]);
  int num_steps = atoi(argv[3]);

  // Now create some rng's.  Let's create 2
  rng_t *rng = new rng_t[num_rngs];
  if (-1 == init_rng(rng, seed, num_rngs))
    return -1;

  int rng_size = num_rngs * sizeof(cl_ulong8);
  int rng_path_size = num_rngs * num_steps * sizeof(cl_ulong);

  // Create Ocl Env
  OclEnv env("rng_test");

  // Create cl buffers for init rng & rng path
  cl::Buffer rng_buf(*env.GetContext(),
                     CL_MEM_READ_WRITE,
                     rng_size);

  cl::Buffer rng_path_buf(*env.GetContext(),
                          CL_MEM_READ_ONLY,
                          rng_path_size);

  // Write initial seeds to GPU
  env.GetCq(0)->enqueueWriteBuffer(rng_buf, true, 0, rng_size, (void*) rng);

  // Setup arguments.
  env.GetKernel(0)->setArg(0, rng_buf);
  env.GetKernel(0)->setArg(1, rng_path_buf);
  env.GetKernel(0)->setArg(2, 0);  // start
  env.GetKernel(0)->setArg(3, num_steps);  // finish
  env.GetKernel(0)->setArg(4, num_steps);

  cl::NDRange global_range(num_rngs);
  cl::NDRange local_range(1);

  env.GetCq(0)->enqueueNDRangeKernel(
    *env.GetKernel(0),
    cl::NullRange,
    global_range,
    local_range,
    NULL,
    NULL);

  env.GetCq(0)->finish();

  // Read back the buffer
  cl_ulong *rng_path = new cl_ulong[num_rngs * num_steps];
  env.GetCq(0)->enqueueReadBuffer(rng_path_buf, true, 0, rng_path_size, (void*) rng_path);

  // Dump somewhere.
  int fd = creat("./rng_output", 0666);
  if (-1 == fd)
  {
    perror("Couldn't open file");
    exit(1);
  }

  write(fd, (void*) rng_path, rng_path_size);
  close(fd);

  delete[] rng;

  return 0;
}

