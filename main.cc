// Copyright 2014 Jeff Taylor

#include <unistd.h>
#include <cassert>
#include <thread>

#include "fifo.h"
#include "oclenv.h"
#include "oclptxhandler.h"
#include "threading.h"

// log base 2
int lb(int x)
{
  int r = 0;
  x>>=1;
  while (x)
  {
    r++;
    x>>=1;
  }
  return r;
}

// 54 72 36 12 17 42 53 16 873 14 423
int main(int argc, char **argv)
{
  const int kStepsPerKernel = 10;
  const int kNumReducers = 2;
  FILE *global_fd;

  if (argc < 2)
  {
    fprintf(stderr, "Usage: %s <collatz numbers>.\n", argv[0]);
    return -1;
  }

  // Add the particles to our list
  Fifo<struct OclPtxHandler::particle_data> particles_fifo(lb(argc-1)+1);
  struct OclPtxHandler::particle_data *data;
  for (int i = 1; i < argc; ++i)
  {
    data = new OclPtxHandler::particle_data;
    *data = {(uint64_t) atoi(argv[i])};
    particles_fifo.PushOrDie(data);
  }

  // Create our oclenv
  OclEnv env("collatz");
  env.Init();

  global_fd = fopen("./path_output", "w");
  if (NULL == global_fd)
  {
    perror("Couldn't open file");
    exit(1);
  }

  struct OclPtxHandler::particle_attrs attrs = {kStepsPerKernel, 0};
  int num_dev = env.HowManyDevices();

  // Create a new oclptxhandler.
  OclPtxHandler *handler = new OclPtxHandler[num_dev];
  std::thread *gpu_managers[num_dev];

  for (int i = 0; i < num_dev; ++i)
  {
    handler[i].Init(env.GetContext(),
                    env.GetCq(i),
                    env.GetKernel(i),
                    NULL, NULL, NULL, 0, NULL,  // 5 bedpost data values.
                    &attrs,
                    global_fd);

    gpu_managers[i] = new std::thread(
        threading::RunThreads,
        &handler[i],
        &particles_fifo,
        kNumReducers);
  }

  for (int i = 0; i < num_dev; ++i)
  {
    gpu_managers[i]->join();
  }
  delete[] handler;

  fclose(global_fd);

  return 0;
}
