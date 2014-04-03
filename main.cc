/* Copyright 2014
 *  Afshin Haidari
 *  Steve Novakov
 *  Jeff Taylor
 */

#include <unistd.h>
#include <cassert>
#include <thread>
#include <chrono>

#include "fifo.h"
#include "oclenv.h"
#include "oclptxhandler.h"
#include "samplemanager.h"
#include "threading.h"


cl_ulong8 rng_zero = {0,};

int main(int argc, char **argv)
{
  const int kStepsPerKernel = 1000;
  const int kNumReducers = 1;
  FILE *global_fd;
  Fifo<struct OclPtxHandler::particle_data> *particles_fifo;

  // Create our oclenv
  OclEnv env;
  env.OclInit();
  env.NewCLCommandQueues();

  // Startup the samplemanager
  SampleManager *sample_manager = &SampleManager::GetInstance();
  sample_manager->ParseCommandLine(argc, argv);
  particles_fifo = sample_manager->GetSeedParticles();

  env.AvailableGPUMem(
    sample_manager->GetFDataPtr(),
    sample_manager->GetOclptxOptions(),
    sample_manager->GetWayMasksToVector().size(),
    NULL,
    NULL
  );
  // todo(Steve) pass the masks
  env.AllocateSamples(
    sample_manager->GetFDataPtr(),
    sample_manager->GetPhiDataPtr(),
    sample_manager->GetThetaDataPtr(),
    sample_manager->GetBrainMaskToArray(),
    sample_manager->GetExclusionMaskToArray(),
    sample_manager->GetTerminationMaskToArray(),
    NULL // todo(steve) fix way mask implementation and pass it
  );

  env.CreateKernels("standard");

  auto t_end = std::chrono::high_resolution_clock::now();
  auto t_start = std::chrono::high_resolution_clock::now();

  global_fd = fopen("./path_output", "w");
  if (NULL == global_fd)
  {
    perror("Couldn't open file");
    exit(1);
  }

  struct OclPtxHandler::particle_attrs attrs = {
    sample_manager->brain_mask_dim(),
    kStepsPerKernel,
    sample_manager->GetOclptxOptions().nsteps.value(), // max_steps
    0, // Particles per side not determined here.
    env.GetEnvData()->nx,
    env.GetEnvData()->ny,
    env.GetEnvData()->nz,
    env.GetEnvData()->ns, // num_samples
    sample_manager->GetOclptxOptions().c_thr.value(), // curv threshold
    env.GetEnvData()->n_waypts,
    sample_manager->GetOclptxOptions().steplength.value(),
    env.GetEnvData()->lx,
    env.GetEnvData()->ly,
    env.GetEnvData()->lz
    }; // num waymasks.
  int num_dev = env.HowManyDevices();

  // Create a new oclptxhandler.
  OclPtxHandler *handler = new OclPtxHandler[num_dev];
  std::thread *gpu_managers[num_dev];

  // may want to change the location of this later
  t_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_dev; ++i)
  {
    handler[i].Init(env.GetContext(),
                    env.GetCq(i),
                    env.GetKernel(i),
                    env.GetSumKernel(i),
                    &attrs,
                    global_fd,
                    env.GetEnvData(),
                    env.GetDevicePdf(i));

    gpu_managers[i] = new std::thread(
        threading::RunThreads,
        &handler[i],
        particles_fifo,
        kNumReducers);
  }

  for (int i = 0; i < num_dev; ++i)
  {
    gpu_managers[i]->join();
  }

  t_end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float> delta_t = t_end - t_start;
  printf("Time to track [s]: %f\n", delta_t.count());

  env.PdfsToFile("pdf_out");

  delete[] handler;

  fclose(global_fd);

  return 0;
}
