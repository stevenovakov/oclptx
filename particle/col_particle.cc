// Definition for collatz particle methods.

#include "particle/col_particle.h"

namespace particle
{

struct particles *NewParticles(
    cl::OclEnv* env,
    struct particle_attrs *attrs)
{
  struct particles *p = new struct particles;
  if (!p)
    return NULL;

  p->attrs = attrs;
  p->env = env;

  // TODO(jeff) compute num_particles
  p->num_particles = 42;

  p->gpu_data = new cl::Buffer(
      env->GetContext(),
      CL_MEM_WRITE_ONLY,
      p->num_particles * sizeof(struct particle_data));
  if (!p->gpu_data)
    return NULL;

  p->gpu_complete = new cl::Buffer(
      env->GetContext(),
      CL_MEM_READ_WRITE,
      p->num_particles * sizeof(cl_bool));
  if (!p->gpu_complete)
    return NULL;

  p->gpu_path = new cl::Buffer(
      env->GetContext(),
      CL_MEM_READ_ONLY,
      p->num_particles * p->attrs.num_steps * sizeof(cl_ulong));
  if (!p->gpu_path)
    return NULL;

  return p;
}

void FreeParticles(struct particles *p)
{
  delete p->gpu_path;
  delete p->gpu_complete;
  delete p->gpu_data;
  delete p;
}

int WriteParticle(
    struct particles *p,
    struct particle_data *data,
    int offset)
{
  // Note: locking.  This function is technically thread-unsafe, but that
  // shouldn't matter because threading is set up for only one thread to ever
  // call these methods.
  cl_int ret;
  cl_bool zero = 0;
  assert(offset < num_particles);

  // Write particle_data
  // TODO(jeff): think about multi-gpu
  ret = p->env->getCq(0)->enqueueWriteBuffer(
      p->gpu_data, 
      true, 
      offset * sizeof(struct particle_data),
      sizeof(struct particle_data),
      reinterpret_cast<void*>data);
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    abort();
  }

  // gpu_complete = 0
  ret = p->env->getCq(0)->enqueueWriteBuffer(
      p->gpu_complete,
      true,
      offset * sizeof(cl_bool),
      sizeof(cl_bool),
      reinterpret_cast<void*>(&zero));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    abort();
  }

  return 0;
}

void ReadStatus(struct particles *p, int offset, int count, cl_bool *ret)
{
  p->env->getCq(0)->enqueueReadBuffer(
      p->gpu_complete,
      true,
      offset * sizeof(cl_bool),
      count * sizeof(cl_bool),
      reinterpret_cast<cl_bool*>(ret));
}

void DumpPath(struct particles *p, int offset, int count, FILE *fd)
{
  cl_ulong *buf = new cl_ulong[count * num_steps];
  int ret;

  ret = p->env->getCq(0)->enqueueReadBuffer(
      p->gpu_path,
      true,
      offset * num_steps * sizeof(cl_ulong),
      count * num_steps * sizeof(cl_ulong),
      reinterpret_cast<cl_ulong*>(buf));
  if (CL_SUCCESS != ret)
  {
    puts("Failed to read back path");
    abort();
  }

  // Now dumpify.
  for (int i = 0; i < count, ++i)
  {
    for (int step = 0; step < num_steps; ++step)
    {
      value = buf[i * num_steps + step];
      fprintf(fd, "%i:%i", i + offset, value);
    }
  }

  delete buf;
}

}  // namespace particle
