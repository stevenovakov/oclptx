// Definition for collatz particle methods.

#include "collatz_particle.h"

#include <assert.h>

namespace particle
{

struct particles *NewParticles(
    OclEnv* env,
    struct particle_attrs *attrs)
{
  struct particles *p = new struct particles;
  if (!p)
    return NULL;

  p->attrs = *attrs;
  p->env = env;

  // TODO(jeff) compute num_particles
  p->attrs.particles_per_side = 42;

  p->gpu_data = new cl::Buffer(
      *env->GetContext(),
      CL_MEM_READ_WRITE,
      2 * p->attrs.particles_per_side * sizeof(struct particle_data));
  if (!p->gpu_data)
    return NULL;

  p->gpu_complete = new cl::Buffer(
      *env->GetContext(),
      CL_MEM_WRITE_ONLY,
      2 * p->attrs.particles_per_side * sizeof(cl_ushort));
  if (!p->gpu_complete)
    return NULL;

  p->gpu_path = new cl::Buffer(
      *env->GetContext(),
      CL_MEM_WRITE_ONLY,
      2 * p->attrs.particles_per_side * p->attrs.num_steps * sizeof(cl_ulong));
  if (!p->gpu_path)
    return NULL;

  // Initialize "completion" buffer.
  cl_ushort *temp_completion = new cl_ushort[2*p->attrs.particles_per_side];
  for (int i = 0; i < 2 * p->attrs.particles_per_side; ++i)
    temp_completion[i] = 1;

  p->env->GetCq(0)->enqueueWriteBuffer(
      *p->gpu_complete, 
      true, 
      0,
      2 * p->attrs.particles_per_side * sizeof(cl_ushort),
      reinterpret_cast<void*>(temp_completion));

  delete[] temp_completion;

  return p;
}

void FreeParticles(struct particles *p)
{
  delete p->gpu_path;
  delete p->gpu_complete;
  delete p->gpu_data;
  delete p;
}

void WriteParticle(
    struct particles *p,
    struct particle_data *data,
    int offset)
{
  // Note: locking.  This function is technically thread-unsafe, but that
  // shouldn't matter because threading is set up for only one thread to ever
  // call these methods.
  cl_int ret;
  cl_ushort zero = 0;
  assert(offset < 2 * p->attrs.particles_per_side);

  printf("Write particle %li to offset %i\n", data->value, offset);

  // Write particle_data
  ret = p->env->GetCq(0)->enqueueWriteBuffer(
      *p->gpu_data, 
      true, 
      offset * sizeof(struct particle_data),
      sizeof(struct particle_data),
      reinterpret_cast<void*>(data));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    abort();
  }

  // gpu_complete = 0
  ret = p->env->GetCq(0)->enqueueWriteBuffer(
      *p->gpu_complete,
      true,
      offset * sizeof(cl_ushort),
      sizeof(cl_ushort),
      reinterpret_cast<void*>(&zero));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    abort();
  }
}

void ReadStatus(struct particles *p, int offset, int count, cl_ushort *ret)
{
  p->env->GetCq(0)->enqueueReadBuffer(
      *p->gpu_complete,
      true,
      offset * sizeof(cl_ushort),
      count * sizeof(cl_ushort),
      reinterpret_cast<cl_ushort*>(ret));
}

void DumpPath(struct particles *p, int offset, int count, FILE *fd)
{
  cl_ulong *buf = new cl_ulong[count * p->attrs.num_steps];
  int ret;
  int value;

  ret = p->env->GetCq(0)->enqueueReadBuffer(
      *p->gpu_path,
      true,
      offset * p->attrs.num_steps * sizeof(cl_ulong),
      count * p->attrs.num_steps * sizeof(cl_ulong),
      reinterpret_cast<void*>(buf));
  if (CL_SUCCESS != ret)
  {
    puts("Failed to read back path");
    abort();
  }

  // Now dumpify.
  for (int id = 0; id < count; ++id)
  {
    for (int step = 0; step < p->attrs.num_steps; ++step)
    {
      value = buf[id * p->attrs.num_steps + step];
      fprintf(fd, "%i:%i\n", id + offset, value);
    }
  }

  delete buf;
}

int particles_per_side(struct particles *p)
{
  return p->attrs.particles_per_side;
}

}  // namespace particle
