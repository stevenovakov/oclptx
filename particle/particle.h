// Generic Particle.  This header defines some functions useful in a variety
// of similar-but-different context.  Namely:
//  - RNG testing
//  - collatz testing (for race conditions)
//  - normal probtrackx
//  - etc?

#ifndef PARTICLE_H_
#define PARTICLE_H_

namespace particle
{

// Struct containing *all* relevant particle data, defined elsewhere.  Data
// should be added here if it requires special treatment, such as completion
// data, init-time defined arrays, etc.  Also includes information these
// structs require, such as array lengths.
struct particles;

// Struct containing fixed length particle data.  This struct is replicated
// exactly on the GPU, so adding elements to it will involve adding them in 
// two places:
//  1. the respective *_particle.h file.
//  2. the respective kernel.
struct particle_data;

// Particle initialization data.  This describes things such as "steps per
// kernel"
struct particle_attrs;

// Allocate new particles, both on GPU and on Host.
struct particles *NewParticles(cl::OclEnv* env, struct particle_attrs *attrs);
// Free Host side particles.
void FreeParticles(struct particles *particles);

// Write a single particle
void WriteParticle(struct particles *p, struct particle_data *data, int offset);
// Read the "completion" buffer back into the vector pointed to by ret.
void ReadStatus(struct particles *p, int offset, int count, cl_bool *ret);

void DumpPath(struct particles *p, int offset, int count, FILE *fd);

}  // namespace particle

#endif  // PARTICLE_H_
