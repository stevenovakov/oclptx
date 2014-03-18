// Copyright 2014 Jeff Taylor
//
// Various thread functions

#ifndef THREADING_H_
#define THREADING_H_

#include "particle/particle.h"
#include "threading/fifo.h"

namespace threading
{

void RunThreads(
    struct particles *gpu,
    Fifo<struct particle_data> *input_particles,
    int num_reducers);

}  // namespace threading

#endif  // THREADING_H_
