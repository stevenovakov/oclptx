// Copyright 2014 Jeff Taylor
//
// Various thread functions

#ifndef THREADING_H_
#define THREADING_H_

#include "particle.h"
#include "fifo.h"

namespace threading
{

void RunThreads(
    struct particle::particles *gpu,
    OclPtxHandler *handler,
    Fifo<struct particle::particle_data> *particles,
    int num_reducers);

}  // namespace threading

#endif  // THREADING_H_
