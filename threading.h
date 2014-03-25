// Copyright 2014 Jeff Taylor
//
// Various thread functions

#ifndef THREADING_H_
#define THREADING_H_

#include "fifo.h"

namespace threading
{

void RunThreads(
    OclPtxHandler *handler,
    Fifo<struct OclPtxHandler::particle_data> *particles,
    int num_reducers);

}  // namespace threading

#endif  // THREADING_H_
