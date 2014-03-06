// Copyright 2014 Jeff Taylor
//
// Various thread functions

#ifndef THREADING_H_
#define THREADING_H_

namespace threading
{
void Worker(struct shared_data *p, Gpu *gpu, char *kick);
void Reducer(struct shared_data *p, Fifo<threading::collatz_data> *particles);
}  // namespace threading

#endif  // THREADING_H_
