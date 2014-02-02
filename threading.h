// Copyright 2014 Jeff Taylor
//
// Various thread functions

#ifndef THREADING_H_
#define THREADING_H_

namespace threading
{
void Worker(struct global_fifos *fifos, Gpu *gpu, char *kick);
int Reducer(struct global_fifos *fifos, char *kick);
int Watchdog();
}  // namespace threading

#endif  // THREADING_H_
