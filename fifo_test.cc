// Copyright 2014 Jeff Taylor
// Test case for fifo

#include "fifo.h"


#include<cassert>
#include<cstdio>
#include<cstdint>

int main()
{
  // First things first, does usage example work?

  Fifo myfifo(3);  // create a FIFO with room for 7 entries.
  int *in = new int;  // create an integer
  myfifo.PushOrDie((void*) in);  // Put it onto the fifo (not mine anymore)
  int *out;
  out = (int*)myfifo.Pop();

  assert(out == in);
  printf("out: %p\n", out);
  printf("in:  %p\n", in);

  delete out;

  // Now lets put 7 values in (fill it), and take them out, check that order is right.
  // Specifically, let's make sure we wraparound at least once.
  for (intptr_t i = 0; i < 7; i++)
    myfifo.PushOrDie((void*) i);

  puts("Full\n");

  for (intptr_t i = 0; i < 7; i++)
    assert(i == (intptr_t)myfifo.Pop());

  puts("Empty\n");

  // Check that it signals "empty" state properly.
  assert(NULL == myfifo.Pop());

  puts("Really empty\n");

  // OK, now lets make it overflow
  for (intptr_t i = 0; i < 8; i++)
    myfifo.PushOrDie((void*) i);

  return 0;
}
