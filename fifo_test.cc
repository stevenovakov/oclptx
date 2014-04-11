// Copyright 2014 Jeff Taylor
// Test case for fifo

#include "fifo.h"

#include<cassert>
#include<cstdio>
#include<cstdint>

int main()
{
  // First things first, does usage example work?

  Fifo<int> myfifo(7);  // create a FIFO with room for 7 entries.
  int *in = new int;  // create an integer
  myfifo.PushOrDie(in);
  int *out;
  out = myfifo.Pop();

  assert(out == in);
  printf("out: %p\n", static_cast<void*>(out));
  printf("in:  %p\n", static_cast<void*>(in));

  delete out;

  // Now lets put 7 values in (fill it), and take them out, check that order
  // is right.  Specifically, let's make sure we wraparound at least once.
  for (intptr_t i = 0; i < 7; i++)
    myfifo.PushOrDie(reinterpret_cast<int*>(i));

  puts("Full");

  for (intptr_t i = 0; i < 7; i++)
    assert(i == reinterpret_cast<intptr_t>(myfifo.Pop()));

  puts("Empty");

  // Check that it signals "empty" state properly.
  assert(NULL == myfifo.Pop());

  puts("Really empty");

  // OK, now lets make it overflow
  for (intptr_t i = 0; i < 8; i++)
    myfifo.PushOrDie(reinterpret_cast<int*>(i));

  return 0;
}
