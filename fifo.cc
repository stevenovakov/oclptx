// Function definitions for fifo class

#include "fifo.h"

#include<cstdlib>
#include<mutex>

Fifo::Fifo(int order):
  order_(order),
  head_(0),
  tail_(0)
{
  fifo_ = new void*[1<<order];
}

Fifo::~Fifo()
{
  delete[] fifo_;
}

void Fifo::PushOrDie(void *val)
{
  std::lock_guard<std::mutex> lock(head_mutex_);

  if (1 == (tail_ - head_ & ((1<<order_)-1)))
    // FIFO Overflow
    abort();

  fifo_[head_] = val;
  head_ = (head_ + 1) & ((1<<order_)-1);

  // lock_guard releases head_mutex automatically
  return;
}

void *Fifo::Pop()
{
  std::lock_guard<std::mutex> lock(tail_mutex_);
  
  if (head_ == tail_) // FIFO Empty
    return NULL;

  void *retval = fifo_[tail_];
  
  tail_ = (tail_ + 1) & ((1<<order_)-1);

  // lock_guard releases tail_mutex automatically
  return retval;
}
