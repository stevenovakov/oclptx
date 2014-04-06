// Copyright 2014 Jeff Taylor
//
// A minimal atomic FIFO with thread safe operations.
// The exclusion of a check_empty and check_full function is intentional, as
// these encourage thread-unsafe behaviour.
//
// Order indicates how many entries, as a power of two, the fifo will include.
// eg order 2 => 2^2-1 = 3 entries.
//
// It is intended that the FIFOs will be sized for maximum possible number of
// entries, and are not resizable.
//
// If the FIFO is empty, pop will return NULL.  If it is full, push will DIE.
//
// There's a reason for the latter.  I could include a IsFifoFull() method,
// but this is likely to create races (check for full, and then push, but
// something may have happened in between).  Alternatively, we could return
// some error code.  This doesn't accomplish anything except to put the
// responsibility on the caller, who must either abort(), or block.
//
// In general, if you push an item onto the FIFO, you shouldn't touch its
// contents anymore, and when you pull an item off the FIFO, it belongs to you.
// It is your responsibility to either give the FIFO to someone else or delete
// its contents.
//
// Obviously, putting a pointer to local data on the FIFO is a really dumb
// idea.  Don't do it.
//
// Sample Usage:
//   Fifo myfifo(3); // create a FIFO with room for 7 entries.
//   int *myint = new int(42); // create an integer
//   myfifo.PushOrDie((void*) myint); // Put it onto the fifo (not mine anymore)
//
//   // some other thread
//   int *now_my_int;
//   now_my_int = (int*)myfifo.Pop();
//
//   // Do something with now_my_int...
//
//   delete now_my_int;
//
#ifndef FIFO_H_
#define FIFO_H_

#include <unistd.h>

#include <cstdlib>
#include <mutex>

template<typename T> class Fifo
{
 public:
  explicit Fifo(int order);
  ~Fifo();
  void PushOrDie(T *val);
  T *Pop();
  T *PopOrBlock();
  int count();
 private:
  int order_;
  int head_;  // front of queue---points to open slot
  int tail_;  // back of queue---points to full slot
  T **fifo_;

  std::mutex head_mutex_;
  std::mutex tail_mutex_;
};

template<typename T> Fifo<T>::Fifo(int count):
  head_(0),
  tail_(0)
{
  // log base 2
  order_ = 1;
  count>>=1;
  while (count)
  {
    order_++;
    count>>=1;
  }

  fifo_ = new T*[1 << order_];
}

template<typename T> Fifo<T>::~Fifo()
{
  delete[] fifo_;
}

template<typename T> void Fifo<T>::PushOrDie(T *val)
{
  std::lock_guard<std::mutex> lock(head_mutex_);

  if (1 == ((tail_ - head_) & ((1 << order_) - 1)))
  {
    puts("FIFO overflow");
    abort();
  }

  fifo_[head_] = val;
  head_ = (head_ + 1) & ((1 << order_) - 1);

  // lock_guard releases head_mutex automatically
  return;
}

template<typename T> T *Fifo<T>::Pop()
{
  std::lock_guard<std::mutex> lock(tail_mutex_);

  if (head_ == tail_)  // FIFO Empty
    return NULL;

  T *retval = fifo_[tail_];

  tail_ = (tail_ + 1) & ((1 << order_) - 1);

  // lock_guard releases tail_mutex automatically
  return retval;
}

template<typename T> T *Fifo<T>::PopOrBlock()
{
  T *ret;
  while (1)
  {
    ret = Pop();
    if (ret)
      return ret;
    usleep(1000);  // TODO(jeff) don't poll
  }
}

template<typename T> int Fifo<T>::count()
{
  return (head_ - tail_) & ((1 << order_) - 1);
}

#endif  // FIFO_H_
