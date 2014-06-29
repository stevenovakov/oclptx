// Copyright 2014 Jeff Taylor
//
// A minimal atomic FIFO with thread safe operations.
//
// It is intentional that the FIFOs will be sized for maximum possible number of
// entries, and are not resizable.
//
// If the FIFO is empty, `Pop()` will block until new data appears.  If it is
// full, `Push()` will block until there is space.
//
// This FIFO also includes a `Finish()` method.  This method signals that the
// FIFO cannot accept any new data, and can only be popped.  When `Finish()` has
// been called, and there is no data left, the popper will recieve a NULL
// pointer.  This is a sign to them that no new data will appear (and they can
// safely flush whatever data they have and then quit).
//
// Finish() is only meaningful if there is only one writer.  If there are two
// writers, we need some different mechanism to indicate that *all* of them have
// finished.
//
// Data should be allocated with `new` before being pushed onto the FIFO, and
// not referenced after being pushed. Likewise, data popped off the FIFO should
// be freed with `delete` (eventually---it is the popper's responsibility not to
// forget).
//
// Sample Usage::
//
//   Fifo myfifo<int>(13); // create a FIFO with room for >=13 entries.
//   int *myint = new int(42); // create an integer
//   myfifo.Push(myint); // Put it onto the fifo (not mine anymore)
//
//   // some other thread
//   int *now_my_int;
//   now_my_int = myfifo.Pop();
//
//   // Do something with now_my_int...
//
//   delete now_my_int;
//
// Notes:
//
//  * Tail points to the next item to be popped.
//  * Head points to the free space where the next item to be pushed will
//  appear.
//  * Size is a power of two, and all operations are modulo size.
//  * If Head == Tail, we have an empty FIFO (The next item to be popped is an
//  empty space.
//  * If Head == Tail + 1, we can't fit any more items in the FIFO.  We
//  technically have a free space, but adding one more item would cause head ==
//  tail, confusing us.
//  * Finish() is simply a NULL pointer in the FIFO.
//
#ifndef FIFO_H_
#define FIFO_H_

#include <unistd.h>

#include <cstdlib>
#include <condition_variable>
#include <mutex>

template<typename T> class Fifo
{
 public:
  explicit Fifo(int order);
  ~Fifo();
  void Push(T *val);
  void Finish();
  T *Pop();
  int64_t count();
 private:
  int order_;
  int head_;  // front of queue---points to open slot
  int tail_;  // back of queue---points to full slot
  T **fifo_;

  int64_t count_;  // How many particles have passed through?

  std::mutex head_mutex_;
  std::mutex tail_mutex_;
  std::condition_variable full_cv_;
  std::condition_variable empty_cv_;
};

template<typename T> Fifo<T>::Fifo(int count):
  head_(0),
  tail_(0),
  count_(0)
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

template<typename T> void Fifo<T>::Push(T *val)
{
  std::unique_lock<std::mutex> lock(head_mutex_);

  while (1 == ((tail_ - head_) & ((1 << order_) - 1))) // FIFO Full
    full_cv_.wait(lock);

  fifo_[head_] = val;
  head_ = (head_ + 1) & ((1 << order_) - 1);

  // lock_guard releases head_mutex automatically
  empty_cv_.notify_one();
  return;
}

template<typename T> void Fifo<T>::Finish()
{
  Push(NULL);
  empty_cv_.notify_all();
  return;
}

template<typename T> T *Fifo<T>::Pop()
{
  std::unique_lock<std::mutex> lock(tail_mutex_);

  while (head_ == tail_)  // FIFO Empty
    empty_cv_.wait(lock);

  T *retval = fifo_[tail_];

  if (!retval) // Finished.  Don't actually pop.
    return NULL;

  tail_ = (tail_ + 1) & ((1 << order_) - 1);

  __sync_fetch_and_add(&count_, 1);

  full_cv_.notify_one();
  return retval;
}

template<typename T> int64_t Fifo<T>::count()
{
  return count_;
}

#endif  // FIFO_H_
