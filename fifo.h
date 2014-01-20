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

#include <mutex>

class Fifo
{
 public:
  void PushOrDie(void *val);
  void *Pop();
  Fifo(int order);
  ~Fifo();
 private:
  int order_;
  int head_; // front of queue---points to open slot
  int tail_; // back of queue---points to full slot
  void **fifo_;

  std::mutex head_mutex_;
  std::mutex tail_mutex_;
};

