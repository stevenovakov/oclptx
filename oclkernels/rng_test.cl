/*  Copyright 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

#include "rng.cl"

__kernel void RngTest(
  __global rng_t *rng,        /* RW */
  __global ulong *rng_output, /* WO */
  int            start,       /* RO */
  int            finish,      /* RO */
  int            buf_size     /* RO */
)
{
  int i;
  int glid = get_global_id(0);

  for (i = start; i < finish; ++i)
  {
    int index = glid*buf_size + i;
    rng_output[index] = Rand(rng + glid);
  }
}

