/* Copyright 2014
 *  Afshin Haidari
 *  Jeff Taylor
 *  Steve Novakov
 */

typedef ulong8 rng_t;

//
// 5-state 64 bit LFSR.  Parameters used are from the paper
// P. L'Ecuyer, "Tables of Maximally Equidistributed Combined LFSR Generators",
//  Mathematics of Computation, 68, 225 (1999), 261--269
// http://www.ams.org/journals/mcom/1999-68-225/S0025-5718-99-01039-X/S0025-5718-99-01039-X.pdf
//
// ulong Rand(__global rng_t *z)
// {
//   ulong b;
//   b = ((((*z).s0 << 1) ^ (*z).s0) >> 53);
//   (*z).s0 = ((((*z).s0 & 18446744073709551614UL) << 10) ^ b);

//   b = ((((*z).s1 << 24) ^ (*z).s1) >> 50);
//   (*z).s1 = ((((*z).s1 & 18446744073709551104UL) << 5) ^ b);

//   b = ((((*z).s2 << 3) ^ (*z).s2) >> 23);
//   (*z).s2 = ((((*z).s2 & 18446744073709547520UL) << 29) ^ b);

//   b = ((((*z).s3 << 5) ^ (*z).s3) >> 24);
//   (*z).s3 = ((((*z).s3 & 18446744073709420544UL) << 23) ^ b);

//   b = ((((*z).s4 << 3) ^ (*z).s4) >> 33);
//   (*z).s4 = ((((*z).s4 & 18446744073701163008UL) << 8) ^ b);

//   // ret will be uniformly distributed 64-bit number.
//   ulong ret = ((*z).s0 ^ (*z).s1 ^ (*z).s2 ^ (*z).s3 ^ (*z).s4);

//   return ret;
// }
ulong Rand(__global rng_t *z)
{
  ulong b;
  b = ((((*z).s0 << 1) ^ (*z).s0) >> 53);
  (*z).s0 = ((((*z).s0 & 18446744073709551614UL) << 10) ^ b);

  b = ((((*z).s1 << 24) ^ (*z).s1) >> 50);
  (*z).s1 = ((((*z).s1 & 18446744073709551104UL) << 5) ^ b);

  b = ((((*z).s2 << 3) ^ (*z).s2) >> 23);
  (*z).s2 = ((((*z).s2 & 18446744073709547520UL) << 29) ^ b);

  b = ((((*z).s3 << 5) ^ (*z).s3) >> 24);
  (*z).s3 = ((((*z).s3 & 18446744073709420544UL) << 23) ^ b);

  b = ((((*z).s4 << 3) ^ (*z).s4) >> 33);
  (*z).s4 = ((((*z).s4 & 18446744073701163008UL) << 8) ^ b);

  // ret will be uniformly distributed 64-bit number.
  ulong ret = ((*z).s0 ^ (*z).s1 ^ (*z).s2 ^ (*z).s3 ^ (*z).s4);

  return ret;
}

