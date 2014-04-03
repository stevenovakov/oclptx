// Copyright 2014
//  Afshin Haidari
//  Steve Novakov
//  Jeff Taylor
//
//  Global structs.  These are shared between oclptxhandler and the kernels, so
//  if they are changed here, they need to be changed in oclptxhandler as well.

#ifndef ATTRS_H_
#define ATTRS_H_

#include "rng.h"

// Struct representing the persistent state of a single particle.
struct particle_data
{
  rng_t rng; //RW
  float4 position;
  float4 dr;
} __attribute__((aligned(64)));

// Struct full of useful constants.
struct particle_attrs
{
  float4 brain_mask_dim;
  int steps_per_kernel;
  int max_steps;
  int particles_per_side;
  uint sample_nx;
  uint sample_ny;
  uint sample_nz;
  uint num_samples;
  float curvature_threshold;
  uint n_waypoint_masks;
  float step_length;
  uint lx;  // Loopcheck sizes
  uint ly;
  uint lz;
} __attribute__((aligned(16)));

#endif  // ATTRS_H_
