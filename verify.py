#!/usr/bin/env python3
import subprocess
import random

with open("./oclptx_path_output", 'r') as f:
    oclptx_out = f.readlines()

with open("./ptx2_path_output", 'r') as f:
    ptx_out = f.readlines()

def eq(vec, ptx_str):
    delta = 1e-10
    ptx_str = ptx_str.lstrip('n')
    print(ptx_str)
    ptx = [float(x) for x in ptx_str.split(',')]
    assert len(vec) == len(ptx)
    for i in range(len(ptx)):
        if abs(vec[i]-ptx[i]) > delta:
            return False
    return True

values = set()
# First, create a set with the start lines of each of the ptx2 particles.
for i in range(len(ptx_out)):
    if ptx_out[i][0] == 'n':
        values.add(i)

streams = {}
for line in oclptx_out:
    if line == '':
        continue
    line = line.rstrip('n\n')
    line = line.split(':')
    s = line[0]
    pos = [float(x) for x in line[1].split(',')]

    # Figure out if this stream should have finished
    if s in streams and streams[s]+1 < len(ptx_out) and 'n' == ptx_out[streams[s]+1][0]:
        stream_finished = True
    else:
        stream_finished = False

    # New stream
    if s not in streams or stream_finished:
        # Check if any of the 'values' lines hold our position.
        for line in values:
            if eq(pos, ptx_out[line]):
                streams[s] = line
                values.remove(line)
                break
        else:
            assert 0, "Invalid new particle: %s" % line
    # Continued particle.  Check it matches the next line
    else:
        assert eq(pos, ptx_out[streams[s]+1]), \
            "Incorrect tracking. ocl: %s, ptx: %s " %  \
            (line, ptx_out[streams[s]+1])
        streams[s] += 1

assert 0 == len(values), "Some values left: %s" % repr(values)

print("Success!")

