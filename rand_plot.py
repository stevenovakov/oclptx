import struct
import pylab

data = []

s = 0
with open("./rng_output",'rb') as f:
    d = f.read(8)
    while d:
        s += 1
        if 0 == s % 100000: print(s)
        data.append(struct.unpack('Q',d))
        d = f.read(8)

pylab.hist(d)
