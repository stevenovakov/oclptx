import struct
import pylab
import scipy

s = 0
with open("./rng_output",'rb') as f:
    read_data = scipy.fromfile(file=f, dtype=scipy.uint64)

pylab.hist(read_data)
pylab.show()
