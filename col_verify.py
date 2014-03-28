#!/usr/bin/env python3
import subprocess
import random

random.seed(42)
num_to_test = 1000

values = set()
with open("./col_args",'w') as f:
    for i in range(num_to_test):
        r = random.randint(1,100000)
        f.write(str(r) + '\n')
        values.add(r)

# First make call with particle list
subprocess.call(["./oclptx"] + [str(v) for v in values])

with open("./path_output", 'r') as f:
    out = f.read()

def collatz(n):
    if n % 2:
        return (n * 3) + 1
    else:
        return n / 2

streams = {}
for line in out.split("\n"):
    if line == '':
        continue
    line = line.rstrip('n')
    s, result = [int(x) for x in line.split(":")]
    # New stream
    if s not in streams or 2 == streams[s]:
        assert result in values, \
            "Invalid particle %i started on thread %i" % (result, s)
        streams[s] = result
        values.remove(result)
    # Continued particle.  Either continued collatz or new particle
    else:
        assert result == collatz(streams[s]), "Incorrect collatz"
        streams[s] = result # Collatz

assert 0 == len(values), "Some values left: %s" % repr(values)

print("Success!")

