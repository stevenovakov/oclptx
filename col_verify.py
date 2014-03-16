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
out = subprocess.check_output(["./oclptx"] + [str(v) for v in values])

def collatz(n):
    if n % 2:
        return (n * 3) + 1
    else:
        return n / 2

streams = {}
for line in out.decode().split("\n"):
    if line == '':
        continue
    s, result = [int(x) for x in line.split(":")]
    # New particle
    if s not in streams or 2 == streams[s]:
        assert result in values, "New particle started with invalid value"
        streams[s] = result
        values.remove(result)
    # Continued particle
    else:
        assert result == collatz(streams[s]), "Incorrect collatz"
        streams[s] = result

assert 0 == len(values), "Some values left: %s" % repr(values)

print("Success!")

