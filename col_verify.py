#!/usr/bin/env python3
import subprocess
import random

#values = {54, 72, 36, 12, 17, 42, 53, 16, 873, 14, 423}
values = set()
for i in range(100000):
    values.add(random.randint(1,1000))

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
        assert result in values, "New particle (new stream) started with invalid value"
        streams[s] = result
        values.remove(result)
    # Continued particle
    else:
        assert result == collatz(streams[s]), "Incorrect collatz"
        streams[s] = result

print("Success!")

