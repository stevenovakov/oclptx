import os
import random

currentDirectory = os.getcwd()

fileList = os.listdir(currentDirectory)

filePath = currentDirectory + "/" + "fdt_coordinates.txt"

print "Writing to \n" +  filePath

f = open(filePath, "wb")

n = 100000

print "Center"
x = 160
y = 129
z = 111
x_range = 5
y_range = 5
z_range = 5

for i in xrange(0, n):
  newx = x + x_range*(random.random() - 0.5)
  newy = y + y_range*(random.random() - 0.5)
  newz = z + z_range*(random.random() - 0.5)
  f.write(" ".join([str(newx), str(newy), str(newz)]) + "\n")

# print "Cortex"
# x = 51
# y = 51
# z = 35
# x_range = 60
# y_range = 60
# z_range = 30

# for i in xrange(0, n/2):
#   newx = x + x_range*(random.random() - 0.5)
#   newy = y + y_range*(random.random() - 0.5)
#   newz = z + z_range*(random.random() - 0.5)
#   f.write(" ".join([str(newx), str(newy), str(newz)]) + "\n")

# print "Stem"
# x = 51
# y = 51
# z = 25
# x_range = 20
# y_range = 20
# z_range = 10

# for i in xrange(0, n/2):
#   newx = x + x_range*(random.random() - 0.5)
#   newy = y + y_range*(random.random() - 0.5)
#   newz = z + z_range*(random.random() - 0.5)
#   f.write(" ".join([str(newx), str(newy), str(newz)]) + "\n")

f.close()
