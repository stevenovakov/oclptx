from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import os
import json

currentDirectory = os.getcwd()

fileList = os.listdir(currentDirectory)
fileList.sort()
fileList.reverse()


def PlotPath(fname):

  filePath = currentDirectory + "/" + fname
  picPath = filePath.replace(".dat", ".png")

  print "PARSING: " + filePath + "\n"

  readfile = open(filePath, "rb")

  data = readfile.readlines()

  plotset = {}
  type_done = {1 : 0, # 1:  max steps break
               2 : 0, # 2: curv thresh break
               3 : 0, # 3: loopcheck break
               4 : 0, # 4: out of bounds break
               5 : 0} # 5: termination break

  for row in data:
    split1 = row.replace("\n","").split(":")

    thread = split1[0]
    coords = split1[1]

    if thread not in plotset.keys():
      plotset[thread] = [0,{}]

    if "n" in coords:
      plotset[thread][0] += 1
      plotset[thread][1][plotset[thread][0]] = ([],[],[])

    coords = [float(item) for item in split1[1].rstrip('n').split(",")]
    plotset[thread][1][plotset[thread][0]][0].append(coords[0])
    plotset[thread][1][plotset[thread][0]][1].append(coords[1])
    plotset[thread][1][plotset[thread][0]][2].append(coords[2])


  fig = plt.figure()
  #ax = fig.gca(projection='3d')
  ax = Axes3D(fig)

  for key in plotset.keys():
    for key2 in plotset[key][1].keys():
      ax.plot(plotset[key][1][key2][0],
        plotset[key][1][key2][1], plotset[key][1][key2][2])

  ax.set_xlim3d([0, 102])
  ax.set_ylim3d([0, 102])
  ax.set_zlim3d([0, 60])

  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")

  plt.show()
  #fig.canvas.print_figure(picPath)

#
# Main
#
#

for fname in fileList:

  if fname == "path_output":
    PlotPath(fname)

