

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
  
  readfile = open(filePath, "rb").read()

  data = json.loads(readfile)
  
  total = len(data)

  print total

  len_thresh = 2

  plotset = []

  xtemp = []
  ytemp = []
  ztemp = []

  for i in xrange(0, total):
    for j in xrange(0, len(data[i])):
      xtemp.append(data[i][j][0])
      ytemp.append(data[i][j][1])
      ztemp.append(data[i][j][2])

    if (len(xtemp) > len_thresh):
      plotset.append([xtemp, ytemp, ztemp])
    xtemp = []
    ytemp = []
    ztemp = []
  
  fig = plt.figure()
  #ax = fig.gca(projection='3d')
  ax = Axes3D(fig)
  
  for item in plotset:
    ax.plot(item[0], item[1], item[2])
  
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
  
  if ".dat" not in fname:
    continue
  
  exitvar = 0
      
  for fn in fileList:
    if (fname.replace(".dat", "") in fn) and (".png" in fn):
      exitvar = 1
      break
      
  if exitvar == 1:
    continue
      
  if "_PATHS" in fname:
    print "Would you like to plot: " + str(fname) + " (y/n) ";
    choice = raw_input()
    
    if choice.lower() == "y":
      PlotPath(fname)
    