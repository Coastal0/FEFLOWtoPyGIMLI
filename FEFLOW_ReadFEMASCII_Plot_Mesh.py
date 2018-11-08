# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 09:59:17 2018

@author: 264401k
"""

import numpy as np
from tqdm import tqdm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


#fIN = r"K:\results\QR_Quad_10kCells_ASCIIFEM.fem"
#fIN = r"K:\results\QR_Quad_120kCells_ASCIIFEM.fem"

tk.Tk().withdraw()
fIN = tk.filedialog.askopenfilename()
print(fIN)

with open(fIN) as file:
    array = file.readlines()


ProblemTitle = array[0]
ClassHeader = array[1]
Class = [int(x) for x in array[2].split()]

DimensHeader = array[3]
tLine = [int(x) for x in array[4].split()]
nNodes = tLine[0]
nEle = tLine[1]
nDims = tLine[2]  # e.g. 3 for triangles, 4 for quads.

# Elements
elements = []
for i in tqdm(array[8:nEle+7]):
    elements.append([int(x) for x in i.split()])

# Nodes
indices = [i for i, s in enumerate(array) if 'COOR' in s]
nodesXY_1D = []
for i in tqdm(array[indices[0]+1:indices[1]]):
    nodesXY_1D.extend([float(x) for x in list(filter(None, i.strip('\n').split(',')))])
nodeX = nodesXY_1D[0:nNodes]
nodeY = nodesXY_1D[nNodes:]

# Combined
patches = []
for i, n in tqdm(enumerate(elements)):
    p1 = nodeX[n[0]-1], nodeY[n[0]-1]
    p2 = nodeX[n[1]-1], nodeY[n[1]-1]
    p3 = nodeX[n[2]-1], nodeY[n[2]-1]
    p4 = nodeX[n[3]-1], nodeY[n[3]-1]
    pts = np.stack([p1, p2, p3, p4])

    polygon = Polygon(pts, True)
    patches.append(polygon)

p = PatchCollection(patches, alpha=0.4)
p.set_edgecolor('k')
plt.rc('font', family='Arial', size = 12)
fig, ax = plt.subplots()
ax.add_collection(p)
ax.set_aspect(1)
ax.autoscale(True)
plt.xlim(-50,50)
plt.ylim(-30,15)
plt.ylabel('Elevation (mAHD)')
plt.xlabel('Distance (m)')
fig.set_size_inches(10, 5)
fig.set_tight_layout('tight')
plt.show()
