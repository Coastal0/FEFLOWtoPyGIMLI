# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:02:53 2018

@author: 264401k
"""

import pybert as pb
import pandas as pd
import pygimli as pg
import numpy as np
import workbook as w
import tkinter as tk
from tkinter import filedialog


print("pyGIMLI:", pg.__version__)
print("Pandas:", pd.__version__)
print("Numpy:", np.__version__)

tk.Tk().withdraw()
fNames = tk.filedialog.askopenfilenames()
print(fNames)

ert = pb.ERTManager(debug=True)

#fNames = [r"K:\results\QR_Quad_200md_10kCells_3MLpa_Mass.dat"]

dataDict, data, coords = w.loadData(fNames)
hull, bPoly = w.loadHull(r"K:\boundsXY_10kCells.mat")
nodes = w.loadNodes(r"K:\nodesXY.mat")
bPolyMesh = w.fillPoly(bPoly, coords, hull)  # Fill boundary mesh with nodes
w.checkMesh(bPolyMesh)  # Check each node has a value.
topoArray = w.getTopo(hull)  # Extract the topography from the concave hull
dMesh = w.makeDataMesh(coords, 0)  # Make a mesh using datapoints
ertScheme, meshERT = w.createArray(0, 600, 5, 'wa', topoArray, enlarge = 1)

dInterpDict = {}
for d in dataDict.keys():
    print(d)
    data = dataDict[d][0]
    if 'MINIT' not in data:
        print('Mass not present... breaking.')
        break
    else:
        data['dataCol'] = data['MINIT']
    dInterpDict[d], times = w.makeInterpVector(data, dMesh, bPolyMesh)  # Add data to the nodes

resBulk = w.convertFluid(dInterpDict[0], bPolyMesh, meshERT)

#ax, cb = pg.show(meshERT, resBulk, showMesh = True, colorBar = True, cMap = 'jet_r')
#ax.set_ylim(-40,40)
#ax.set_xlim(-20,300)

print('Forward Modelling...')
# ERI Stuff
#invalids = 0
#for i,m in enumerate(ertScheme("m")):
#    if m < int(ertScheme("b")[i]):
#        invalids = invalids + 1
#        ertScheme.markInvalid(int(i))
#for i,n in enumerate(ertScheme("n")):
#    if n < int(ertScheme("m")[i]):
#        invalids = invalids + 1
#        ertScheme.markInvalid(int(i))
#print(invalids)
ertScheme.save('testests', "a b m n valid")

# Set geometric factors to one, so that rhoa = r
#    ertScheme.set('k', pb.geometricFactor(ertScheme))
ertScheme.set('k', np.ones(ertScheme.size()))
simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme, verbose = True, noiseAbs=0.0,  noiseLevel = 0.01)
simdata.set("r", simdata("rhoa"))

# Calculate geometric factors for flat earth
flat_earth_K = pb.geometricFactors(ertScheme)
simdata.set("k", flat_earth_K)

# Set output name
dataName = fNames[0][:-4]+'_data.ohm'
simdata.save(dataName, "a b m n r err k")
print('Done.')