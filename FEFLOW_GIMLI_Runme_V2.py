# -*- coding: utf-8 -*-
import workbook as w
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pygimli as pg
import numpy as np
import tkinter as tk
from tkinter import filedialog


print("pyGIMLI:", pg.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", mpl.__version__)
print("Numpy:", np.__version__)

tk.Tk().withdraw()
fNames = tk.filedialog.askopenfilenames()
print(fNames)

dataDict, data, coords = w.loadData(fNames)

# %% Mesh and coordinate geometry
hull, bPoly = w.loadHull(r"K:\boundsXY_10kCells.mat")
nodes = w.loadNodes(r"K:\nodesXY.mat")
bPolyMesh = w.fillPoly(bPoly, coords, hull)  # Fill boundary mesh with nodes
w.checkMesh(bPolyMesh)  # Check each node has a value.
topoArray = w.getTopo(hull)  # Extract the topography from the concave hull
dMesh = w.makeDataMesh(coords, 0)  # Make a mesh using datapoints

# %% Data geometry
dInterpDict = {}
for d in dataDict.keys():
    print(d)
    data = dataDict[d][0]
    if 'SINIT' in data:
        print('Mass and Saturation present')
        data['dataCol'] = data['MINIT'] * data['SINIT']
    elif 'MINIT' in data and 'SINIT' not in data:
        print('Mass present')
        data['dataCol'] = data['MINIT']
    elif 'MINIT' or 'SINIT' not in data:
        print('Neither mass or solute present in data')
        data['dataCol'] = data.iloc[:,-1]
    
    dInterpDict[d], times = w.makeInterpVector(data, dMesh, bPolyMesh)  # Add data to the nodes
 
#%% Plot Difference
# Difference Map (if two+ dInterps present)
#if np.shape(dInterp)[-1]
#    diffVector = (dInterp[:,1] - dInterp[:,0])/dInterp[:,0]
#    norm = mpl.colors.SymLogNorm(linthresh=0.01, linscale=1, vmin=-max(abs(diffVector)), vmax=max(abs(diffVector)))
#    fig, ax = plt.subplots()
#    im = pg.mplviewer.drawMPLTri(ax, bPolyMesh, data=diffVector, cMin=None, cMax=None, cmap='RdBu_r', logScale=True, norm=norm)
#    cbar = fig.colorbar(im, ax=ax, extend='neither', orientation = 'horizontal', aspect = 100, format = mpl.ticker.LogFormatter(linthresh = norm.linthresh))
#    
#    def logformat(cbar):
#        new_labels = []
#        for label in cbar.ax.xaxis.get_ticklabels():
#            if len(label.get_text())>0:
#                print(label.get_text())
#                new_labels.append("%g" % float(label.get_text()))
#            else:
#                new_labels.append("")
#    
#        cbar.ax.xaxis.set_ticklabels(new_labels)
#    
#    logformat(cbar)
#    
#    formatActiveFig()
#    plotWells()
#else:
#    print('NO.')


# %% Show Model
# Custom Cmap
C = np.loadtxt(r"F:/testCmap3.txt")
ccmap = mpl.colors.ListedColormap(C/255)
    
#cmap = plt.cm.get_cmap('jet')
#cmap = jet_light
#cmap = jet_alpha
cmap.set_under('lightgray',0.5)
    
cmap = mpl.colors.ListedColormap(['wheat', 'lavender'])
bounds = [0.1,0.2,0.3]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Load and Show data
for j in dInterpDict.keys():
    print(j)
    mesh = bPolyMesh
    dataVec = dInterpDict[j]
    
    setAboveWT = 1
    setValue = 0.3
    if setAboveWT == 1:
        for c in mesh.cells():
            if c.center()[1] > 0:
                dataVec[c.id()] = setValue
    
    checkZeros = 1
    if checkZeros == 1:
        if 'MINIT' in data:
            dataVec[dataVec == 0] = 0.0001
        if 'COND' in data:
            dataVec[dataVec < min(data['COND'])] = np.mean(data['COND'])
    
    nData = pg.cellDataToPointData(bPolyMesh, dataVec)

    fig, ax = plt.subplots()
    plt.rc('font', family='Arial', size = 12)
    
    if 'MINIT' in data.columns:
        print('Plotting Mass Concentration data...')
        _ , cbar = pg.show(mesh, dataVec, label="Mass Concentration [mg/l]", cMap=ccmap, cMin=360, cMax=36000, extend="both", logScale=True, colorbar = True, ax = ax)
        pg.mplviewer.drawField(ax, mesh, nData, levels = [500,1000,10000, 20000, 35000], cMin = min(nData), cMax = np.floor(max(nData)), fillContour=False, logScale = True, colors=['black'], linewidths=0.2, alpha=1)
        plt.sca(plt.gcf().get_axes()[0])
        cbar.ax.xaxis.set_ticklabels(np.ceil(cbar.get_ticks()).astype(int))
    
    elif 'COND' in data.columns:
        print('Plotting Hydraulic Conductivity data...')
        _, cb = pg.show(mesh, dataVec, label="Hydrualic Conductivity (m/d)",
                         cMap=cmap,  colorBar=True, cMin=None, cMax=None,
                         extend="both", logScale=True, ax = ax)
        _, _ = pg.show(mesh, dataVec, label="Hydrualic Conductivity (m/d)",
                 cMap=cmap,  colorBar=False, extend="both", logScale=True, ax = ax)
        pg.mplviewer.drawField(ax, mesh, np.log10(nData), cMin=np.log10(min(data['COND'])), nLevs=5, cMax=np.log10(max(data['COND'])),
                               fillContour=False, colors=['black'])
    else:
        print('Plotting whatever...')
        im, cb = pg.mplviewer.drawModel(ax = ax, mesh = mesh, data = dataVec, cMap=cmap, norm = norm, cMin=0.1, cMax=0.3, extend="both")        
 
    #cbar.ax.xaxis.set_ticklabels(np.ceil(cbar.get_ticks()).astype(int))
    #cbar.ax.minorticks_on()
    
# Formatting
    def formatActiveFig():
        fig = plt.gcf()
        fig.set_size_inches(12, 3.5)
    
    
        plt.sca(fig.get_axes()[0])
        plt.plot(topoArray[:, 0], topoArray[:, 1], 'k')
        ax = plt.gca()
        ax.tick_params(which = 'both', direction = 'in')
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.set_xlim(left=-20, right=700)
        ax.set_ylim(top=50, bottom=-30)
        ax.set_aspect(aspect=1)
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Elevation (mASL)')
        #ax.set_title('Solute Distribution -- FEPEST Model',loc= 'left')
    #    ax.set_title('t = ' + str(int(np.round(data['Time'],0).unique()[0])),loc= 'left')
    
    #    ax.set_title('2 ML/yr',loc= 'left')
        #ax.set_title('Hydraulic Conductivity inc. High-Permeability Structure', loc = 'left')
    #    ax.set_title('Hydraulic Conductivity -- FEPEST', loc = 'left')
    #    ax.set_title('Block 2a',loc= 'left')
        fig.set_tight_layout('tight')
    
    
    def plotWells():
        # Plot Wells
        plt.sca(plt.gcf().get_axes()[0])
        WellCoords = pd.read_table(r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\FEFLOW_Supplementary\SIMWell_Coords.dat",  delim_whitespace = True)
    
        def getElev(array, value):
          val = array[abs(array[:,0] - value).argmin()][1]
          return val
    
        for well in WellCoords.iterrows():
        #    print(well)
            plt.plot([well[1]['X'],well[1]['X']], [getElev(topoArray,well[1]['X']),well[1]['Y']], 'k')
            plt.annotate(s = well[1]['LABEL'], xy = [well[1]['X'],8+getElev(topoArray,well[1]['X'])], ha = 'center', fontsize = 12)
    
        plt.plot([min(topoArray[:,0]), max(topoArray[:,0])],[0,0], 'k--')
        
    plotWells()
    formatActiveFig()
    
    filename = fNames[0][:-4]+'.png'
    fig.savefig(fname = filename, format = 'png', dpi = 300)
