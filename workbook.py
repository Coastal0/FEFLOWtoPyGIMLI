# -*- coding: utf-8 -*-
"""
This contains most of the subroutines used for FEFLOW conversion.
These are designed to be called from a master script.
"""

def loadHull(hullID):
    """
    hullID is a directory pointing to the concave hull of the pointcloud
    e.g. r"G:\directory\boundsXY.mat"
    """
    import h5py
    import numpy as np
    import pygimli as pg

    # Extract the concave hull from MATLAB output.
    with h5py.File(hullID, 'r') as file:
        hImport = np.array(file['boundsXY'])
        hull = hImport.T
    print(hull.shape, 'coordinates found')

    # Remove duplicate coordinates.
    if (hull[0, :] == hull[-1::]).all():
        print('Duplicate coordinates found (start/end)')
        hull = hull[0:-1]
        print(hull.shape, 'coordinates remaining')

    # Round to 5 decimal places (avoids floatingpoint issues later)
    hull = np.round(hull, decimals = 5)

    # Create the exterior boundary of the FEFLOW mesh from the outer bounds.
    bPoly = pg.meshtools.createPolygon(verts=hull, isClosed=1)
    print('Boundary Polygon (bPoly):', bPoly)
    return hull, bPoly


def loadNodes(nodesID):
    """
    nodesID is a directory pointing to the concave hull of the pointcloud
    e.g. r"G:\directory\boundsXY.mat"
    """
    import h5py
    import numpy as np
    import pygimli as pg
    # Extract the concave hull from MATLAB output.
    with h5py.File(nodesID, 'r') as file:
        print(file.keys())
        hImport = np.array(file['nodesXY'])
        nodes = hImport.T
    print(nodes.shape, 'coordinates found')
    return nodes

def loadDataToDict(fName):
    import pandas as pd
    import numpy as np
    # Read in mass-concentration node data
    data = pd.read_table(fName, delim_whitespace=True)
    if 'node' in data.columns:
        maxNodes = max(data.Node)
    else:
        maxNodes = data.iloc[:,3].max()
    print('Number of nodes found =', maxNodes)
    # Extract coordinates of mesh.
    if 'Time' in data.columns:
        print(pd.unique(data.Time).size, 'time steps found:', pd.unique(data.Time))
        coords = np.round(np.stack((data.X[data.Time == data.Time[0]].values, data.Y[data.Time == data.Time[0]].values), axis=1), decimals = 5)
        print(len(coords), 'X-Y locations found for time', data.Time[0])
#        maxNodes = max(data.Time == data.Time[0])
    else:
#        maxNodes = max(data.Node)
        coords = np.round(np.stack((data.X.values, data.Y.values), axis=1), decimals = 5)
        if maxNodes != coords.shape[0]:
            print('Number of reported nodes =', maxNodes)
            print('Number of nodes found =', coords.shape[0])
            print('Number of nodes does not match. (Inactive elements in FEFLOW?)')
        else:
            print(len(coords), 'X-Y locations found.')
    return data, coords

#from shapely.ops import cascaded_union, polygonize
#import shapely.geometry as geometry
#from scipy.spatial import Delaunay
#import numpy as np
#import math
#def alpha_shape(points, alpha):
#    """
#    Compute the alpha shape (concave hull) of a set
#    of points.
#    @param points: Iterable container of points.
#    @param alpha: alpha value to influence the
#        gooeyness of the border. Smaller numbers
#        don't fall inward as much as larger numbers.
#        Too large, and you lose everything!
#    """
#    if len(points) < 4:
#        # When you have a triangle, there is no sense
#        # in computing an alpha shape.
#        return geometry.MultiPoint(list(points)).convex_hull
#    def add_edge(edges, edge_points, coords, i, j):
#        """
#        Add a line between the i-th and j-th points,
#        if not in the list already
#        """
#        if (i, j) in edges or (j, i) in edges:
#            # already added
#            return
#        edges.add( (i, j) )
#        edge_points.append(coords[ [i, j] ])
##    coords = np.array([point.coords[0] for point in points])
#    tri = Delaunay(coords)
#    edges = set()
#    edge_points = []
#    # loop over triangles:
#    # ia, ib, ic = indices of corner points of the
#    # triangle
#    for ia, ib, ic in tri.vertices:
#        pa = coords[ia]
#        pb = coords[ib]
#        pc = coords[ic]
#        # Lengths of sides of triangle
#        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
#        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
#        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
#        # Semiperimeter of triangle
#        s = (a + b + c)/2.0
#        # Area of triangle by Heron's formula
#        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
#        circum_r = a*b*c/(4.0*area)
#        # Here's the radius filter.
#        #print circum_r
#        if circum_r < 1.0/alpha:
#            add_edge(edges, edge_points, coords, ia, ib)
#            add_edge(edges, edge_points, coords, ib, ic)
#            add_edge(edges, edge_points, coords, ic, ia)
#    m = geometry.MultiLineString(edge_points)
#    triangles = list(polygonize(m))
#    return cascaded_union(triangles), edge_points
#concave_hull, edge_points = alpha_shape(points, alpha=1.87)
#_ = plot_polygon(concave_hull)
#_ = pl.plot(x,y,'o', color='#f16824')

def loadData(fNames, convertTime = 0):
    import pandas as pd
    import numpy as np
    dataDict = {}
    for i, f in enumerate(fNames):
        print('Loading', f)
        dataDict[i] = loadDataToDict(f)

    # Check coordinates match, and combine data columns.
    if len(fNames) > 1:
        print('Multiple input files found...', 'Expecting MASS and SAT')
        if (dataDict[0][1] == dataDict[1][1]).all():
            coords = dataDict[1][1]
            # Make single dataframe entity using keywords from column headers.
            for i in dataDict:
                if 'MINIT' in dataDict[i][0].columns:
                    data = dataDict[i][0]
            for i in dataDict:
                if 'SINIT' in dataDict[i][0].columns:
                    data['SINIT'] = dataDict[i][0]['SINIT']
        else:
            print('Error! Coordinates in files do not match.')
    else:
        print('Single datafile supplied...')
        data = dataDict[0][0]
        coords = dataDict[0][1]

    # Convert time-column to date
    if 'Time' in data.columns and convertTime == 0:
        if len(pd.unique(data['Time'])) > 1:
            print('Time data found, assuming [days] increment')
            import datetime as dt
            startDate = dt.date(1990, 1, 1)
            dTime = []
            for d in data['Time']:
                dTime.append(startDate + dt.timedelta(days = 365 * d))
            data['DateTime'] = dTime

    print('Datafile Headers are:', list(data.columns))
    print('Loading finished.')
    return dataDict, data, coords

def fillPoly(bPoly, coords, hull):
    import numpy as np
    import pygimli as pg
    from tqdm import tqdm

    """
    Takes a boundary polygon and fills with nodes, excluding hullnodes.
    """
    # Fill the empty polygon with nodes
    print('Filling polygon with nodes...')

    roundedCoords = np.round(coords, 3)
    roundedHull = np.round(hull, 3)
    counter = 0
    for node in tqdm(roundedCoords):
        if any(np.all(roundedHull[:] == node, axis = 1)):
            counter = counter + 1
        else:
            bPoly.createNode(node)
    print(counter)
    print(bPoly)
    bPolyMesh = pg.meshtools.createMesh(bPoly, 0)
    return bPolyMesh


def checkMesh(mesh):
    """
    Check if each cell of a mesh has a node.
    """
    # Check each cell/node has a value.
    i = 0
    for n in mesh.nodes():
        if len(n.cellSet()) == 0:
            print(n, n.pos(), " have no cells!")
            i = i+1
    print(str(i)+' nodes have no cells')


def getTopo(hull):
    """
    Takes the hull and finds top boundary values.
    """
    import numpy as np

    topo = hull[(hull[:, 1] >= -5)]
    topo = topo[np.lexsort((topo[:, 1], topo[:, 0]))]
    _, idx, idc = np.unique(topo[:, 0], return_index=1, return_counts=1)
    tu = topo[idx]
    for i in enumerate(idc):
            if i[1] > 1:
                tu[i[0]] = np.max(topo[topo[i[0], 0] == topo[:, 0]], 0)
    _, idx, idc = np.unique(tu[:, 0], return_index=1, return_counts=1)
    topo = tu[idx]
    return topo


def createArray(start, end, spacing, schemeName, topoArray, enlarge = 1):
    """
    Creates an ERT array and makes ertMesh based on array.
    """
    import numpy as np
    import pygimli as pg
    import pybert as pb

    print('Creating array...')
    sensor_firstX = start
    sensor_lastX = end
    sensor_dx = spacing

    sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
    sensor_z = np.interp(sensor_x, topoArray[:, 0], topoArray[:, 1])
    sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T

    if schemeName == 'dd' and enlarge == 1:
        print('Expanding array...')
        ertScheme = pb.createData(sensors, schemeName=schemeName, enlarge = 1)
    else:
        ertScheme = pb.createData(sensors, schemeName=schemeName, enlarge = 0)
        print('Not exapnding array...')
    ertScheme.save('ertScheme')
    # Topography before (left-of) electrodes
    topoPnts_x = np.arange(topoArray[0,0],sensor_firstX,sensor_dx)
    topoPnts_z = np.interp(topoPnts_x, topoArray[:, 0], topoArray[:, 1])
    topoPnts_stack = np.stack([topoPnts_x,np.around(topoPnts_z,2)]).T
    topoPnts = np.insert(sensors[:,[0,1]],0,topoPnts_stack,0)

        # Create ERT mesh (based on sensors)
    print('Creating modelling mesh...')
    meshERT = pg.meshtools.createParaMesh(topoPnts, quality=32,
                                          paraMaxCellSize=3, paraDepth=100,
                                          paraDX=0.01)
    print('ERT Mesh = ', meshERT)
    meshERTName = 'meshERT_'+schemeName+'_'+str(spacing)+'m'
    meshERT.save(meshERTName)
    return ertScheme, meshERT



#%% Data mesh
def makeDataMesh(coords, show):
    import pygimli as pg
    import numpy as np
    print('Creating datamesh from pointcloud...')
    dMesh = pg.meshtools.createMesh(np.ndarray.tolist(coords), quality = 0)
    if show == 1:
        pg.show(dMesh)
    print(dMesh)
    print('Done.')
    return (dMesh)


def makeInterpVector(data, dMesh, bPolyMesh, t = None):
    import pygimli as pg
    import numpy as np
    print('Interpolation data to mesh...')
    if "Time" in data and len(np.unique(data.Time)) > 1:
        if t is not None:
            if t in np.unique(data.Time):
                print('Using specified time value: ', t)
                dataCol = data['MINIT'][data['Time'] == t].values
                dInterp = pg.interpolate(dMesh, dataCol, bPolyMesh.cellCenter())
                print('Done')
                return dInterp, None
            else:
                print('Value not found. No interpolation done...')
                return None, None
        else:
            print('Multiple times found')
            times = np.zeros([int(len(data)/len(np.unique(data.Time))),len(np.unique(data.Time))])
            dInterp = np.zeros([bPolyMesh.cellCount(),len(np.unique(data.Time))])
            for i, t in enumerate(np.unique(data.Time)):
                print("Converting time to data vector:", t)
                times[:,i] = data['MINIT'][data['Time'] == t].as_matrix()
                dInterp[:,i] = pg.interpolate(dMesh, times[:,i], bPolyMesh.cellCenter())
            print('Done')
            return dInterp, times
    else:
        dInterp = pg.interpolate(dMesh, data['dataCol'], bPolyMesh.cellCenter())
        print('Done')
        return dInterp, None


def convertFluid(dInterp, bPolyMesh, meshERT, saveVec=0):
    import numpy as np
    from pygimli.physics.petro import resistivityArchie as pgArch

    print('Converting fluid cond to formation cond...')
    k = 0.7  # Linear conversion factor from TDS to EC
    sigmaFluid = dInterp / (k*10000)  # dInterp (mg/L) to fluid conductivity (S/m)
    print('Fluid conductivity range: ', min(1000*sigmaFluid), max(1000*sigmaFluid), 'mS/m')
    rFluid = 1/sigmaFluid
#    print(rFluid)
    print('Interpolating mesh values...')
    resBulk = pgArch(rFluid, porosity=0.3, m=2, mesh=bPolyMesh, meshI=meshERT, fill=1)
    print('No.# Values in fluid data',resBulk.shape[0])
    print('No.#Cells in ERT Mesh: ',meshERT.cellCount())
    print('No.# Data == No.# Cells?', resBulk.shape[0] == meshERT.cellCount())

    # Apply background resistivity model
    for c in meshERT.cells():
        if c.center()[1] > 0:
            resBulk[c.id()] = 1000. # Resistivity of the vadose zone
        elif c.center()[1] < -30:
            resBulk[c.id()] = 20. # Resistivity of the substrate

    for c in meshERT.cells():
        if c.marker() == 1 and c.center()[0] < 0 and c.center()[1] > -30:
            resBulk[c.id()] = 2 # Resistivity of the ocean-side forward modelling region.
    print('Done.')

    if saveVec == 1:
        print('Saving...')
        resBulkName = 'resBulk_.vector'
        np.savetxt(resBulkName,resBulk)
    return(resBulk)


def simulate(meshERT, resBulk, ertScheme, fName):
    import pybert as pb
    import numpy as np

    ert = pb.ERTManager(debug=True)
    print('#############################')
    print('Forward Modelling...')

    # Set geometric factors to one, so that rhoa = r
#    ertScheme.set('k', pb.geometricFactor(ertScheme))
    ertScheme.set('k', np.ones(ertScheme.size()))

    simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme,
                                  noiseAbs=0.0,  noiseLevel = 0.01)
    simdata.set("r", simdata("rhoa"))

    # Calculate geometric factors for flat earth
    flat_earth_K = pb.geometricFactors(ertScheme)
    simdata.set("k", flat_earth_K)

    # Set output name
    dataName = fName[:-4]+'_data.ohm'
    simdata.save(dataName, "a b m n r err k")
    print('Done.')
    print(str('#############################'))
    #    pg.show(meshERT, resBulk)
    return simdata


def showModel(bPolyMesh, dInterp,fName):
    import pygimli as pg
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax, cb = pg.show(bPolyMesh, dInterp, colorBar=True, cmap='jet', showMesh=0, cMin = 50, cMax = 35000)
    fig = plt.gcf()
    fig.set_size_inches(17, 4)
    ax.minorticks_on()
#    ax.set_xlim(left=-20, right=600)
#    ax.set_ylim(top=35, bottom=-30)

    ax.set_title(fName,loc= 'left')
    ax.set_ylabel('Elevation (mASL)')
    ax.set_xlabel('Distance (m)')

    cb.ax.minorticks_on()
    cb.ax.xaxis.set_ticklabels(cb.get_ticks().astype(int), minor = False)
    cb.set_label('Formation Resistivity ($\Omega$$\cdot$m)')
    cb.set_label('Mass Concentration [mg/L]')
    fig.set_tight_layout('tight')
    return fig, ax, cb