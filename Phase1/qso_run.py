from readfits import readSdssFitsFile
from astropy.table import Table
from astropy.table import Column
import os
from qsoParam import parameterization
from QsoCL import *
from plotData import *
from basics import *
import numpy as np
import time
np.set_printoptions(formatter={'float': '{: 3.4f}'.format})



def main():
       
    '''
 data directories
 '''
    feTempDir = '/qsopython/data/Fe_templates/'
    sdssDir = '/qsopython/data/fits/'
    dtpath = '/qsopython/data/'

    '''
 read data fits
 '''
    path_s = os.listdir(sdssDir)
    specFiles = []
    for item in path_s:
        if ".fits" in item:
            specFiles.append(sdssDir + item)
    specFiles.sort()
    '''
 choosing the qso spectrum
 '''
    qso = []
    for i in range(len(specFiles)):
    #for i in range(2):
        qso.append(readSdssFitsFile(specFiles[i]))

    '''
 Loading Data
 '''

    spectralLines, continuumWindows, ampWavelength, feWindows, feTemplate = [], [], 3000, [], []

    path_dt = os.listdir(dtpath)
    for item in path_dt:
        if "spectral_lines" in item:
            spectralLines = Table.read(
                dtpath + item,
                names=(
                    'name',
                    'range0',
                    'range1',
                    'fitGuess0',
                    'fitGuess1',
                    'fitGuess2'),
                format='ascii')
        if "cont_windows" in item:
            continuumWindows = Table.read(
                dtpath + item, format='ascii.no_header')
        if "iron_emission_temp" in item:
            feTemplate = Table.read(
                dtpath + item,
                names=(
                    'wavelengths',
                    'values'),
                format='ascii')
        if "iron_emission_windows" == item:
            print(dtpath + item)
            feWindows = Table.read(dtpath + item, format='ascii.no_header')
      
    print(feWindows)
    col1 = []
    col2 = []

    for i in range(len(spectralLines)):
     row1 = []
     row1.append(spectralLines['range0'][i])
     row1.append(spectralLines['range1'][i])
     col1.append(row1)
     row2 = []
     row2.append(spectralLines['fitGuess0'][i])
     row2.append(spectralLines['fitGuess1'][i])
     row2.append(spectralLines['fitGuess2'][i])
     row2.append(0)
     col2.append(row2)

    range_ = Column(col1, name='range')
    fitGuess = Column(col2, name='fitGuess')

    spectralLines.remove_columns(['range0','range1','fitGuess0','fitGuess1','fitGuess2'])
    spectralLines.add_column(range_,index = 1)
    spectralLines.add_column(fitGuess,index = 2)


    path_fe = os.listdir(feTempDir)
    feTempFiles = []
    for item in path_fe:
     if "broadened.dat" in item:
      feTempFiles.append(feTempDir+item)
    feTempFiles.sort()

    
    
    fitParameters = {
        "fwhmn": 1600.0,
        "fwhmt": 900.0,
        "feScaleRate": 1.0,
        "feFitRange": (
            2200.0,
            2650.0),
        "isSubC": False,
        "fitType": 'WIN'}
    
    options = {
        "spectralLines": spectralLines,
        "continuumWindows": continuumWindows,
        "ampWavelength": ampWavelength,
        "feWindows": feWindows,
        "feTemplate": feTemplate,
        "fitParameters": fitParameters}
    
    QsoCL = QsoCL() #Initialize an object of type QsoCL

    start = time.time()
    result = parameterization(QsoCL,qso,options) # Run Parameterization
    end = time.time()
    t = end - start
   
    print ('\nExecution time: ',(t))
    
    
  
    


if __name__ == '__main__':
    main()
