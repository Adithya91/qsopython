#!/usr/bin/env python

import datetime
from time import time
import numpy as np
import math
from spectrum import *
from continuum import *
from tools import *
from plotData import *
from fe import *
from basics import *
from mavg import *
from gaussian import *


ASTRO_OBJ_SPEC_SIZE = 4096

#To make default spectrum size as 4096
def expandArray(inpmatrix, size):
    newMatrix = np.zeros(size)
    for i in range(len(inpmatrix)):
        newMatrix[i] = inpmatrix[i]
    ind = np.where(newMatrix == 0)
    for j in ind:
        #newMatrix[j] = math.inf
        newMatrix[j] = 0
    return newMatrix

#To calculate iron template scale rates
def calcFeTemplateScaleRates(QuasarCL,spectrumsMatrix_,templateFeMatrix_,sizesVector_,fitParametersList_):
 h = spectrumsMatrix_.shape[0]  # spectrumsSize
 w = spectrumsMatrix_.shape[1]  # spectrumsNumber
 
 sizes = sizesVector_
 spectrumsMatrix = spectrumsMatrix_
 templateFeMatrix = templateFeMatrix_
 fitParameters = fitParametersList_
 reglinYaxResults = np.transpose(np.zeros(w,dtype=np.float64,order='F'))
 reglinYaxResultsVec = reglinYax(QuasarCL,templateFeMatrix, spectrumsMatrix, w, h,
				sizes, reglinYaxResults);
 
 for i in range(len(reglinYaxResultsVec)):
   reglinYaxResultsVec[i] = reglinYaxResultsVec[i]*fitParameters['feScaleRate']

 
 reglinYaxResults = reglinYaxResultsVec
 return reglinYaxResults



#To perform convolution 
def cpuConvolve(signal,kernel,same):
 result_size = (len(signal)+len(kernel))-1
 result = list(np.zeros(result_size))
 for i in range(result_size):
  if i>=len(kernel)-1:
    kmin = i - (len(kernel)-1) 
  else:
    kmin = 0
  if i<len(signal)-1:
    kmax = i
  else:
    kmax = len(signal)-1
  for k in range(kmin,kmax):
   result[i]+=signal[k]*kernel[i-k]
 
 if(same):
  if len(kernel)%2 == 0:
   kernel_center = math.ceil((len(kernel)-1)/2)
  else:
   kernel_center =  math.ceil(len(kernel)/2)
  del result[0:kernel_center]
  del result[result_size - (len(kernel)-1-kernel_center):result_size] 
 
 return result


#To generate iron template matrix
def calcFeTemplateMatrix(QuasarCL, wavelengthsMatrix, sizesVector_,feTemplateList_, fitParametersList_):
 h = wavelengthsMatrix.shape[0]  # spectrumsSize
 w = wavelengthsMatrix.shape[1]  # spectrumsNumber
 
 if fitParametersList_['fwhmn']>fitParametersList_['fwhmt']:
  C = 299792458.0
  sincov = math.pow(math.pow(fitParametersList_['fwhmn'],2.0)-math.pow(fitParametersList_['fwhmt'],2.0),0.5)/2.0
  sincov /= math.pow(2.0*math.log(2.0),0.5)*C/ 1e3;
  
  feGauss = list(feTemplateList_['wavelengths'])
    
  for i in range(len(feGauss)):
   feGauss[i] = math.log10(feGauss[i])
   
  a = 1.0 / sincov / math.pow(2 * math.pi, 0.5)
  b = feGauss[math.ceil(len(feGauss) / 2)]
  c = sincov
  d = 0.0
  
  for j in range(len(feGauss)):
   feGauss[j] = (a* math.exp(-0.5* (feGauss[j]-b) * (feGauss[j]-b)/math.pow(c,2.0))) + d
   	
  feValues = feTemplateList_['values']
  templateFeValues = np.asarray(cpuConvolve(feValues, feGauss, 'true'), dtype=np.float64, order='F')
 
 else:
  templateFeValues = feTemplateList_['values']

 

 templateFeMatrix = np.zeros((w,ASTRO_OBJ_SPEC_SIZE),dtype = np.float64)
 zeros = np.zeros((w,ASTRO_OBJ_SPEC_SIZE),dtype = np.float64)
 templateFeMatrix = np.transpose(np.asarray(templateFeMatrix, dtype=np.float64, order='F'))
 zeros = np.transpose(np.asarray(zeros, dtype=np.float64, order='F'))

 templateFeValues = feTemplateList_['values']
 templateFeLambda = feTemplateList_['wavelengths']

 templateFeMatrix = addSpectrum(QuasarCL,zeros, wavelengthsMatrix, sizesVector_, w,
	     templateFeLambda, templateFeValues, len(templateFeValues));
 return templateFeMatrix

#To perform iron emission fitting
def feFitTest(QuasarCL,spectrumsData, feTemplate, feWindows, fitParameters):
 spectrumsMatrix = spectrumsData['spectrumsMatrix']
 wavelengthsMatrix = spectrumsData['wavelengthsMatrix']
 errorsMatrix = spectrumsData['errorsMatrix']
 continuumsMatrix = spectrumsData['continuumsMatrix']
 sizes = spectrumsData['sizes']
 
 templateFeMatrix = calcFeTemplateMatrix(QuasarCL, wavelengthsMatrix, sizes, feTemplate, fitParameters)
 
 q=11
 plot(wavelengthsMatrix[0:sizes[q],q], templateFeMatrix[0:sizes[q],q],'black','Wavelength [Å]','flux[erg $s^-1$ $cm^-1$ $A^-1$]','template','file')

 
 templateFeMatrixCopy = templateFeMatrix
 if(~fitParameters['isSubC']):
  spectrumsMatrix = minusMatrix(QuasarCL,spectrumsMatrix, continuumsMatrix)

 plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q],'black','Wavelength [Å]','flux[erg $s^-1$ $cm^-1$ $A^-1$]','template','file')


 if(fitParameters['fitType']=='WIN'or fitParameters['fitType']=='FWIN'):
  h = spectrumsMatrix.shape[0]  # spectrumsSize
  w = spectrumsMatrix.shape[1]  # spectrumsNumber
  templateFeMatrixCopy,reducedChisqsFiltered,scaleRates,sizesFeWindows = filtered_chisqs(QuasarCL,QuasarCL.makeBuffer(wavelengthsMatrix),
                    					  QuasarCL.makeBuffer(errorsMatrix),
                    					  QuasarCL.makeBuffer(spectrumsMatrix), 
                                                          QuasarCL.makeBuffer(continuumsMatrix), 
                                                          QuasarCL.makeBuffer(templateFeMatrix), 
                                                          QuasarCL.makeBuffer(sizes),fitParameters,h,w)
  
 spectrumsMatrix = spectrumsData['spectrumsMatrix']
 wavelengthsMatrix = spectrumsData['wavelengthsMatrix']
 errorsMatrix = spectrumsData['errorsMatrix']
 continuumsMatrix = spectrumsData['continuumsMatrix']	
 
 
 if (~fitParameters['isSubC']):
  spectrumsMatrix = minusMatrix(QuasarCL,spectrumsMatrix, continuumsMatrix)
 
 reducedChisqsFull = calcReducedChisqs(QuasarCL,spectrumsMatrix, templateFeMatrixCopy,
			errorsMatrix, sizes)
 temp = divideR(QuasarCL,templateFeMatrixCopy, continuumsMatrix)
 ewsFull = trapz(QuasarCL,temp, wavelengthsMatrix, sizes)
 templateFeMatrix = templateFeMatrixCopy
 

 reducedChisqsFeRange = reducedChisqsFull
 ewsFeRange = ewsFull
 plot(wavelengthsMatrix[0:sizes[q],q], templateFeMatrix[0:sizes[q],q],'black','Wavelength [Å]','flux[erg $s^-1$ $cm^-1$ $A^-1$]','template','file')
 fedict = {
           "feTemplateMatrix" : templateFeMatrixCopy, 
	   "feScaleRates" : scaleRates, 
	   "feWindowsSizes" : sizesFeWindows, 
	   "feWindowsReducedChisqs" : reducedChisqsFiltered,
	   "feFullReducedChisqs" : reducedChisqsFull, 
	   "feFullEWs" : ewsFull, 
	   "feRangeReducedChisqs" : reducedChisqsFeRange, 
	   "feRangeEWs" : ewsFeRange
           }
 return fedict

#To determine the continuum curve
def continuumTest(QuasarCL,
        spectrumsMatrix,
        wavelengthsMatrix,
        errorsMatrix,
        sizes,
        windows,
        ampWavelength):



    wVec = list(windows['col1'])
    wVec.extend(list(windows['col2']))

    h = spectrumsMatrix.shape[0]  # spectrumsSize
    w = spectrumsMatrix.shape[1]  # spectrumsNumber
    winSize = len(wVec)
    
    
    # kernel to kernel buffer transfer
    
    filteredMatrices = filter_matrix(QuasarCL,
        QuasarCL.makeBuffer(spectrumsMatrix), QuasarCL.makeBuffer(wavelengthsMatrix), QuasarCL.makeBuffer(errorsMatrix),
        QuasarCL.makeBuffer(sizes), QuasarCL.makeBuffer(np.asarray(wVec)),h,w,winSize)

    
    spectrumsMatrixFiltered = filteredMatrices['spectrumsMatrix']
    wavelengthsMatrixFiltered = filteredMatrices['wavelengthsMatrix']
    errorsMatrixFiltered = filteredMatrices['errorsMatrix']
    newSizes = filteredMatrices['newSizes']
    maxSize = filteredMatrices['maxSize']
       

    q=11
    plot(wavelengthsMatrixFiltered[0:newSizes[q],q], spectrumsMatrixFiltered[0:newSizes[q],q],'black','Wavelength [Å]','flux[erg $s^-1$ $cm^-1$ $A^-1$]','filtered flux','file')

    spectrumsMatrixFilteredCopy = log10(QuasarCL,spectrumsMatrixFiltered)
    wavelengthsMatrixFilteredCopy = log10(QuasarCL,wavelengthsMatrixFiltered)

    
    plot(wavelengthsMatrixFilteredCopy[0:newSizes[q],q], spectrumsMatrixFilteredCopy[0:newSizes[q],q],'black','Wavelength [Å]','flux[erg $s^-1$ $cm^-1$ $A^-1$]','filtered flux log10','file')
    
    reglin = reglin_results(QuasarCL,
        QuasarCL.makeBuffer(wavelengthsMatrixFilteredCopy),
        QuasarCL.readBuffer(spectrumsMatrixFilteredCopy),
        ampWavelength,
        QuasarCL.readBuffer(newSizes),h,w)
    
   

    continuumMatrixFiltered = calcCw(QuasarCL,wavelengthsMatrixFiltered, reglin['cReglinResults'])
    
    
    plot(wavelengthsMatrixFiltered[0:newSizes[q],q], continuumMatrixFiltered[0:newSizes[q],q],'black','Wavelength [Å]','flux[erg $s^-1$ $cm^-1$ $A^-1$]','filtered continuum of 12th quasar','file')

    chisqsFiltered = chisqs(QuasarCL,
        QuasarCL.readBuffer(spectrumsMatrixFiltered),
        QuasarCL.readBuffer(continuumMatrixFiltered),
        QuasarCL.readBuffer(errorsMatrixFiltered),
        QuasarCL.readBuffer(newSizes),h,w)

    cfunDcfun = calcCfunDcfun(QuasarCL,wavelengthsMatrix, reglin['cReglinResults'], reglin['reglinResults'])
    

    plot(wavelengthsMatrix[0:newSizes[q],q], cfunDcfun['continuum'][0:newSizes[q],q],'black','Wavelength [Å]','flux[erg $s^-1$ $cm^-1$ $A^-1$]','filtered continuum of 12th quasar','file')

    continuumMatrixAltered = calcCw(QuasarCL,wavelengthsMatrix, reglin['cReglinResults'])

    continuum_dict = {
        "dcontinuumsMatrix": cfunDcfun['dContinuum'],
        "continuumsMatrix": continuumMatrixAltered,
        "continuumChisqs": chisqsFiltered,
        "continuumReglin": reglin['reglinResults'],
        "reglin": reglin['cReglinResults']
        }
    return continuum_dict


#To compute elements for Gaussian fitting
def fitElementTest(QuasarCL,spectrumsLinesMatrix, continuumsMatrix, wavelengthsMatrix, 
		   errorsMatrix, sizesVector, element):
 spectrumsLinesMatrixCopy = spectrumsLinesMatrix
 wavelengthsMatrixCopy = wavelengthsMatrix
 sizesCopy = sizesVector
 el_range = list(element['range'])
 element_range_list = []
 for i in range(len(el_range)):
  for j in range(len(el_range[0])):
   element_range_list.append(el_range[i][j])
 
 h = spectrumsLinesMatrix.shape[0]  # spectrumsSize
 w = spectrumsLinesMatrix.shape[1]  # spectrumsNumber
 
 elemSize = len(element_range_list)
 
 filteredMatrices = filter_matrix_fit(QuasarCL,
        QuasarCL.makeBuffer(spectrumsLinesMatrix), 
        QuasarCL.makeBuffer(wavelengthsMatrix), 
        QuasarCL.makeBuffer(continuumsMatrix),
        QuasarCL.makeBuffer(sizesVector), 
        QuasarCL.makeBuffer(np.asarray(element_range_list)),h,w,elemSize) 

 spectrumsLinesMatrix= filteredMatrices['spectrumLinesMatrix']
 wavelengthsMatrix = filteredMatrices['wavelengthsMatrix']
 continuumsMatrix = filteredMatrices['continuumsMatrix']
 sizes = filteredMatrices['sizes']
 
 el_fg = list(element['fitGuess'])
 element_fg_list = []
 for i in range(len(el_fg)):
  for j in range(len(el_fg[0])):
   element_fg_list.append(el_fg[i][j])
 fitGResults = element_fg_list * continuumsMatrix.shape[0]
 
 fitG_Results = fit_Gaussian(QuasarCL,QuasarCL.makeBuffer(spectrumsLinesMatrix), 
                QuasarCL.makeBuffer(wavelengthsMatrix),
                QuasarCL.makeBuffer(continuumsMatrix), 
                QuasarCL.makeBuffer(errorsMatrix),
                QuasarCL.makeBuffer(wavelengthsMatrixCopy),
                QuasarCL.makeBuffer(spectrumsLinesMatrixCopy),
                QuasarCL.makeBuffer(sizes),
                QuasarCL.makeBuffer(sizesCopy),
                fitGResults,h,w)
 
 elDict = {"fitParams": fitG_Results['fitResults'],
	    "ews" : fitG_Results['ews'],
	    "chisqs":fitG_Results['chisqs'],
	    "gaussianFWHMs": fitG_Results['gaussianFWHMs']
            }
 return elDict
 

#Parameterization main function
def parameterization(QuasarCL,qso,options):
    
    spectralLines= options['spectralLines']
    continuumWindows = options['continuumWindows']
    ampWavelength = options['ampWavelength']
    feWindows= options['feWindows']
    fitParameters = options['fitParameters']	
    feTemplate = options['feTemplate']
    spectrumMatrix = []
    wavelengthMatrix = []
    sizesVector = []
    errorsMatrix = []

    for i in range(len(qso)):
        spectrumMatrix.append(qso[i]['flux'])
        sizesVector.append(qso[i]['flux'].shape[0])
        wavelengthMatrix.append(qso[i]['wavelength'])
        errorsMatrix.append(qso[i]['error'])

    
    for i in range(len(spectrumMatrix)):
        spectrumMatrix[i] = expandArray(
            spectrumMatrix[i], ASTRO_OBJ_SPEC_SIZE)
        wavelengthMatrix[i] = expandArray(
            wavelengthMatrix[i], ASTRO_OBJ_SPEC_SIZE)
        errorsMatrix[i] = expandArray(errorsMatrix[i], ASTRO_OBJ_SPEC_SIZE)    
   
    
    
    sizesVec = np.asarray(sizesVector, dtype=np.uint32)

    # Transpose of the matrices

    wavMat = np.transpose(
        np.asarray(
            wavelengthMatrix,
            dtype=np.float64,
            order='F'))
    specMat = np.transpose(np.asarray(spectrumMatrix, dtype=np.float64, order='F'))
    errMat = np.transpose(np.asarray(errorsMatrix, dtype=np.float64, order='F'))
       
        
    window = 50  # window of average

    specMat = centered_mavg(QuasarCL,
        specMat,
        sizesVec,
        window)

    filteredMatrices = filterZeros(QuasarCL,
        errMat,
        specMat,
        wavMat,
        sizesVec)

    newSizes = countIfNotInf(QuasarCL,filteredMatrices['bMatrix'])
    maxSize = max(newSizes)
    

    spectrumsMatrix = filteredMatrices['aMatrix']
    wavelengthsMatrix = filteredMatrices['bMatrix']
    errorsMatrix = filteredMatrices['spectrumsMatrix']

    spectrumsMatrixCopy = spectrumsMatrix
    
    
    max_iter = 1
   
    for i in range(max_iter):
     continuumResults = continuumTest(QuasarCL,
        spectrumsMatrix,
        wavelengthsMatrix,
        errorsMatrix,
        newSizes, continuumWindows, ampWavelength)


     spectrumsData = {"spectrumsMatrix" : spectrumsMatrix,
		     "wavelengthsMatrix" : wavelengthsMatrix,
		     "errorsMatrix" : errorsMatrix,
		     "continuumsMatrix" : continuumResults['continuumsMatrix'],
		     "sizes" : newSizes}

     sizes=spectrumsData['sizes']
     continuumsMatrix=spectrumsData['continuumsMatrix']
     sizesCOPY=sizes
     continuumsMatrixCOPY=continuumsMatrix
     wavelengthsMatrixCOPY=wavelengthsMatrix
     q=11
     plt.xlabel("Wavelength [Å]")
     plt.ylabel("flux[erg $s^-1$ $cm^-1$ $A^-1$]")
     plt.plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], continuumsMatrix[0:sizes[q],q])
     plt.show()
     plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q],'black','Wavelength [Å]','flux[erg $s^-1$ $cm^-1$ $A^-1$]','12th quasar','file')
     plot(wavelengthsMatrix[0:sizes[q],q], continuumsMatrix[0:sizes[q],q],'black','Wavelength [Å]','flux[erg $s^-1$ $cm^-1$ $A^-1$]','final continuum of 12th quasar','file')

     feResults = feFitTest(QuasarCL, spectrumsData, feTemplate, feWindows, fitParameters)
     sizes=sizesCOPY
     continuumsMatrix=continuumsMatrixCOPY
     wavelengthsMatrix=wavelengthsMatrixCOPY
     print(feResults['feTemplateMatrix'][0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], continuumsMatrix[0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], feResults['feTemplateMatrix'][0:sizes[q],q])
     #plt.show()
     if i < max_iter:
      spectrumsMatrix = minusMatrix(QuasarCL,spectrumsMatrix, feResults['feTemplateMatrix'])
    
     spectrumsEmissionLines = minusMatrix(QuasarCL,spectrumsMatrix, feResults['feTemplateMatrix'])
     spectrumsEmissionLines = minusMatrix(QuasarCL,spectrumsEmissionLines, continuumResults['continuumsMatrix'])
    
     fitElementsResults = fitElementTest( QuasarCL,spectrumsEmissionLines,continuumResults['continuumsMatrix'], 
					 wavelengthsMatrix, errorsMatrix, newSizes,spectralLines)
    
     spectrumsMatrixNoIron = minusMatrix(QuasarCL,spectrumsMatrix, feResults['feTemplateMatrix'])
     spectrumsMatrixEmissionLines = minusMatrix(QuasarCL,spectrumsMatrixNoIron,continuumResults['continuumsMatrix'])
     
     gaussiansingle = lambda a,c,sig,x: a*np.exp(-.5*((x-c)/sig)**2)
     print(qso[q]['name'])
     plt.xlabel("Wavelength [Å]")
     plt.ylabel("flux[erg $s^-1$ $cm^-1$ $A^-1$]")     
     plt.plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrixEmissionLines[0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], continuumsMatrix[0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], feResults['feTemplateMatrix'][0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q],gaussiansingle(fitElementsResults['fitParams'][0],fitElementsResults['fitParams'][1],
     fitElementsResults['fitParams'][2],wavelengthsMatrix[0:sizes[q],q]))
     
     plt.show()

     paramDict = {
                 "continuumChisqs": continuumResults['continuumChisqs'],
		 "continuumReglin": continuumResults['continuumReglin'],
		 "reglin": continuumResults['reglin'],
		 "feScaleRates": feResults['feScaleRates'],
		 "feWindowsSizes": feResults['feWindowsSizes'],
		 "feWindowsReducedChisqs": feResults['feWindowsReducedChisqs'],
		 "feFullReducedChisqs":feResults['feFullReducedChisqs'],
		 "feFullEWs":feResults['feFullEWs'],
		 "feRangeReducedChisqs":feResults['feRangeReducedChisqs'],
		 "feRangeEWs": feResults['feRangeEWs'],
		 "elementsFits": fitElementsResults,
		 "spectrumsEmissionLines": spectrumsEmissionLines,
		 "fitElementsResults": fitElementsResults,
		 "newSizes": newSizes,
		 "wavelengthsMatrix": wavelengthsMatrix,
		 "spectrumsMatrix":spectrumsMatrix,
                }
    return paramDict
