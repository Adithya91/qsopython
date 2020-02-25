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
def calcFeTemplateScaleRates(QsoCL,spectrumsMatrix_,templateFeMatrix_,sizesVector_,fitParametersList_):
 h = spectrumsMatrix_.shape[0]  # spectrumsSize
 w = spectrumsMatrix_.shape[1]  # spectrumsNumber
 
 sizes = sizesVector_
 spectrumsMatrix = spectrumsMatrix_
 templateFeMatrix = templateFeMatrix_
 fitParameters = fitParametersList_
 reglinYaxResults = np.transpose(np.zeros(w,dtype=np.float64,order='F'))
 reglinYaxResultsVec = reglinYax(QsoCL,templateFeMatrix, spectrumsMatrix, w, h,
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
def calcFeTemplateMatrix(QsoCL,wavelengthsMatrix, sizesVector_, feTemplateList_, fitParametersList_):
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

 templateFeMatrix = addSpectrum(QsoCL,zeros, wavelengthsMatrix, sizesVector_, w,
	     templateFeLambda, templateFeValues, len(templateFeValues));
 return templateFeMatrix


#To perform iron emission fitting
def feFitTest(QsoCL,spectrumsData, feTemplate, feWindows, fitParameters):
 spectrumsMatrix = spectrumsData['spectrumsMatrix']
 wavelengthsMatrix = spectrumsData['wavelengthsMatrix']
 errorsMatrix = spectrumsData['errorsMatrix']
 continuumsMatrix = spectrumsData['continuumsMatrix']
 sizes = spectrumsData['sizes']
 
 templateFeMatrix = calcFeTemplateMatrix(QsoCL, wavelengthsMatrix, sizes, feTemplate, fitParameters)
 q=11
 print(templateFeMatrix.shape)
 #for i in range(len(templateFeMatrix[0:sizes[q],q])): print(templateFeMatrix[i,q])
 print(len(templateFeMatrix[0:sizes[q],q]))
 print(sizes)
 print(wavelengthsMatrix[0:sizes[q],q])
 print(len(wavelengthsMatrix[0:sizes[q],q]))
 


 templateFeMatrixCopy = templateFeMatrix
 if(~fitParameters['isSubC']):
  spectrumsMatrix = minusMatrix(QsoCL,spectrumsMatrix, continuumsMatrix)



 if(fitParameters['fitType']=='WIN'or fitParameters['fitType']=='FWIN'):
  print(feWindows)
  feWin = list(feWindows['col1'])
  feWin = feWin + list(feWindows['col2'])
  print(spectrumsMatrix.shape)
  print(wavelengthsMatrix.shape)
  print(errorsMatrix.shape)
  print(sizes)
  print(feWin)
  #filteredMatrices = filterWithWavelengthWindows(QsoCL,spectrumsMatrix, wavelengthsMatrix, errorsMatrix,
  #						 sizes, feWin)
  #spectrumsMatrix = filteredMatrices['spectrumsMatrix']
  #wavelengthsMatrix = filteredMatrices['wavelengthsMatrix']
  #errorsMatrix = filteredMatrices['errorsMatrix']
  #print(spectrumsMatrix.shape)
  #print(wavelengthsMatrix.shape)
  #print(errorsMatrix.shape)
  #plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q],'black','wavelength','flux','template','file')

  filteredMatrices = filterInfs(QsoCL,spectrumsMatrix, continuumsMatrix, templateFeMatrix, sizes)
  spectrumsMatrix = filteredMatrices['spectrumsMatrix']
  continuumsMatrix = filteredMatrices['aMatrix']
  templateFeMatrix = filteredMatrices['bMatrix']
  print(spectrumsMatrix.shape)
  print(wavelengthsMatrix.shape)
  print(errorsMatrix.shape)
  print("przed plot")
  q=11
  print(spectrumsMatrix[0:sizes[q],q])
  plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q],'black','wavelength','flux','template','file')

  
 if(fitParameters['fitType']=='FWIN'):
  filteredMatrices = filterZeros(QsoCL,templateFeMatrix, wavelengthsMatrix, errorsMatrix, sizes)
  templateFeMatrix = filteredMatrices['spectrumsMatrix']
  wavelengthsMatrix = filteredMatrices['aMatrix']
  errorsMatrix = filteredMatrices['bMatrix'] 
   
  filteredMatrices = filterInfs(QsoCL,templateFeMatrix, continuumsMatrix, spectrumsMatrix, sizes)
  templateFeMatrix = filteredMatrices['spectrumsMatrix']
  continuumsMatrix = filteredMatrices['aMatrix']
  spectrumsMatrix = filteredMatrices['bMatrix']
  
 print(sizes)
 spectrumsMatrixFiltered = spectrumsMatrix
 wavelengthsMatrixFiltered = wavelengthsMatrix
 errorsMatrixFiltered = errorsMatrix
 continuumsMatrixFiltered = continuumsMatrix
 templateFeMatrixFiltered = templateFeMatrix
 sizesFeWindows = sizes
 maxSpectrumSize = ASTRO_OBJ_SPEC_SIZE
 
 if(fitParameters['fitType']=='WIN'or fitParameters['fitType']=='FWIN'):
  sizesFeWindows = countIfNotInf(QsoCL,filteredMatrices['spectrumsMatrix'])
  maxSize = max(sizesFeWindows)
  spectrumsMatrixFiltered = copyIfNotInf(QsoCL,spectrumsMatrix, maxSize)
  wavelengthsMatrixFiltered = copyIfNotInf(QsoCL,wavelengthsMatrix, maxSize)
  errorsMatrixFiltered = copyIfNotInf(QsoCL,errorsMatrix, maxSize)
  continuumsMatrixFiltered = copyIfNotInf(QsoCL,continuumsMatrix, maxSize)
  templateFeMatrixFiltered = copyIfNotInf(QsoCL,templateFeMatrix, maxSize) 
 
 print("before scale")
 scaleRates = calcFeTemplateScaleRates(QsoCL,spectrumsMatrixFiltered, 
					templateFeMatrixFiltered, sizesFeWindows, fitParameters)
 print(scaleRates)
 templateFeMatrixFiltered = multiplyCol(QsoCL,templateFeMatrixFiltered, scaleRates)
 templateFeMatrixCopy = multiplyCol(QsoCL,templateFeMatrixCopy, scaleRates)
 print("after multicol")
	
 reducedChisqsFiltered = calcReducedChisqs(QsoCL,spectrumsMatrixFiltered, templateFeMatrixFiltered,
					   errorsMatrixFiltered, sizesFeWindows)
 spectrumsMatrix = spectrumsData['spectrumsMatrix']
 wavelengthsMatrix = spectrumsData['wavelengthsMatrix']
 errorsMatrix = spectrumsData['errorsMatrix']
 continuumsMatrix = spectrumsData['continuumsMatrix']	
 print("after calc")
 
 if (~fitParameters['isSubC']):
  spectrumsMatrix = minusMatrix(QsoCL,spectrumsMatrix, continuumsMatrix)
 
 reducedChisqsFull = calcReducedChisqs(QsoCL,spectrumsMatrix, templateFeMatrixCopy,
			errorsMatrix, sizes)
 temp = divideR(QsoCL,templateFeMatrixCopy, continuumsMatrix)
 ewsFull = trapz(QsoCL,temp, wavelengthsMatrix, sizes)
 templateFeMatrix = templateFeMatrixCopy
 
# if(fitParameters['fitType']=='WIN'or fitParameters['fitType']=='FWIN'):
#  filteredMatrices = filterWithWavelengthWindows(QsoCL,spectrumsMatrix, wavelengthsMatrix, 
#						errorsMatrix, sizes, list(fitParameters['feFitRange']))
#  spectrumsMatrix = filteredMatrices['spectrumsMatrix']
#  wavelengthsMatrix = filteredMatrices['wavelengthsMatrix']
#  errorsMatrix = filteredMatrices['errorsMatrix']
  
#  filteredMatrices = filterInfs(QsoCL,spectrumsMatrix, continuumsMatrix, templateFeMatrix,sizes)
#  templateFeMatrix = filteredMatrices['bMatrix']
#  continuumsMatrix = filteredMatrices['aMatrix']
#  spectrumsMatrix = filteredMatrices['spectrumsMatrix']
  
#  sizesFiltered = countIfNotInf(QsoCL,spectrumsMatrix)
#  maxSize = max(sizesFiltered)
  
#  spectrumsMatrix = copyIfNotInf(QsoCL,spectrumsMatrix, maxSize)
#  wavelengthsMatrix = copyIfNotInf(QsoCL,wavelengthsMatrix, maxSize)
#  errorsMatrix= copyIfNotInf(QsoCL,errorsMatrix, maxSize)
#  continuumsMatrix = copyIfNotInf(QsoCL,continuumsMatrix, maxSize)
#  templateFeMatrix = copyIfNotInf(QsoCL,templateFeMatrix, maxSize)

#  reducedChisqsFeRange = calcReducedChisqs(QsoCL,spectrumsMatrix, templateFeMatrix,
#					     errorsMatrix, sizesFiltered)
											   
#  templateFeMatrix = divideR(QsoCL,templateFeMatrix, continuumsMatrix)
#  ewsFeRange = trapz(QsoCL,templateFeMatrix, wavelengthsMatrix, sizesFiltered)
# else:
 reducedChisqsFeRange = reducedChisqsFull
 ewsFeRange = ewsFull
 plot(wavelengthsMatrix[0:sizes[q],q], templateFeMatrix[0:sizes[q],q],'black','wavelength','flux','template','file')
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
def continuumTest(QsoCL,
        spectrumsMatrix,
        wavelengthsMatrix,
        errorsMatrix,
        sizes,
        windows,
        ampWavelength):

#    q=0
#    plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q],'black','wavelength','flux','usual flux','file')

    wVec = list(windows['col1'])
    wVec.extend(list(windows['col2']))

    h = spectrumsMatrix.shape[0]  # spectrumsSize
    w = spectrumsMatrix.shape[1]  # spectrumsNumber
    winSize = len(wVec)
    
    filteredMatrices = filterWithWavelengthWindows(QsoCL,
        spectrumsMatrix, wavelengthsMatrix, errorsMatrix, sizes, wVec)
    

    filteredMatrices = filterNonpositive(QsoCL,filteredMatrices['spectrumsMatrix'], 
                       filteredMatrices['wavelengthsMatrix'], filteredMatrices['errorsMatrix'], sizes)
    

    newSizes = countIfNotInf(QsoCL,filteredMatrices['spectrumsMatrix'])
        
    maxs = max(newSizes)
    maxs = maxs if maxs > 0 else 64
    maxSize = QsoCL.calcGlobalSize(maxs)
    print('maxSize',maxSize)
    

    spectrumsMatrixFiltered = copyIfNotInf(QsoCL,filteredMatrices['spectrumsMatrix'], maxSize)
    wavelengthsMatrixFiltered = copyIfNotInf(QsoCL,filteredMatrices['wavelengthsMatrix'], maxSize)
    errorsMatrixFiltered = copyIfNotInf(QsoCL,filteredMatrices['errorsMatrix'], maxSize)
    
    
    

    

    spectrumsMatrixFilteredCopy = log10(QsoCL,spectrumsMatrixFiltered)
    wavelengthsMatrixFilteredCopy = log10(QsoCL,wavelengthsMatrixFiltered)

    

    cReglinResults = reglinR(QsoCL,
        wavelengthsMatrixFilteredCopy,
        spectrumsMatrixFilteredCopy,
        newSizes)

   

    if ampWavelength > (math.pow(2.225074e-308, 10.001)):
        lampLog10 = math.log10(ampWavelength)
        wavelengthsMatrixFilteredCopy = minusScalar(QsoCL, wavelengthsMatrixFilteredCopy, lampLog10)

    reglinResults = reglinR(QsoCL, wavelengthsMatrixFilteredCopy, spectrumsMatrixFilteredCopy, newSizes)

    #print(reglinResults)
    
    reglin = fixReglinResults(QsoCL,cReglinResults, reglinResults)

    #print(reglin['cReglinResults'])

    #print(reglin['reglinResults'])

    continuumMatrixFiltered = calcCw(QsoCL,wavelengthsMatrixFiltered, reglin['cReglinResults'])
    #continuumMatrixFiltered = calcCw(QsoCL,wavelengthsMatrixFiltered, reglin['reglinResults'])

    #print(newSizes[q])
    #plot(wavelengthsMatrixFiltered[0:newSizes[q],q], continuumMatrixFiltered[0:newSizes[q],q],'black','wavelength','flux','filtered continuum of 12th quasar','file')


    chisqsFiltered = chisqR(QsoCL,
        spectrumsMatrixFiltered,
        continuumMatrixFiltered,
        errorsMatrixFiltered,
        newSizes)
    chisqsFiltered = reduceContinuumChisqs(QsoCL,chisqsFiltered, newSizes)
    

    cfunDcfun = calcCfunDcfun(QsoCL,wavelengthsMatrix, reglin['cReglinResults'], reglin['reglinResults'])
    #cfunDcfun = calcCfunDcfun(QsoCL,wavelengthsMatrix, cReglinResults, reglinResults)

    #plot(wavelengthsMatrix[0:newSizes[q],q], cfunDcfun['continuum'][0:newSizes[q],q],'black','wavelength','flux','filtered continuum of 12th quasar','file')

    continuumMatrixAltered = calcCw(QsoCL,wavelengthsMatrix, reglin['cReglinResults'])

    continuum_dict = {
        "dcontinuumsMatrix": cfunDcfun['dContinuum'],
        #"continuumsMatrix": cfunDcfun['continuum'],
        "continuumsMatrix": continuumMatrixAltered,
        "continuumChisqs": chisqsFiltered,
        "continuumReglin": reglin['reglinResults'],
        "reglin": reglin['cReglinResults']
        }
    return continuum_dict



#To compute elements for Gaussian fitting
def fitElementTest(QsoCL,spectrumsLinesMatrix, continuumsMatrix, wavelengthsMatrix, 
		   errorsMatrix, sizesVector, element):
 spectrumsLinesMatrixCopy = spectrumsLinesMatrix
 wavelengthsMatrixCopy = wavelengthsMatrix
 sizesCopy = sizesVector
 el_range = list(element['range'])
 element_range_list = []
 for i in range(len(el_range)):
  for j in range(len(el_range[0])):
   element_range_list.append(el_range[i][j])

 filteredMatrices = filterWithWavelengthWindows(QsoCL,spectrumsLinesMatrix, wavelengthsMatrix, 
					        continuumsMatrix, sizesVector,element_range_list)
 sizes = countIfNotInf(QsoCL,filteredMatrices['spectrumsMatrix'])
 maxSize = max(sizes)
 spectrumsLinesMatrix= copyIfNotInf(QsoCL,filteredMatrices['spectrumsMatrix'], maxSize)
 wavelengthsMatrix = copyIfNotInf(QsoCL,filteredMatrices['wavelengthsMatrix'], maxSize)
 continuumsMatrix = copyIfNotInf(QsoCL,filteredMatrices['errorsMatrix'], maxSize)
 el_fg = list(element['fitGuess'])
 element_fg_list = []
 for i in range(len(el_fg)):
  for j in range(len(el_fg[0])):
   element_fg_list.append(el_fg[i][j])
 fitGResults = element_fg_list * continuumsMatrix.shape[0]
 fitGResults = fitGaussian(QsoCL,spectrumsLinesMatrix, wavelengthsMatrix, sizes, fitGResults)
 gaussiansMatrix = calcGaussian(QsoCL,wavelengthsMatrix, fitGResults, sizes)
 gaussiansMatrix = divideR(QsoCL, gaussiansMatrix, continuumsMatrix)
 print("fitGResults",fitGResults.shape)
 ews = trapz(QsoCL, gaussiansMatrix, wavelengthsMatrix, sizes)
 gaussianChisqs = calcGaussianChisqs(QsoCL, wavelengthsMatrixCopy, spectrumsLinesMatrixCopy,
					errorsMatrix, fitGResults, sizesCopy)
 gaussianFWHMs = calcGaussianFWHM(QsoCL, fitGResults)
  
 print("fitparams",fitGResults.shape)
 print("chisqs",gaussianChisqs.shape)
 
 elDict = {"fitParams": fitGResults,
	    "ews" : ews,
	    "chisqs":gaussianChisqs,
	    "gaussianFWHMs": gaussianFWHMs
            }
 return elDict
 


def parameterization(QsoCL,qso,options):
    
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
       
    originalMatrices= {"spectrumsMatrix":specMat,"wavelengthsMatrix":wavMat,"errorsMatrix":errMat,"sizes":sizesVec}
    
    window = 50  # window of average

    specMat = centered_mavg(QsoCL,
        specMat,
        sizesVec,
        window)

    filteredMatrices = filterZeros(QsoCL,
        errMat,
        specMat,
        wavMat,
        sizesVec)

    newSizes = countIfNotInf(QsoCL,filteredMatrices['bMatrix'])
    maxSize = max(newSizes)
    # print(newSizes)

    spectrumsMatrix = filteredMatrices['aMatrix']
    wavelengthsMatrix = filteredMatrices['bMatrix']
    errorsMatrix = filteredMatrices['spectrumsMatrix']

    spectrumsMatrixCopy = spectrumsMatrix
    # print(spectrumsMatrixCopy)
    
    max_iter = 1
   
    for i in range(max_iter):
     continuumResults = continuumTest(QsoCL,
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
     print(sizes)
     q=11
     plt.plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], continuumsMatrix[0:sizes[q],q])
     plt.show()
     plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q],'black','wavelength','flux','12th quasar','file')
     plot(wavelengthsMatrix[0:sizes[q],q], continuumsMatrix[0:sizes[q],q],'black','wavelength','flux','final continuum of 12th quasar','file')

     feResults = feFitTest(QsoCL, spectrumsData, feTemplate, feWindows, fitParameters)
     sizes=sizesCOPY
     continuumsMatrix=continuumsMatrixCOPY
     wavelengthsMatrix=wavelengthsMatrixCOPY
     print(sizes)
     print(len(sizes))
     print(continuumsMatrix.shape)
     print(wavelengthsMatrix.shape)
     print(feResults['feTemplateMatrix'][0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], spectrumsMatrix[0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], continuumsMatrix[0:sizes[q],q])
     plt.plot(wavelengthsMatrix[0:sizes[q],q], feResults['feTemplateMatrix'][0:sizes[q],q])
     plt.show()
     if i < max_iter:
      spectrumsMatrix = minusMatrix(QsoCL,spectrumsMatrix, feResults['feTemplateMatrix'])
    
     spectrumsEmissionLines = minusMatrix(QsoCL,spectrumsMatrix, feResults['feTemplateMatrix'])
     spectrumsEmissionLines = minusMatrix(QsoCL,spectrumsEmissionLines, continuumResults['continuumsMatrix'])
    
     fitElementsResults = fitElementTest( QsoCL,spectrumsEmissionLines,continuumResults['continuumsMatrix'], 
					 wavelengthsMatrix, errorsMatrix, newSizes,spectralLines)
     #drawChosenPeaksComponents
     spectrumsMatrixORIG = originalMatrices['spectrumsMatrix']
     spectrumsMatrixNoIron = minusMatrix(QsoCL,spectrumsMatrix, feResults['feTemplateMatrix'])
     spectrumsMatrixEmissionLines = minusMatrix(QsoCL,spectrumsMatrixNoIron,continuumResults['continuumsMatrix'])
     

     
     gaussiansingle = lambda a,c,sig,x: a*np.exp(-.5*((x-c)/sig)**2)
     
     
     #drawChosenSpectrumWithPeaksRawData(q, spectrumsMatrixEmissionLines, wavelengthsMatrix, outputfit, getParams(quasars), sizesVector)
     
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
