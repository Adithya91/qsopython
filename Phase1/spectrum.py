import pyopencl as cl
import numpy as np
ASTRO_OBJ_SPEC_SIZE = 4096


#To filter infinity values from spectrum
def filterInfs(QsoCL,spectrumsMatrix_, aMatrix_, bMatrix_, sizesVector_):
 queue = cl.CommandQueue(QsoCL.ctx)
 
 h = spectrumsMatrix_.shape[0]  # spectrumsSize
 w = spectrumsMatrix_.shape[1]  # spectrumsNumber
 spec = spectrumsMatrix_
 a = aMatrix_
 b = bMatrix_

 spec_g = QsoCL.makeBuffer(spectrumsMatrix_)
 a_g =QsoCL.makeBuffer(aMatrix_)
 b_g = QsoCL.makeBuffer(bMatrix_)
 sizes_g = QsoCL.makeBuffer(sizesVector_)
 
 _knl = QsoCL.buildKernel('spectrums_kernels.cl').filterInfs
 
 _knl(queue, (w,ASTRO_OBJ_SPEC_SIZE), (1,QsoCL.maxWorkGroupSize), spec_g, a_g, b_g, sizes_g)
 cl.enqueue_copy(queue, spec, spec_g)
 cl.enqueue_copy(queue, a, a_g)
 cl.enqueue_copy(queue, b, b_g)
 out = {
    "spectrumsMatrix":spec,
    "aMatrix":a,
    "bMatrix":b
     }
 return out

#To filter zeros from the input matrices

def filterZeros(QsoCL,inpMat, MatA, MatB, sizes):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber

    spec = np.zeros((w, h), dtype=np.float64)
    spec = np.transpose(np.asarray(spec, dtype=np.float64, order='F'))
    a = np.zeros((w, h), dtype=np.float64)
    a = np.transpose(np.asarray(a, dtype=np.float64, order='F'))
    b = np.zeros((w, h), dtype=np.float64)
    b = np.transpose(np.asarray(b, dtype=np.float64, order='F'))

    inp_g = QsoCL.readBuffer(inpMat)
    a_g = QsoCL.readBuffer(MatA)
    b_g = QsoCL.readBuffer(MatB)
    bufferSizes = QsoCL.readBuffer(sizes)

    _knl = QsoCL.buildKernel('spectrums_kernels.cl').filterZeros

    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE),
         (1, QsoCL.maxWorkGroupSize), inp_g, a_g, b_g, bufferSizes)
    cl.enqueue_copy(queue, spec, inp_g)
    cl.enqueue_copy(queue, a, a_g)
    cl.enqueue_copy(queue, b, b_g)
    out = {
    "spectrumsMatrix":spec,
    "aMatrix":a,
    "bMatrix":b
     }
    return out

#To filter the spectrum using a window
def filterWithWavelengthWindows(QsoCL,
        spectrumsMatrix,
        wavelengthsMatrix,
        errorsMatrix,
        sizes,
        windows):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    
    h = spectrumsMatrix.shape[0]  # spectrumsSize
    w = spectrumsMatrix.shape[1]  # spectrumsNumber
    
    windowsVector = np.asarray(windows)
    winSize = len(windowsVector)

    wavMat = np.zeros((w, h), dtype=np.float64)
    wavMat = np.transpose(np.asarray(wavMat, dtype=np.float64, order='F'))
    specMat = np.zeros((w, h), dtype=np.float64)
    specMat = np.transpose(np.asarray(specMat, dtype=np.float64, order='F'))
    errMat = np.zeros((w, h), dtype=np.float64)
    errMat = np.transpose(np.asarray(errMat, dtype=np.float64, order='F'))

    sMat = QsoCL.readBuffer(spectrumsMatrix)
    wMat = QsoCL.readBuffer(wavelengthsMatrix)
    eMat = QsoCL.readBuffer(errorsMatrix)
    sizVec = QsoCL.readBuffer(sizes)
    winVec = QsoCL.readBuffer(windowsVector)

    _knl = QsoCL.buildKernel('spectrums_kernels.cl').filterWithWavelengthWindows

    _knl.set_scalar_arg_dtypes(
        [None, None, None, None, None, np.uint32])
    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE), (1, QsoCL.maxWorkGroupSize),
         wMat, sMat, eMat, sizVec, winVec, winSize)
    cl.enqueue_copy(queue, wavMat, wMat)
    cl.enqueue_copy(queue, specMat, sMat)
    cl.enqueue_copy(queue, errMat, eMat)
    out = {
    "wavelengthsMatrix":wavMat,
    "spectrumsMatrix":specMat,
    "errorsMatrix":errMat
     }
    return out

#To filter negative values

def filterNonpositive(QsoCL,specMat, wavMat, errMat, sizesVector):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    
    h = specMat.shape[0]  # spectrumsSize
    w = specMat.shape[1]  # spectrumsNumber

    wMat = np.zeros((w, h), dtype=np.float64)
    wMat = np.transpose(np.asarray(wMat, dtype=np.float64, order='F'))
    sMat = np.zeros((w, h), dtype=np.float64)
    sMat = np.transpose(np.asarray(sMat, dtype=np.float64, order='F'))
    eMat = np.zeros((w, h), dtype=np.float64)
    eMat = np.transpose(np.asarray(eMat, dtype=np.float64, order='F'))

    s_g = QsoCL.readBuffer(specMat)
    w_g = QsoCL.readBuffer(wavMat)
    e_g = QsoCL.readBuffer(errMat)
    sizVec = QsoCL.readBuffer(sizesVector)

    _knl = QsoCL.buildKernel('spectrums_kernels.cl').filterNonpositive

    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE),
         (1, QsoCL.maxWorkGroupSize), s_g, w_g, e_g, sizVec)
    cl.enqueue_copy(queue, wMat, w_g)
    cl.enqueue_copy(queue, sMat, s_g)
    cl.enqueue_copy(queue, eMat, e_g)
    out = {
    "wavelengthsMatrix":wMat,
    "spectrumsMatrix":sMat,
    "errorsMatrix":eMat
     }
    return out

#To aggregate the spectrums

def addSpectrum(QsoCL,spectrumsMatrix,wavelengthsMatrix,sizes,size,toAddWavelengths,
                toAddSpectrum,toAddSize):
 queue = cl.CommandQueue(QsoCL.ctx)
 
 output = spectrumsMatrix
 globalSize = QsoCL.calcGlobalSize(size)
 wav_g = QsoCL.readBuffer(wavelengthsMatrix)
 spec_g = QsoCL.readBuffer(spectrumsMatrix)
 sizes_g = QsoCL.readBuffer(sizes)
 toAddwav_g = QsoCL.readBuffer(toAddWavelengths)
 toAddsp_g = QsoCL.readBuffer(toAddSpectrum)
 output_g = QsoCL.writeBuffer(output)

 _knl = QsoCL.buildKernel('spectrums_kernels.cl').addSpectrum
 _knl.set_scalar_arg_dtypes(
        [None, None, None, np.uint32, None, None,np.uint32,None])
 _knl(queue, (globalSize,),
         (QsoCL.maxWorkGroupSize,), wav_g, spec_g, sizes_g, size,toAddwav_g,toAddsp_g ,toAddSize,output_g)
 cl.enqueue_copy(queue, output, output_g)
 return output
