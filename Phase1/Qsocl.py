#!/usr/bin/env python


import pyopencl as cl
import os

#Class to initialize buffer objects and build kernel functions


class Qsocl:
  def __init__(self):
    print("Constructor Initialized")
    # Context
    self.ctx = cl.create_some_context()

    # Platform and device information
    self.platforms = cl.get_platforms()
    self.devices = self.platforms[0].get_devices(cl.device_type.GPU)

    # group size
    self.maxWorkGroupSize = self.devices[0].max_work_group_size    
       
    # kernel files location
    self.kernelpath = '/content/kernel/'
    
  def buildKernel(self, kernelfile):
    f = open(os.path.join(self.kernelpath, kernelfile), 'r')
    fstr = ''.join(f.readlines())
    clFunc = cl.Program(self.ctx, fstr).build()
    f.close()
    return clFunc

  def calcGlobalSize(self, dataSize):
    size = dataSize
    remainder = size % self.maxWorkGroupSize
    if (remainder != 0):
        size += self.maxWorkGroupSize - remainder
    if (size < dataSize):
        print("Error in calculating global_work_size.")
    return size
  #To create buffer object with read-only access
  def readBuffer(self,inputMatrix):
   mf = cl.mem_flags
   readBuff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inputMatrix)
   return readBuff
  #To create buffer object with write-only access
  def writeBuffer(self,outputMatrix):
   mf = cl.mem_flags
   writeBuff = cl.Buffer(self.ctx, mf.WRITE_ONLY, outputMatrix.nbytes)
   return writeBuff
  #To create buffer with read-write access
  def makeBuffer(self,inputMatrix):
   mf = cl.mem_flags
   readBuff = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=inputMatrix)
   return readBuff
  
