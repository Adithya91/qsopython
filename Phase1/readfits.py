import math
import astropy.io
from astropy.io import fits
import numpy as np


'''
readfits = fits.open('/home/adithya/Desktop/sample.fits')
'''
def get_column(matrix, i):
    return [row[i] for row in matrix]
	
def get_key(val, items): 
    for key, value in items: 
         if val == value: 
             return key 
  
    return 0

def coordinatesToName(ra, dec, prefix):
  r = []
  d = []
  r.append( math.floor(ra / (360. / 24.)))
  r.append(math.floor(((ra / (360. / 24.)) % 1.) * 60.))
  r.append(((((ra / (360. / 24.)) % 1.) * 60.) % 1.) * 60.)
  d.append(math.floor(abs(dec)))
  d.append(math.floor((abs(dec) % 1.) * 60))
  d.append((((abs(dec) % 1.) * 60) % 1.) * 60.)
  sign = '+'
  if (dec < 0):
   sign = '-'
  return prefix+str(int(r[0]))+str(int(r[1]))+str(round(r[2], 2))+sign+str(int(d[0]))+str(int(d[1]))+str(round(d[2], 2))
	
def readSdssFitsFile(filename):	
 readfits = fits.open(filename)	
 O = readfits[2].data
 X = readfits[1].data
 Y = readfits[2].data
 wave =  10**np.asarray(get_column(X,1))
 flux = np.asarray(get_column(X,0))
 fluxErr = np.asarray(get_column(X,2))
 mask = np.asarray(get_column(X,3))
		
 Y_hdr = readfits[2].header
 if(get_key('Z',Y_hdr.items())!=0): 
  redshiftVal= get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('Z',Y_hdr.items()))+1][5:9])-1)[0]
 else:
  redshiftVal = None
 if(get_key('Z_ERR',Y_hdr.items())!=0): 
  redshiftErr = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('Z_ERR',Y_hdr.items()))+1][5:9])-1)[0]
 else:
  redshiftErr = None
 if(get_key('CLASS',Y_hdr.items())!=0): 
  objClas = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('CLASS',Y_hdr.items()))+1][5:9])-1)
 else:
  objClas = None
 if(get_key('SUBCLASS',Y_hdr.items())!=0):  
  objSClas = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('SUBCLASS',Y_hdr.items()))+1][5:9])-1)
 else:
  objSClas = None
 if(get_key('RA',Y_hdr.items())!=0):
  ra = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('RA',Y_hdr.items()))+1][5:9])-1)[0]
 else:
  ra = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('PLUG_RA',Y_hdr.items()))+1][5:9])-1)[0]
 if(get_key('RA',Y_hdr.items())!=0):
  dec = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('DEC',Y_hdr.items()))+1][5:9])-1)[0]
 else:
  dec = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('PLUG_DEC',Y_hdr.items()))+1][5:9])-1)[0]
 objName = coordinatesToName(ra,dec,'SDSS J')
 if(get_key('FIBERID',Y_hdr.items())!=0):
  fiber = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('FIBERID',Y_hdr.items()))+1][5:9])-1)
 else:
  fiber = None
 if(get_key('PLATE',Y_hdr.items())!=0):
  plate = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('PLATE',Y_hdr.items()))+1][5:9])-1)
 else:
  plate = None
 if(get_key('MJD',Y_hdr.items())!=0):
  mjd = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('MJD',Y_hdr.items()))+1][5:9])-1)
 else:
  mjd = None
 if(get_key('SN_MEDIAN',Y_hdr.items())!=0): 
  sn = get_column(Y,int(list(Y_hdr)[Y_hdr.index(get_key('SN_MEDIAN',Y_hdr.items()))+1][5:9])-1)
 else:
  sn = None
 O_hdr = readfits[2].header	
 if(get_key('BUNIT',Y_hdr.items())!=0):
  unit = get_column(O,int(list(O_hdr)[O_hdr.index(get_key('BUNIT',O_hdr.items()))+1][5:9])-1)
 else:
  unit = None
 X_hdr = readfits[1].header
 if(get_key('DATE-OBS',X_hdr.items())!=0):
  dateObs = list(X_hdr)[X_hdr.index(get_key('DATE-OBS',X_hdr.items()))+1]
 else:
  dateObs = None
 if(get_key('MAG',X_hdr.items())!=0):  
  mag = list(X_hdr)[X_hdr.index(get_key('MAG',X_hdr.items()))+1]
 else:
  mag = None
 wave = wave/(1+redshiftVal)
 output = {'wavelength' : wave, 'flux' : flux, 'error' : fluxErr, 'mask' : mask, 'unit' : unit,'mjd' : mjd, 'plate' : plate, 'fiber' : fiber, 'redshift' : redshiftVal, 'redshiftErr' : redshiftErr,'ra' : ra, 'dec' : dec, 'name' : objName, 'date' : dateObs, 'type' : objClas,'mag' : mag, 'sn' : sn}
 return output

if __name__ == '__main__':
 output = readSdssFitsFile('/home/adithya/Desktop/sample.fits')
 print (output)
