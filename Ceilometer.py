#%%
#from netCDF4 import Dataset  # use scipy instead
from scipy.io import netcdf #### <--- This is the library to import.
import matplotlib.pyplot as plt

import netCDF4 as nc
import numpy as np
import numpy.matlib
import math
import statistics
import xarray as xr
from linreg import *


# read multiple files (wildcard)
files = nc.MFDataset('./Cronyn/2019/09/24/20190924_YXU-Cronyn_CHM160155_*.nc') 
#filesL2 = nc.MFDataset('./Cronyn/L2/2017/07/30/L2_*.nc') 
zenith = 90 - files.variables['zenith'][:]
print(files.variables['time'])
density = files.variables['beta_raw'][:]
range = files.variables['range'][:]
altitude = files.variables['altitude'][:]
time = files.variables['time'][:]
 

#atten = filesL2.variables['beta_att_raw']


range = np.array(range)
time = np.array(time)
np.shape(time)
plotalt = np.matlib.repmat(range*np.sin(zenith)+altitude, len(time), 1)
#plt.plot(density,plotalt)

logdensity = np.log(density)

fig = plt.figure()
ax = fig.add_subplot(111)

dens = density.transpose()

ytime = np.linspace(0, 14304.546, 5740)

#for i in (time):

#NOTE I should make the time be the same dimensions as the altitude 


#ax.scatter(time, plotalt[:,1], c=logdensity[:,1], cmap = 'seismic')
#ax.scatter(time, y[0,:], c=logdensity/1024, cmap = 'RdBu')

#plt.show()


# %%
import math

def coadd(q,z,layer):

    l = math.floor(len(q) / layer) * layer
    q = q[1:l+1]
    z = z[1:l+1]
    # Reshape q and z so that the bins from each layer are in the
    # same column
    qc = np.reshape(q, (layer,int(l/layer)), order='F')
    zc = np.reshape(z, (layer,int(l/layer)), order='F')
    qc = (np.mean(qc,0))
    zc = (np.mean(zc,0))
    print(np.shape(qc))
    return [qc, zc]
#%%
overlap = np.loadtxt('data.txt',dtype = float)
print(overlap)


#%%
##----------------------
# FOLLOWING plots the correct profile of the log density graph
### NIGHT AVERAGE PLOT
a = int(len(density)/2)
d_night = dens[:,0:a] 
d_nightsum = np.mean(d_night,1)
for f in overlap:
  dens_overlap = d_nightsum * overlap
print(dens_overlap)

[x,y] = coadd(dens_overlap,range,10)
x = np.abs(x)
logprofile = np.log(x)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)

error = np.ones(5740)
#fit = linreg(dens_overlap,range, error)
#ax1.plot(x,fit[2])
#m = fit[0][0]
#b = fit[0][1]


#xmodel = np.poly1d(model)


w, b  = np.polyfit(y[50:90],x[50:90],deg=1)
line  = w * y[0:90] + b
print(w)
print(b)

#ax2.plot(line,y[0:90], label = 'Rayleigh fit, slope: -0.954 intercept: 17980.6')

ax1.semilogx(x, y, label = 'Corrected Data')
plt.legend()
plt.xlim(10**3.5, 10**7)
plt.ylim(0, 15000)
plt.xlabel('normalized range corrected signal')
plt.ylabel('Height (m)')
plt.title('Night Average Signal: 2019/09/24')
fig.show()
#%%


# %%
##DAY AVERAGE SIGNAL
a = int(len(density)/2)
print(a)
d_day = dens[:,a:a*2] 
d_daysum = np.mean(d_day,1)
for f in overlap:
  dens_overlap = d_daysum * overlap
print(dens_overlap)
[x,y] = coadd(dens_overlap,range,10)
#x = np.abs(x)
logprofile = np.log(x)

plt.semilogx(x, y)
plt.xlim(10**3.5, 10**7)
plt.ylim(0, 15000)
plt.xlabel('normalized range corrected signal')
plt.ylabel('Height (m)')
plt.title('Day Average Signal: 2019/09/04')
plt.show()

# %%
from scipy.integrate import quad
import scipy.integrate as scint

plotrange = y

## Re - initialize variables

# x is the array of power values 
def integrand(f,k):

  func = np.exp(f-Pref/k)
  return func

#integral = 

def extinction_eqn(k, alpha_ref, LR, integral):

  numer = LR * np.exp((height - Pref)/k)
  denom = (1/alpha_ref) + 2/k * LR * integral
  
  ans = numer/denom
  return ans


  
solnrange = plotrange[0:38]
print(solnrange)

for height in solnrange:

  integral = scint.solve_ivp(integrand, solnrange, np.array([alpha_ref]))


 
#integral = quad(integrand, 0, 4000)

#%%
def extinction(r_ref, alpha_ref, k,LR):
  """
  P - the power at the desired height
  Pref - the power at the reference height
  LR - the lidar ratio
  P0 - the power at the calculating height
  alpha_ref - the extinction coefficient at the chosen reference height
  
  """
  # Pref - the single value at the reference height
  # Then P needs to be an array from the P at 0 height to P at the reference height


  # Calculate the extinction coefficient at each height and at each interval of attenuated backscatter
  # This is not a model simulation i am working with real data

  #Find the index where the plot height array is equal to the reference height
  refindex = np.where(plotrange ==  r_ref)
  refindex = int(refindex[0])
  
  solnrange = plotrange[0:refindex]
  print(solnrange)
  #plotstep = np.linspace(0,r_ref, refindex)
  
  # Determine the power at the reference height to be used
  Pref = logprofile[refindex]
  # Initialize the array of backscatter power to calculate the coefficient over
  # This is the array to be integrated over 
  P = logprofile[0:int(refindex)]
  plotstep = np.array([P])
  #print(Pref)
  extinction_coeff = np.empty(refindex)
  integrand = np.empty(refindex)
  integral = np.empty(refindex)
  def integrand(f,k):

    func = np.exp(f-Pref/k)
    return func


  for height in plotstep:
    

    numer = LR * np.exp((height - Pref)/k)

    denom = (1/alpha_ref) + 2/k * LR * integral


    integral = scint.solve_ivp(integrand, solnrange, np.array([alpha_ref]))

  # This will onlt calculate the integral from my chosen single height to the reference height
  # If i wanted to do this along the whole profile of heights, I need to do this iteratively
    #print(integral)
    ans = numer/denom
    #print(ans)
    extinction_coeff = extinction_coeff + ans
    #break
    


  return ans


test = extinction(5791.702, 0.477, 1, 56)
plt.plot(test,y[0:38])

# %%
import xarray as xr

xr.open_mfdataset('/Users/hannanghori/Documents/university shit/4th year/thesis-python/Cronyn/L2/2017/07/30/L2_*.nc')

dataset =  xr.open_mfdataset('/Users/hannanghori/Documents/university shit/4th year/thesis-python/Cronyn/L2/2017/07/30/L2_*.nc', concat_dim="time", data_vars='minimal', coords='minimal', compat='override')

#%%



#%%

















# TEST - IN PROGRESS
def diff_lidar(s, R, k, alpha):
  """Returns the derivative of the logarithm of the normalized range corrected
  backscatter power
  inputs:
  s - the state vector that defines [dRdr, dalphadr]
  R - lidar ratio that is dependant on the aerosol
  k - without prior info, assume this to be 1
  alpha - extinction coefficient"""
  dVdr = 1/R*(dR/dr)


def extinction_mol(r, pg, Tg):
  """This equation calculates the value of the extinction coefficient at the reference height
  at which all contribution is only from the molecular components in the atmosphere
  inputs:
  r - the height at which to calculate the coefficient
  p - the pressure at the height 
  T - the temparture at the height"""

  lambdaa = 1064*10**-9
  N = 5.48*10**-4
  delta = 0.0273
  Ns = N * pg/Tg*np.exp(-r/h)
  # NOte the folowing is the pressure/temparture term scaled for the height
  # It takes the ground temp and pressure and scales it to the range at which the coeff is to be caluclated
  pT = pg/Tg * np.exp(-r/h)
  alpha = (8*np.pi**3/3)*(n**2/lambaa**4*N**2)*(6+3*delta/6-7*delta)*Ns*(T0/p0) * pT
