#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.matlib as matlib
import datetime
import matplotlib

from matplotlib import cm

import pandas as pd
import netCDF4 as nc

import xarray as xr

#---------------------CORRECT COADD-----------
# Iniatalize coadd function to be used below 
# To add in height and time
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
# OPEN FILES

# read multiple files (wildcard)
files = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/14/20200614_YXU-Cronyn_CHM160155_*.nc') 
files1 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/01/20170601_YXU-Cronyn_CHM160155_*.nc') 

#filesL2 = nc.MFDataset('./Cronyn/L2/2017/07/30/L2_*.nc') 
zenith = 90 - files.variables['zenith'][:]
print(files.variables['time'])
density = files.variables['beta_raw'][:]
density1 = files1.variables['beta_raw'][:]

range = files.variables['range'][:]
altitude = files.variables['altitude'][:]
time = files.variables['time'][:]
time1 = files1.variables['time'][:]

 
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

ytime = np.linspace(0, 14306.546, 5740)

#for i in (time):

#NOTE I should make the time be the same dimensions as the altitude 


#ax.scatter(time, plotalt[:,1], c=logdensity[:,1], cmap = 'seismic')
#ax.scatter(time, y[0,:], c=logdensity/1024, cmap = 'RdBu')

#plt.show()


# %%
import math

#------------------------
# The following is the python analog to the coadd function in matlab. This function is used to create the 
# day and night average signals

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

# Apply overlap by multiplying values in 1024 array file

overlap = np.loadtxt('data.txt',dtype = float)
print(overlap)


#%%
##----------------------
# FOLLOWING plots the correct profile of the log density graph
### NIGHT AVERAGE PLOT
a = int(len(density)/2)
d_night = density[:,0:a] 
d_nightsum = np.mean(d_night,1)
#for f in overlap:
#  dens_overlap = d_nightsum * overlap
#print(dens_overlap)

[x,y] = coadd(d_nightsum,range,10)
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
d_day = density[:,a:a*2] 
d_daysum = np.mean(d_day,1)
#for f in overlap:
#  dens_overlap = d_daysum * overlap
#print(dens_overlap)
[x,y] = coadd(dens_overlap,range,10)
#x = np.abs(x)
logprofile = np.log(x)

plt.semilogx(x, y)
plt.xlim(10**3.5, 10**7)
plt.ylim(0, 15000)
plt.xlabel('normalized range corrected signal')
plt.ylabel('Height (m)')
plt.title('Day Average Signal: 2019/09/06')
plt.show()

# %%
from scipy.integrate import quad
import scipy.integrate as scint

# - ----------------------START OF FURTHER ANALYSIS------------------
plotrange = y

## Re - initialize variables

# x is the array of power values 

# Following defines the integral to be evaluated in the expression for the extinction coefficient 


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

#--------------------------------------
# The following attempts to derive the extinction coefficient using integration methods

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

# Set any values less than 1 equal to 1

#beta_att_test = np.array(beta_att_test)

beta = np.where(density <= 0, 1, density) 



#levels = MaxNLocator(nbins=15).tick_values(1, np.max(np.log(beta_abs)))

cmap = plt.get_cmap('viridis')
#cmap = plt.get_cmap('BrBG')

#cmap = set_color('jet')

#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

betaT = np.transpose(beta)
fig, (ax0) = plt.subplots()
beta_abs = np.abs(beta)
im = plt.pcolormesh(range, time, np.log10(beta), cmap=cmap, vmax=None,vmin=None, shading='flat')
fig.colorbar(im)
plt.title('Normalized, Attenuated Backscatter Power')
plt.xlabel('Time UT [h]')
plt.ylabel('Altitude a.s.l. (m)')
#plt.clim(0, -2)
#plt.ylim(0, 3000)
plt.show()


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
#%%


#------------------OPEN FILES 2020, 2019 --------------

#2020:

#L0




files_L0_1 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/01/20200601_YXU-Cronyn_CHM160155_*.nc') 
files_L0_2 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/03/20200603_YXU-Cronyn_CHM160155_*.nc') 
files_L0_3 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/04/20200604_YXU-Cronyn_CHM160155_*.nc') 
files_L0_4 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/06/20200606_YXU-Cronyn_CHM160155_*.nc') 
files_L0_5 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/06/20200606_YXU-Cronyn_CHM160155_*.nc') 
files_L0_6 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/07/20200607_YXU-Cronyn_CHM160155_*.nc') 
files_L0_7 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/08/20200608_YXU-Cronyn_CHM160155_*.nc') 
files_L0_8 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/09/20200609_YXU-Cronyn_CHM160155_*.nc') 
files_L0_9 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/13/20200613_YXU-Cronyn_CHM160155_*.nc') 
files_L0_10 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/14/20200614_YXU-Cronyn_CHM160155_*.nc') 
files_L0_11 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/15/20200615_YXU-Cronyn_CHM160155_*.nc') 
files_L0_12 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/16/20200616_YXU-Cronyn_CHM160155_*.nc') 
files_L0_13 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/17/20200617_YXU-Cronyn_CHM160155_*.nc') 
files_L0_14 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/18/20200618_YXU-Cronyn_CHM160155_*.nc') 


files_L0_15 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/19/20200619_YXU-Cronyn_CHM160155_*.nc') 
files_L0_16 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/20/20200620_YXU-Cronyn_CHM160155_*.nc') 
files_L0_17 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/21/20200621_YXU-Cronyn_CHM160155_*.nc') 
files_L0_18 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/24/20200624_YXU-Cronyn_CHM160155_*.nc') 


files_L0_19 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/25/20200625_YXU-Cronyn_CHM160155_*.nc') 
files_L0_20 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/26/20200626_YXU-Cronyn_CHM160155_*.nc') 
files_L0_21 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/28/20200628_YXU-Cronyn_CHM160155_*.nc') 
files_L0_22 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/29/20200629_YXU-Cronyn_CHM160155_*.nc') 
files_L0_23 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/06/30/20200630_YXU-Cronyn_CHM160155_*.nc') 



files_L0_list = [files_L0_1, files_L0_2, files_L0_3, files_L0_4, files_L0_5, files_L0_6,files_L0_7, files_L0_8, 
            files_L0_9, files_L0_10, files_L0_11, files_L0_12, files_L0_13, files_L0_14, files_L0_15, files_L0_16,
            files_L0_17, files_L0_18, files_L0_19, files_L0_20, files_L0_21, files_L0_22, files_L0_23]

time_test = files_L0_1.variables['time']


#%%
#2019:

#L0 data


files19_L0_1 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/04/20190604_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_2 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/07/20190607_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_3 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/08/20190608_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_4 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/11/20190611_YXU-Cronyn_CHM160155_*.nc') 


files19_L0_5 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/12/20190612_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_6 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/14/20190614_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_7 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/17/20190617_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_8 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/18/20190618_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_9 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/19/20190619_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_10 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/21/20190621_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_11 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/22/20190622_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_12 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/23/20190623_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_13 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/24/20190624_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_14 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/25/20190625_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_15 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/26/20190626_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_16 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/27/20190627_YXU-Cronyn_CHM160155_*.nc') 


files19_L0_list = [files19_L0_1, files19_L0_2, files19_L0_3, files19_L0_4, files19_L0_5, files19_L0_6,files19_L0_7, files19_L0_8, 
            files19_L0_9, files19_L0_10, files19_L0_11, files19_L0_12, files19_L0_13, files19_L0_14, files19_L0_15, files19_L0_16]



#%%
#2018:


#L0

files18_L0_1 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/02/20180602_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_2 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/04/20180604_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_3 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/06/20180606_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_4 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/08/20180608_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_5 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/11/20180611_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_6 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/12/20180612_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_7 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/13/20180613_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_8 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/14/20180614_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_9 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/15/20180615_YXU-Cronyn_CHM160155_*.nc') 

files18_L0_10 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/17/20180617_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_11 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/21/20180621_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_12 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/25/20180625_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_13 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/26/20180626_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_14 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/28/20180628_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_15 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/29/20180629_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_16 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/06/30/20180630_YXU-Cronyn_CHM160155_*.nc') 


files18_L0_list = [files18_L0_1, files18_L0_2, files18_L0_3, files18_L0_4, files18_L0_5, files18_L0_6,files18_L0_7, files18_L0_8, 
            files18_L0_9, files18_L0_10, files18_L0_11, files18_L0_12, files18_L0_13, files18_L0_14, files18_L0_15, files18_L0_16]

#%%
#2017:

files17_L0_1 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/12/20170612_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_2 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/13/20170613_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_3 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/14/20170614_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_4 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/16/20170616_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_5 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/21/20170621_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_6 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/22/20170622_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_7 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/24/20170624_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_8 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/25/20170625_YXU-Cronyn_CHM160155_*.nc') 

files17_L0_9 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/26/20170626_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_10 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/06/28/20170628_YXU-Cronyn_CHM160155_*.nc') 

files17_L0_list = [files17_L0_1, files17_L0_2, files17_L0_3, files17_L0_4, files17_L0_5, files17_L0_6,files17_L0_7, files17_L0_8, files17_L0_9,  files17_L0_10]

#----------------INITIALIZE VARIABLES

#%%
#2020:

b1 = files_L0_1.variables['beta_raw'][:]
b2 = files_L0_2.variables['beta_raw'][:]
b3 = files_L0_3.variables['beta_raw'][:]
b4 = files_L0_4.variables['beta_raw'][:]
b5 = files_L0_5.variables['beta_raw'][:]
b6 = files_L0_6.variables['beta_raw'][:]
b7 = files_L0_7.variables['beta_raw'][:]
b8 = files_L0_8.variables['beta_raw'][:]
b9 = files_L0_9.variables['beta_raw'][:]

b10 = files_L0_10.variables['beta_raw'][:]
b11 = files_L0_11.variables['beta_raw'][:]
b12 = files_L0_12.variables['beta_raw'][:]
b13 = files_L0_13.variables['beta_raw'][:]
b14 = files_L0_14.variables['beta_raw'][:]
b15 = files_L0_15.variables['beta_raw'][:]
b16 = files_L0_16.variables['beta_raw'][:]
b17 = files_L0_17.variables['beta_raw'][:]
b18 = files_L0_18.variables['beta_raw'][:]
b19 = files_L0_19.variables['beta_raw'][:]
b20 = files_L0_20.variables['beta_raw'][:]
b21 = files_L0_21.variables['beta_raw'][:]
b22 = files_L0_22.variables['beta_raw'][:]
b23 = files_L0_23.variables['beta_raw'][:]

pbl1 = files_L0_1['pbl'][:]

pbl2 = files_L0_2['pbl'][:]
pbl3 = files_L0_3['pbl'][:]
pbl4 = files_L0_4['pbl'][:]
pbl5 = files_L0_5['pbl'][:]
pbl6 = files_L0_6['pbl'][:]
pbl7 = files_L0_7['pbl'][:]
pbl8 = files_L0_8['pbl'][:]
pbl1 = files_L0_9['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]
pbl1 = files_L0_1['pbl'][:]


time_L0 = files_L0_1.variables['time'][:]
range_L0 = files_L0_1.variables['range'][:]
#%%

for i in files_L0_list:
  i.close()

#%%
#2019:



c1 = files19_L0_1.variables['beta_raw'][:]
c2 = files19_L0_2.variables['beta_raw'][:]
c3 = files19_L0_3.variables['beta_raw'][:]
c4 = files19_L0_4.variables['beta_raw'][:]
c5 = files19_L0_5.variables['beta_raw'][:]
c6 = files19_L0_6.variables['beta_raw'][:]
c7 = files19_L0_7.variables['beta_raw'][:]
c8 = files19_L0_8.variables['beta_raw'][:]
c9 = files19_L0_9.variables['beta_raw'][:]

c10 = files19_L0_10.variables['beta_raw'][:]
c11 = files19_L0_11.variables['beta_raw'][:]
c12 = files19_L0_12.variables['beta_raw'][:]
c13 = files19_L0_13.variables['beta_raw'][:]
c14 = files19_L0_14.variables['beta_raw'][:]
c15 = files19_L0_15.variables['beta_raw'][:]
c16 = files19_L0_16.variables['beta_raw'][:]

pbl19_1 = files19_L0_1['pbl'][:]
pbl19_2 = files19_L0_2['pbl'][:]
pbl19_3 = files19_L0_3['pbl'][:]
pbl19_4 = files19_L0_4['pbl'][:]
pbl19_5 = files19_L0_5['pbl'][:]
pbl19_6 = files19_L0_6['pbl'][:]
pbl19_7 = files19_L0_7['pbl'][:]
pbl19_8 = files19_L0_8['pbl'][:]
pbl19_9 = files19_L0_9['pbl'][:]
pbl19_10 = files19_L0_10['pbl'][:]
pbl19_11= files19_L0_11['pbl'][:]
pbl19_12 = files19_L0_12['pbl'][:]
pbl19_13= files19_L0_13['pbl'][:]
pbl19_14= files19_L0_14['pbl'][:]
pbl19_15= files19_L0_15['pbl'][:]
pbl19_16= files19_L0_16['pbl'][:]


for i in files19_L0_list:
  i.close()

#%%
#2018

#L0 backscatter:

d1 = files18_L0_1.variables['beta_raw'][:]
d2 = files18_L0_2.variables['beta_raw'][:]
d3 = files18_L0_3.variables['beta_raw'][:]
d4 = files18_L0_4.variables['beta_raw'][:]
d5 = files18_L0_5.variables['beta_raw'][:]
d6 = files18_L0_6.variables['beta_raw'][:]
d7 = files18_L0_7.variables['beta_raw'][:]
d8 = files18_L0_8.variables['beta_raw'][:]
d9 = files18_L0_9.variables['beta_raw'][:]
d10 = files18_L0_10.variables['beta_raw'][:]
d11 = files18_L0_11.variables['beta_raw'][:]
d12 = files18_L0_12.variables['beta_raw'][:]
d13 = files18_L0_13.variables['beta_raw'][:]
d14 = files18_L0_14.variables['beta_raw'][:]
d15 = files18_L0_15.variables['beta_raw'][:]
d16 = files18_L0_16.variables['beta_raw'][:]


pbl18_1 = files18_L0_1['pbl'][:]

pbl18_2 = files18_L0_2['pbl'][:]
pbl18_3 = files18_L0_3['pbl'][:]
pbl18_4 = files18_L0_4['pbl'][:]
pbl18_5 = files18_L0_5['pbl'][:]
pbl18_6 = files18_L0_6['pbl'][:]
pbl18_7 = files18_L0_7['pbl'][:]
pbl18_8 = files18_L0_8['pbl'][:]
pbl18_8 = files18_L0_8['pbl'][:]
pbl18_9 = files18_L0_9['pbl'][:]
pbl18_10 = files18_L0_10['pbl'][:]

pbl18_11 = files18_L0_11['pbl'][:]
pbl18_12 = files18_L0_12['pbl'][:]
pbl18_13 = files18_L0_13['pbl'][:]
pbl18_14 = files18_L0_14['pbl'][:]
pbl18_15 = files18_L0_15['pbl'][:]
pbl18_16 = files18_L0_16['pbl'][:]



for i in files18_L0_list:
  i.close()


#%%

#2017

#L0 backscatter:

e1 = files17_L0_1.variables['beta_raw'][:]
e2 = files17_L0_2.variables['beta_raw'][:]
e3 = files17_L0_3.variables['beta_raw'][:]
e4 = files17_L0_4.variables['beta_raw'][:]
e5 = files17_L0_5.variables['beta_raw'][:]
e6 = files17_L0_6.variables['beta_raw'][:]
e7 = files17_L0_7.variables['beta_raw'][:]
e8 = files17_L0_8.variables['beta_raw'][:]
e9 = files17_L0_9.variables['beta_raw'][:]

e10 = files17_L0_10.variables['beta_raw'][:]


cbh17_2 = files17_L0_2['cbh'][:]
pbl17_1 = files17_L0_1['pbl'][:]

pbl17_2 = files17_L0_2['pbl'][:]
pbl17_3 = files17_L0_3['pbl'][:]
pbl17_4 = files17_L0_4['pbl'][:]
pbl17_5 = files17_L0_5['pbl'][:]
pbl17_6 = files17_L0_6['pbl'][:]
pbl17_7 = files17_L0_7['pbl'][:]
pbl17_8 = files17_L0_8['pbl'][:]
pbl17_9 = files17_L0_9['pbl'][:]
pbl17_10 = files17_L0_10['pbl'][:]

time_L0 = files17_L0_1.variables['time'][:]
range_L0 = files17_L0_1.variables['range'][:]

for i in files17_L0_list:
  i.close()
#%%
#-----------------------OVERLAP---------------------------------

#The overlap file is an array of 1024 values. 
# Resize this array to be 511 values in order to multiply with beta_att


overlap = np.loadtxt('data.txt',dtype = float)
print(overlap)

#Determine exact times at which there is clear sky rating, i.e. octa = 0



start = datetime.datetime(2000, 1, 1)
dt_array = np.array([start + datetime.timedelta(seconds=i) for i in range(5740)])


customdate_L0 = datetime.datetime(year=1906, month=1, day=1, hour=0, second=0)
realtime_L0 = [   datetime.timedelta(seconds=f) + customdate_L0 for f in ((time_L0))]

utc_L0 = np.array([f.strftime('%H:%M') for f in (realtime_L0)])

#total_cloud_L2 = files_L2.variables['cloud_amount'][:]

#for l in np.where(total_cloud_L2 == 0):
#  print(utc_L2[l])



#------------------PROCESS ALL 2020





#%%

##------

#5740/24 hours = 239

timespan1 = 21*239
timespan2 = 23*239
#np.where(utc_L2 == '21:00')
#timespan2 = np.where(utc_L2 == '23:55' )

#d_night_test1 = b1[:,248:250] 
d_night_test1 = b1[(timespan1):(timespan2),:] 

pbl20_1 = pbl1[:,0][timespan1:timespan2]
pbl20_time1 = realtime_L0[timespan1:timespan2]


d_nightsum_test1 = np.mean(d_night_test1,0)

noise1 = np.mean(d_nightsum_test1[950:1024])
print(noise1)
d_nightsum_test1 = d_nightsum_test1 + noise1

for f in overlap:
  dens_overlap_test1 = d_nightsum_test1 * overlap
#If overlap is needed:
[x1,y1] = coadd(dens_overlap_test1,range_L0,5)

##------
#timespan1 = 12*239
#np.where(utc_L2 == '00:00')
#timespan2 =  14*239 

timespan1 = 13*239
#np.where(utc_L2 == '00:00')
timespan2 =  14*239 
#np.where(utc_L2 == '09:00' )


d_night_test2 = b2[(timespan1):(timespan2),:] 
d_nightsum_test2 = np.mean(d_night_test2,0)

noise2 = np.mean(d_nightsum_test2[950:1024])
print(noise2)
d_nightsum_test2 = d_nightsum_test2 + noise2

for f in overlap:
  dens_overlap_test2 = d_nightsum_test2 * overlap
#If overlap is needed:
[x2,y2] = coadd(dens_overlap_test2,range_L0,5)

#-------

timespan1 = 7*239
#np.where(utc_L2 == '00:00')
timespan2 = 8*239
#np.where(utc_L2 == '03:00' )

d_night_test3 = b3[(timespan1):(timespan2),:] 
d_nightsum_test3 = np.mean(d_night_test3,0)

noise3 = np.mean(d_nightsum_test3[900:1024])
print(noise3)
d_nightsum_test3 = d_nightsum_test3 + noise3

for f in overlap:
  dens_overlap_test3 = d_nightsum_test3 * overlap
#If overlap is needed:
[x3,y3] = coadd(dens_overlap_test3,range_L0,5)

##------

timespan1 = 9* 239 
#np.where(utc_L2 == '22:00')
timespan2 = 12 *239
#np.where(utc_L2 == '23:55' )


d_night_test4 = b4[(timespan1):(timespan2),:] 
d_nightsum_test4 = np.mean(d_night_test4,0)

noise4 = np.mean(d_nightsum_test4[950:1024])
print(noise4)
d_nightsum_test4 = d_nightsum_test4 + noise4

for f in overlap:
  dens_overlap_test4 = d_nightsum_test4 * overlap
#If overlap is needed:
[x4,y4] = coadd(dens_overlap_test4,range_L0,5)

##------

timespan1 = 6*239
#np.where(utc_L2 == '00:00')
timespan2 = 12 * 239
#np.where(utc_L2 == '03:00' )


d_night_test5 = b5[(timespan1):(timespan2),:] 
d_nightsum_test5 = np.mean(d_night_test5,0)

noise5 = np.mean(d_nightsum_test5[950:1024])
print(noise5)
d_nightsum_test5 = d_nightsum_test5 - noise5

for f in overlap:
  dens_overlap_test5 = d_nightsum_test5 * overlap
#If overlap is needed:
[x5,y5] = coadd(dens_overlap_test5,range_L0,5)

#-------

timespan1 = 0*239 
#np.where(utc_L2 == '00:00')
timespan2 = 9*239
#np.where(utc_L2 == '09:00' )


d_night_test6 = b6[(timespan1):(timespan2),:] 
d_nightsum_test6 = np.mean(d_night_test6,0)

noise6 = np.mean(d_nightsum_test6[950:1024])
print(noise6)
d_nightsum_test6 = d_nightsum_test6 + noise6

for f in overlap:
  dens_overlap_test6 = d_nightsum_test6 * overlap
#If overlap is needed:
[x6,y6] = coadd(dens_overlap_test6,range_L0,5)

#--------

timespan1 = 19*239 
#np.where(utc_L2 == '22:30')
timespan2 = 24*239 
#np.where(utc_L2 == '22:55' )

d_night_test7 = b7[(timespan1):(timespan2),:] 
d_nightsum_test7 = np.mean(d_night_test7,0)

noise7 = np.mean(d_nightsum_test7[950:1024])
print(noise7)
d_nightsum_test7 = d_nightsum_test7 + noise7

for f in overlap:
  dens_overlap_test7 = d_nightsum_test7 * overlap
#If overlap is needed:
[x7,y7] = coadd(dens_overlap_test7,range_L0,5)

#-----

timespan1 = 15 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 18* 239 
#np.where(utc_L2 == '01:00' )

d_night_test8 = b8[(timespan1):(timespan2),:] 
d_nightsum_test8 = np.mean(d_night_test8,0)

noise8 = np.mean(d_nightsum_test8[950:1024])
print(noise8)
d_nightsum_test8 = d_nightsum_test8 + noise8

for f in overlap:
  dens_overlap_test8 = d_nightsum_test8 * overlap
#If overlap is needed:
[x8,y8] = coadd(dens_overlap_test8,range_L0,5)

#------

timespan1 = 22 * 239 
#np.where(utc_L2 == '09:00')
timespan2 = 24*239
#np.where(utc_L2 == '10:00' )

d_night_test9 = b9[(timespan1):(timespan2),:]  
d_nightsum_test9 = np.mean(d_night_test9,0)

noise9 = np.mean(d_nightsum_test9[950:1024])
print(noise9)
d_nightsum_test9 = d_nightsum_test9 + noise9

for f in overlap:
  dens_overlap_test9 = d_nightsum_test9 * overlap
#If overlap is needed:
[x9,y9] = coadd(dens_overlap_test9,range_L0,5)


#------

timespan1 = (18*239) 
#np.where(utc_L2 == '18:00')
timespan2 = (24* 239)  
#np.where(utc_L2 == '19:30' )

d_night_test10 = b10[(timespan1):(timespan2),:] 
d_nightsum_test10 = np.mean(d_night_test10,0)

noise10 = np.mean(d_nightsum_test10[950:1024])
print(noise10)
d_nightsum_test10 = d_nightsum_test10 - noise10

for f in overlap:
  dens_overlap_test10 = d_nightsum_test10 * overlap
#If overlap is needed:
[x10,y10] = coadd(dens_overlap_test10,range_L0,5)


#------

timespan1 = 0
#np.where(utc_L2 == '17:30')
timespan2 = 6*239
#np.where(utc_L2 == '18:30' )

d_night_test11 = b11[(timespan1):(timespan2),:] 
d_nightsum_test11 = np.mean(d_night_test11,0)

noise11 = np.mean(d_nightsum_test11[950:1024])
print(noise10)
d_nightsum_test11 = d_nightsum_test11 - noise11

for f in overlap:
  dens_overlap_test11 = d_nightsum_test11 * overlap
#If overlap is needed:
[x11,y11] = coadd(dens_overlap_test11,range_L0,5)

#------

timespan1 = 6*239
#np.where(utc_L2 == '09:00')
timespan2 = 9*239
#np.where(utc_L2 == '12:00' )

d_night_test12 = b12[(timespan1):(timespan2),:] 
d_nightsum_test12 = np.mean(d_night_test12,0)

noise12 = np.mean(d_nightsum_test12[950:1024])
print(noise10)
d_nightsum_test12 = d_nightsum_test12 + noise12

for f in overlap:
  dens_overlap_test12 = d_nightsum_test12 * overlap
#If overlap is needed:
[x12,y12] = coadd(dens_overlap_test12,range_L0,5)

#------

timespan1 = 0*239
#np.where(utc_L2 == '06:00')
timespan2 = 6*239
#np.where(utc_L2 == '08:30' )

d_night_test13 = b13[(timespan1):(timespan2),:] 
d_nightsum_test13 = np.mean(d_night_test13,0)

noise13 = np.mean(d_nightsum_test13[950:1024])
#print(noise13)
d_nightsum_test13 = d_nightsum_test13 + noise13

for f in overlap:
  dens_overlap_test13 = d_nightsum_test13 * overlap
#If overlap is needed:
[x13,y13] = coadd(dens_overlap_test13,range_L0,5)

#------

timespan1 = 0*239
#np.where(utc_L2 == '12:00')
timespan2 = 6*239
#np.where(utc_L2 == '13:00' )

d_night_test14 = b14[(timespan1):(timespan2),:] 
d_nightsum_test14 = np.mean(d_night_test14,0)

noise14 = np.mean(d_nightsum_test14[950:1024])
#print(noise10)
d_nightsum_test14 = d_nightsum_test14 + noise14

for f in overlap:
  dens_overlap_test14 = d_nightsum_test14 * overlap
#If overlap is needed:
[x14,y14] = coadd(dens_overlap_test14,range_L0,5)

#------

timespan1 = 2*239
#np.where(utc_L2 == '02:00')
timespan2 = 3*239
#np.where(utc_L2 == '06:00' )

d_night_test15 = b15[(timespan1):(timespan2),:] 
d_nightsum_test15 = np.mean(d_night_test15,0)

noise15 = np.mean(d_nightsum_test15[950:1024])
#print(noise10)
d_nightsum_test15 = d_nightsum_test15 + noise15

for f in overlap:
  dens_overlap_test15 = d_nightsum_test15 * overlap
#If overlap is needed:
[x15,y15] = coadd(dens_overlap_test15,range_L0,5)

#------

timespan1 = 9*239
#np.where(utc_L2 == '15:00')
timespan2 = 12*239
#np.where(utc_L2 == '16:00' )

d_night_test16 = b16[(timespan1):(timespan2),:] 
d_nightsum_test16 = np.mean(d_night_test16,0)

noise16 = np.mean(d_nightsum_test16[950:1024])
#print(noise10)
d_nightsum_test16 = d_nightsum_test16 + noise16

for f in overlap:
  dens_overlap_test16 = d_nightsum_test16 * overlap
#If overlap is needed:
[x16,y16] = coadd(dens_overlap_test16,range_L0,5)

#------

timespan1 = 9*239
#np.where(utc_L2 == '06:00')
timespan2 = 12*239
# np.where(utc_L2 == '06:30' )

d_night_test17 = b17[(timespan1):(timespan2),:] 
d_nightsum_test17 = np.mean(d_night_test17,0)

noise17 = np.mean(d_nightsum_test17[950:1024])
#print(noise10)
d_nightsum_test17 = d_nightsum_test17 + noise17

for f in overlap:
  dens_overlap_test17 = d_nightsum_test17 * overlap
#If overlap is needed:
[x17,y17] = coadd(dens_overlap_test17,range_L0,5)

#------June 24:

timespan1 = 9*239
 #np.where(utc_L2 == '18:00')
timespan2 = 12*239
 #np.where(utc_L2 == '17:00' )

d_night_test18 = b18[(timespan1):(timespan2),:] 
d_nightsum_test18 = np.mean(d_night_test18,0)

noise18 = np.mean(d_nightsum_test18[950:1024])
#print(noise10)
d_nightsum_test18 = d_nightsum_test18 + noise18

for f in overlap:
  dens_overlap_test18 = d_nightsum_test18 * overlap
#If overlap is needed:
[x18,y18] = coadd(dens_overlap_test18,range_L0,5)

#------

timespan1 = 3*239
 #np.where(utc_L2 == '18:00')
timespan2 = 9*239
 #np.where(utc_L2 == '17:00' )

d_night_test19 = b19[(timespan1):(timespan2),:] 
d_nightsum_test19 = np.mean(d_night_test19,0)

noise18 = np.mean(d_nightsum_test19[950:1024])
#print(noise10)
d_nightsum_test19 = d_nightsum_test19 + noise18

for f in overlap:
  dens_overlap_test19 = d_nightsum_test19 * overlap
#If overlap is needed:
[x19, y19] = coadd(dens_overlap_test19,range_L0,5)
#------

timespan1 = 0*239
 #np.where(utc_L2 == '18:00')
timespan2 = 1*239
 #np.where(utc_L2 == '17:00' )

d_night_test20 = b20[(timespan1):(timespan2),:] 
d_nightsum_test20 = np.mean(d_night_test20,0)

noise18 = np.mean(d_nightsum_test20[950:1024])
#print(noise10)
d_nightsum_test20 = d_nightsum_test20 + noise18

for f in overlap:
  dens_overlap_test20 = d_nightsum_test20 * overlap
#If overlap is needed:
[x20, y20] = coadd(dens_overlap_test20,range_L0,5)


#------

timespan1 = 6*239
 #np.where(utc_L2 == '18:00')
timespan2 = 9*239
 #np.where(utc_L2 == '17:00' )

d_night_test21 = b21[(timespan1):(timespan2),:] 
d_nightsum_test21 = np.mean(d_night_test21,0)

noise18 = np.mean(d_nightsum_test21[950:1024])
#print(noise10)
d_nightsum_test21 = d_nightsum_test21 + noise18

for f in overlap:
  dens_overlap_test21 = d_nightsum_test21 * overlap
#If overlap is needed:
[x21, y21] = coadd(dens_overlap_test21,range_L0,5)

#------

timespan1 = 0*239
 #np.where(utc_L2 == '18:00')
timespan2 = 6*239
 #np.where(utc_L2 == '17:00' )

d_night_test22 = b22[(timespan1):(timespan2),:] 
d_nightsum_test22 = np.mean(d_night_test22,0)

noise18 = np.mean(d_nightsum_test20[950:1024])
#print(noise10)
d_nightsum_test22 = d_nightsum_test22 + noise18

for f in overlap:
  dens_overlap_test22 = d_nightsum_test22 * overlap
#If overlap is needed:
[x22, y22] = coadd(dens_overlap_test22,range_L0,5)


#------

timespan1 = 6*239
 #np.where(utc_L2 == '18:00')
timespan2 = 7*239
 #np.where(utc_L2 == '17:00' )

d_night_test23 = b23[(timespan1):(timespan2),:] 
d_nightsum_test23 = np.mean(d_night_test23,0)

noise18 = np.mean(d_nightsum_test23[950:1024])
#print(noise10)
d_nightsum_test23 = d_nightsum_test23 + noise18

for f in overlap:
  dens_overlap_test23 = d_nightsum_test23 * overlap
#If overlap is needed:
[x23, y23] = coadd(dens_overlap_test23,range_L0,5)




fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)

ax1.semilogx(x18, y1, label='April 7, 2019')
#plt.semilogx(x1[0:70:], y1[0:70])
#ax2.scatter(x1, pbl20_1[0:204])
ax1.set_xlim([10,10000000])
ax1.set_ylim([0, 10000])


#x16 bump until 3000m
#x18 - spike 1000m



#%%

#-------PROCESS ALL 2019

#------

timespan1 = 0*239
#np.where(utc_L2 == '13:30')
timespan2 = 3*239
#np.where(utc_L2 == '14:30' )


d19_night_test1 = c1[(timespan1):(timespan2),:]  
d19_nightsum_test1 = np.mean(d19_night_test1,0)

noise11 = np.mean(d19_nightsum_test1[800:1024])
print(noise11)
d19_nightsum_test1 = d19_nightsum_test1 + noise11

for f in overlap:
  dens19_overlap_test1 = d19_nightsum_test1 * overlap
#If overlap is needed:
[xx1,yy1] = coadd(dens19_overlap_test1,range_L0,5)

#------

timespan1 = 21*239
#np.where(utc_L2 == '08:30')
timespan2 = 24*239
#np.where(utc_L2 == '10:30' )

d19_night_test2 = c2[(timespan1):(timespan2),:] 
d19_nightsum_test2 = np.mean(d19_night_test2,0)

noise22 = np.mean(d19_nightsum_test2[950:1024])
print(noise22)
d19_nightsum_test2 = d19_nightsum_test2 + noise22

for f in overlap:
  dens19_overlap_test2 = d19_nightsum_test2 * overlap
#If overlap is needed:
[xx2,yy2] = coadd(dens19_overlap_test2,range_L0,5)

#------

timespan1 = 0*239
#np.where(utc_L2 == '12:30')
timespan2 = 6*239
#np.where(utc_L2 == '14:30' )


d19_night_test3 = c3[(timespan1):(timespan2),:] 
d19_nightsum_test3 = np.mean(d19_night_test3,0)

noise33 = np.mean(d19_nightsum_test3[950:1024])
print(noise33)
d19_nightsum_test3 = d19_nightsum_test3 + noise33

for f in overlap:
  dens19_overlap_test3 = d19_nightsum_test3 * overlap
#If overlap is needed:
[xx3,yy3] = coadd(dens19_overlap_test3,range_L0,5)

#-----
timespan1 = 21* 239 
#np.where(utc_L2 == '07:00')
timespan2 = 24*239
#np.where(utc_L2 == '08:30' )


d19_night_test4 = c4[(timespan1):(timespan2),:] 
d19_nightsum_test4 = np.mean(d19_night_test4,0)

noise44 = np.mean(d19_nightsum_test4[950:1024])
print(noise44)
d19_nightsum_test4 = d19_nightsum_test4 + noise44

for f in overlap:
  dens19_overlap_test4 = d19_nightsum_test4 * overlap
#If overlap is needed:
[xx4,yy4] = coadd(dens19_overlap_test4,range_L0,5)

#--------

timespan1 = 0*239
#np.where(utc_L2 == '16:30')
timespan2 = 3*239 + 119
#np.where(utc_L2 == '21:30' )

d19_night_test5 = c5[(timespan1):(timespan2),:] 
d19_nightsum_test5 = np.mean(d19_night_test5,0)

noise55 = np.mean(d19_nightsum_test5[950:1024])
print(noise55)
d19_nightsum_test5 = d19_nightsum_test5 + noise55

for f in overlap:
  dens19_overlap_test5 = d19_nightsum_test5 * overlap
#If overlap is needed:
[xx5,yy5] = coadd(dens19_overlap_test5,range_L0,5)

#-------

timespan1 = 15*239
#np.where(utc_L2 == '01:20')
timespan2 = 16*239
# np.where(utc_L2 == '03:00' )


d19_night_test6 = c6[(timespan1):(timespan2),:]  
d19_nightsum_test6 = np.mean(d19_night_test6,0)

noise66 = np.mean(d19_nightsum_test6[960:1024])
print(noise66)
d19_nightsum_test6 = d19_nightsum_test6 + noise66

for f in overlap:
  dens19_overlap_test6 = d19_nightsum_test6 * overlap
#If overlap is needed:
[xx6,yy6] = coadd(dens19_overlap_test6,range_L0,5)

#--------

timespan1 = 15*239 
#np.where(utc_L2 == '23:00')
timespan2 = 16*239
#np.where(utc_L2 == '23:55' )

d19_night_test7 = c7[(timespan1):(timespan2),:] 
d19_nightsum_test7 = np.mean(d19_night_test7,0)

noise77 = np.mean(d19_nightsum_test7[960:1024])
print(noise77)
d19_nightsum_test7 = d19_nightsum_test7 + noise77

for f in overlap:
  dens19_overlap_test7 = d19_nightsum_test7 * overlap
#If overlap is needed:
[xx7,yy7] = coadd(dens19_overlap_test7,range_L0,5)

#------

timespan1 = 23 * 239 
#np.where(utc_L2 == '06:30')
timespan2 = 24 * 239 
#np.where(utc_L2 == '08:30' )

d19_night_test8 = c8[(timespan1):(timespan2),:] 
d19_nightsum_test8 = np.mean(d19_night_test8,0)

pbl19plot_8 = pbl19_8[:,0][timespan1:timespan2]

noise88 = np.mean(d19_nightsum_test8[950:1024])
print(noise88)
d19_nightsum_test8 = d19_nightsum_test8 + noise88

for f in overlap:
  dens19_overlap_test8 = d19_nightsum_test8 * overlap
#If overlap is needed:
[xx8,yy8] = coadd(dens19_overlap_test8,range_L0,5)


#-------

timespan1 = 0 * 239 
 #np.where(utc_L2 == '00:00')
timespan2 = 6*239 
#np.where(utc_L2 == '03:00' )

d19_night_test9 = c9[(timespan1):(timespan2),:] 
d19_nightsum_test9 = np.mean(d19_night_test9,0)
pbl19plot_9 = pbl19_9[:,0][timespan1:timespan2]

noise99 = np.mean(d19_nightsum_test9[950:1024])
print(noise66)
d19_nightsum_test9 = d19_nightsum_test9 + noise99

for f in overlap:
  dens19_overlap_test9 = d19_nightsum_test9 * overlap
#If overlap is needed:
[xx9,yy9] = coadd(dens19_overlap_test9,range_L0,5)

#-----

timespan1 = 21*239  
#np.where(utc_L2 == '21:00')
timespan2 =24*239  
#np.where(utc_L2 == '23:55' )

d19_night_test10 = c10[(timespan1):(timespan2),:] 
d19_nightsum_test10 = np.mean(d19_night_test10,0)

noise99 = np.mean(d19_nightsum_test10[960:1024])
d19_nightsum_test10 = d19_nightsum_test10 + noise99

for f in overlap:
  dens19_overlap_test10 = d19_nightsum_test10 * overlap
#If overlap is needed:
[xx10,yy10] = coadd(dens19_overlap_test10,range_L0,5)

#--------

timespan1 = 0 * 239  
# np.where(ut5_L2 == '00:00')
timespan2 = 3 *239  
#np.where(utc_L2 == '02:00' )

d19_night_test11 = c11[(timespan1):(timespan2),:] 
d19_nightsum_test11 = np.mean(d19_night_test11,0)

noise99 = np.mean(d19_nightsum_test11[960:1024])
d19_nightsum_test11 = d19_nightsum_test11 - noise99

for f in overlap:
  dens19_overlap_test11 = d19_nightsum_test11 * overlap
#If overlap is needed:
[xx11,yy11] = coadd(dens19_overlap_test11,range_L0,5)

#------

timespan1 = 6 * 239 
#np.where(utc_L2 == '15:30')
timespan2 = 9* 239 
#np.where(utc_L2 == '16:30' )

d19_night_test12 = c12[(timespan1):(timespan2),:] 
d19_nightsum_test12 = np.mean(d19_night_test12,0)

noise99 = np.mean(d19_nightsum_test12[950:1024])
d19_nightsum_test12 = d19_nightsum_test12 - noise99

for f in overlap:
  dens19_overlap_test12 = d19_nightsum_test12 * overlap
#If overlap is needed:
[xx12,yy12] = coadd(dens19_overlap_test12,range_L0,5)

#--------

timespan1 =1 * 239
#np.where(utc_L2 == '03:00')
timespan2 =3 *239 
# np.where(utc_L2 == '06:00' )

d19_night_test13 = c13[(timespan1):(timespan2),:] 
d19_nightsum_test13 = np.mean(d19_night_test13,0)

noise99 = np.mean(d19_nightsum_test13[950:1024])
d19_nightsum_test13 = d19_nightsum_test13 + noise99

for f in overlap:
  dens19_overlap_test13 = d19_nightsum_test13 * overlap
#If overlap is needed:
[xx13,yy13] = coadd(dens19_overlap_test13,range_L0,5)

#---------

timespan1 = 9 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 12 * 239 
#np.where(utc_L2 == '01:00' )

d19_night_test14 = c14[(timespan1):(timespan2),:] 
d19_nightsum_test14 = np.mean(d19_night_test14,0)

noise99 = np.mean(d19_nightsum_test14[950:1024])
d19_nightsum_test14 = d19_nightsum_test14 + noise99

for f in overlap:
  dens19_overlap_test14 = d19_nightsum_test14 * overlap
#If overlap is needed:
[xx14,yy14] = coadd(dens19_overlap_test14,range_L0,5)

#-------

timespan1 =  0 *239 
#np.where(utc_L2 == '09:00')
timespan2 = 3 *239
#np.where(utc_L2 == '12:00' )

d19_night_test15 = c15[(timespan1):(timespan2),:] 
d19_nightsum_test15 = np.mean(d19_night_test15,0)

pbl19plot_15 = pbl19_15[:,0][timespan1:timespan2]

noise99 = np.mean(d19_nightsum_test15[950:1024])
d19_nightsum_test15 = d19_nightsum_test15 - noise99

for f in overlap:
  dens19_overlap_test15 = d19_nightsum_test15 * overlap
#If overlap is needed:
[xx15,yy15] = coadd(dens19_overlap_test15,range_L0,5)

#-----
timespan1 = 21 *239 
#np.where(utc_L2 == '06:00')
timespan2 = 23 *239 
 #np.where(utc_L2 == '06:00' )

d19_night_test16 = c16[(timespan1):(timespan2),:] 
d19_nightsum_test16 = np.mean(d19_night_test16,0)
pbl19plot_16 = pbl19_16[:,0][timespan1:timespan2]

noise99 = np.mean(d19_nightsum_test16[950:1024])
d19_nightsum_test16 = d19_nightsum_test16 - noise99

for f in overlap:
  dens19_overlap_test16 = d19_nightsum_test16 * overlap
#If overlap is needed:
[xx16,yy16] = coadd(dens19_overlap_test16,range_L0,5)

#------
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3 = fig.add_subplot(111)
ax4 = fig.add_subplot(111)
ax5 = fig.add_subplot(111)

#ax1.semilogx(xx7, yy1, label='April 9, 2019')
ax3.plot(xx8, yy1, lw = 0.8, label='June 18, 2019')
ax4.plot(xx9, yy1,lw = 0.8, label='June 19, 2019')
#ax5.plot(xx15, yy1, lw = 0.8,label='June 26, 2019')
ax5.plot(xx16, yy1, lw = 0.8,label='June 27, 2019')
ax1.plot(comb_mean20T, y5, lw = 2,label='June 2020 average')

ax5.scatter(xx8, pbl19plot_8[0:204],marker='.',c='b', s=2,label='Aerosol layer - June 4')
ax5.scatter(xx9, pbl19plot_9[0:204],marker='.',c='y',s=2, label='Aerosol layer - June 11')
ax5.scatter(xx15, pbl19plot_15[0:204],marker='.',c='g',s=2, label='Aerosol layer - June 11')

ax5.scatter(xx16, pbl19plot_16[0:204],marker='.',c='r',s=2, label='Aerosol layer - June 11')

ax1.set_xlim([100,300000])
ax1.set_ylim([0, 7000])
ax1.legend(fontsize='small')
ax1.set_xlabel(' Range-Corrected Backscatter Power (a.u.)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Averaged, Range-Corrected Signal Power: June 2019')
fig.show()

#xx2, xx3, xx4,xx8 , xx9, xx16 bump up to 2000m SIGNIFICANT

#xx7 sspike 4000m
#xx15 spike 2000m - 3000m


#%%
#--------------------------------------------------
#Calculate all profiles for 2018:




#-------

timespan1 = (0 * 239)  
# np.where(utc_L2 == '10:30')
timespan2 = (3*239) 
#np.where(utc_L2 == '11:30')


d18_night_test1 = d1[(timespan1):(timespan2),:]
d18_nightsum_test1 = np.mean(d18_night_test1,0)

pbl18plot_1 = pbl18_1[:,0][timespan1:timespan2]
pbl18_time1 = realtime_L0[timespan1:timespan2]

noise11 = np.mean(d18_nightsum_test1[950:1024])
print(noise11)
d18_nightsum_test1 = d18_nightsum_test1 + noise11

for f in overlap:
  dens18_overlap_test1 = d18_nightsum_test1 * overlap
#If overlap is needed:
[xxx1,yyy1] = coadd(dens18_overlap_test1,range_L0,5)

#------

timespan1 = 0* 239 
#np.where(utc_L2 == '09:00')
timespan2 = 1 * 239 
#np.where(utc_L2 == '12:00')

d18_night_test2 = d2[(timespan1):(timespan2),:]
d18_nightsum_test2 = np.mean(d18_night_test2,0)

pbl18plot_2 = pbl18_2[:,0][timespan1:timespan2]
pbl18_time2 = realtime_L0[timespan1:timespan2]

noise11 = np.mean(d18_nightsum_test2[950:1024])
print(noise11)
d18_nightsum_test2 = d18_nightsum_test2 + noise11

for f in overlap:
  dens18_overlap_test2 = d18_nightsum_test2 * overlap
#If overlap is needed:
[xxx2,yyy2] = coadd(dens18_overlap_test2,range_L0,5)

#-------

timespan1 = (9* 239) 
#np.where(utc_L2 == '06:30')
timespan2 = (11* 239) 
#np.where(utc_L2 == '06:30')

d18_night_test3 = d3[(timespan1):(timespan2),:]
d18_nightsum_test3 = np.mean(d18_night_test3,0)

pbl18plot_3 = pbl18_3[:,0][timespan1:timespan2]
pbl18_time3 = realtime_L0[timespan1:timespan2]

noise11 = np.mean(d18_nightsum_test3[900:1024])
print(noise11)
d18_nightsum_test3 = d18_nightsum_test3 - noise11

for f in overlap:
  dens18_overlap_test3 = d18_nightsum_test3 * overlap
#If overlap is needed:
[xxx3,yyy3] = coadd(dens18_overlap_test3,range_L0,5)

#------

timespan1 = 21 * 239  
 #np.where(utc_L2 == '00:00')
timespan2 = (24*239 ) 
 #np.where(utc_L2 == '01:30')

d18_night_test4 = d4[(timespan1):(timespan2),:] 
d18_nightsum_test4 = np.mean(d18_night_test4,0)

pbl18plot_4 = pbl18_4[:,0][timespan1:timespan2]
pbl18_time4 = realtime_L0[timespan1:timespan2]

noise11 = np.mean(d18_nightsum_test4[950:1024])
print(noise11)
d18_nightsum_test4 = d18_nightsum_test4 + noise11

for f in overlap:
  dens18_overlap_test4 = d18_nightsum_test4 * overlap
#If overlap is needed:
[xxx4,yyy4] = coadd(dens18_overlap_test4,range_L0,5)

#------

timespan1 = 18 * 239
 #np.where(utc_L2 == '09:00')
timespan2 = 21* 239 
#np.where(utc_L2 == '11:30')

d18_night_test5 = d5[(timespan1):(timespan2),:]
d18_nightsum_test5 = np.mean(d18_night_test5,0)

pbl18plot_5 = pbl18_5[:,0][timespan1:timespan2]
pbl18_time5 = realtime_L0[timespan1:timespan2]

noise11 = np.mean(d18_nightsum_test5[950:1024])
print(noise11)
d18_nightsum_test5 = d18_nightsum_test5 + noise11

for f in overlap:
  dens18_overlap_test5 = d18_nightsum_test5 * overlap
#If overlap is needed:
[xxx5,yyy5] = coadd(dens18_overlap_test5,range_L0,5)

#-------

timespan1 =0 * 239 
#np.where(utc_L2 == '12:00')
timespan2 = 9 * 239 
#np.where(utc_L2 == '13:00')

d18_night_test6 = d6[(timespan1):(timespan2),:]
d18_nightsum_test6 = np.mean(d18_night_test6,0)

pbl18plot_6 = pbl18_6[:,0][timespan1:timespan2]
pbl18_time6 = realtime_L0[timespan1:timespan2]

noise11 = np.mean(d18_nightsum_test6[960:1024])
print(noise11)
d18_nightsum_test6 = d18_nightsum_test6 + noise11

for f in overlap:
  dens18_overlap_test6 = d18_nightsum_test6 * overlap
#If overlap is needed:
[xxx6,yyy6] = coadd(dens18_overlap_test6,range_L0,5)

#-----

timespan1 = (23 * 239) 
#np.where(utc_L2 == '22:30')
timespan2 = (24 * 239) 
#np.where(utc_L2 == '23:55')

d18_night_test7 = d7[(timespan1):(timespan2),:]
d18_nightsum_test7 = np.mean(d18_night_test7,0)

noise11 = np.mean(d18_nightsum_test7[950:1024])
print(noise11)
d18_nightsum_test7 = d18_nightsum_test7 + noise11

for f in overlap:
  dens18_overlap_test7 = d18_nightsum_test7 * overlap
#If overlap is needed:
[xxx7,yyy7] = coadd(dens18_overlap_test7,range_L0,5)

#-----

timespan1 = 0 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 9* 239 
#np.where(utc_L2 == '02:00')

d18_night_test8 = d8[(timespan1):(timespan2),:]
d18_nightsum_test8 = np.mean(d18_night_test8,0)

noise11 = np.mean(d18_nightsum_test8[950:1024])
print(noise11)
d18_nightsum_test8 = d18_nightsum_test8 + noise11

for f in overlap:
  dens18_overlap_test8 = d18_nightsum_test8 * overlap
#If overlap is needed:
[xxx8,yyy8] = coadd(dens18_overlap_test8,range_L0,5)

#------

timespan1 = 15 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = (239*18)
#np.where(utc_L2 == '01:30')

d18_night_test9 = d9[(timespan1):(timespan2),:]
d18_nightsum_test9 = np.mean(d18_night_test9,0)

noise11 = np.mean(d18_nightsum_test9[950:1024])
print(noise11)
d18_nightsum_test9 = d18_nightsum_test9 + noise11

for f in overlap:
  dens18_overlap_test9 = d18_nightsum_test9 * overlap
#If overlap is needed:
[xxx9,yyy9] = coadd(dens18_overlap_test9,range_L0,5)

#------

timespan1 = 18*239
#np.where(utL2 == '02:00')
timespan2 =19*239
 #np.where(utc_L2 == '03:00')

d18_night_test10 = d10[(timespan1):(timespan2),:]
d18_nightsum_test10 = np.mean(d18_night_test10,0)

noise11 = np.mean(d18_nightsum_test10[950:1024])
print(noise11)
d18_nightsum_test10 = d18_nightsum_test10 - noise11

for f in overlap:
  dens18_overlap_test10 = d18_nightsum_test10 * overlap
#If overlap is needed:
[xxx10,yyy10] = coadd(dens18_overlap_test10,range_L0,5)

#------

timespan1 = 15 * 239 
#np.where(utc_L2 == '20:00')
timespan2 = 16 * 239 
#np.where(utc_L2 == '21:00')

d18_night_test11 = d11[(timespan1):(timespan2),:]
d18_nightsum_test11 = np.mean(d18_night_test11,0)

noise11 = np.mean(d18_nightsum_test11[950:1024])
print(noise11)
d18_nightsum_test11 = d18_nightsum_test11 + noise11

for f in overlap:
  dens18_overlap_test11 = d18_nightsum_test11 * overlap
#If overlap is needed:
[xxx11,yyy11] = coadd(dens18_overlap_test11,range_L0,5)

#------

timespan1 = 6 * 239 
# np.where(utc_L2 == '00:00')
timespan2 = 9  * 239
# np.where(utc_L2 == '12:00')

d18_night_test12 = d12[(timespan1):(timespan2),:]
d18_nightsum_test12 = np.mean(d18_night_test12,0)

noise11 = np.mean(d18_nightsum_test12[950:1024])
print(noise11)
d18_nightsum_test12 = d18_nightsum_test12 + noise11

for f in overlap:
  dens18_overlap_test12 = d18_nightsum_test12 * overlap
#If overlap is needed:
[xxx12,yyy12] = coadd(dens18_overlap_test12,range_L0,5)

#------

timespan1 = 0 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 2 * 239 
 #np.where(utc_L2 == '06:00')

d18_night_test13 = d13[(timespan1):(timespan2),:]
d18_nightsum_test13 = np.mean(d18_night_test13,0)

noise11 = np.mean(d18_nightsum_test13[950:1024])
print(noise11)
d18_nightsum_test13 = d18_nightsum_test13 - noise11

for f in overlap:
  dens18_overlap_test13 = d18_nightsum_test13 * overlap
#If overlap is needed:
[xxx13,yyy13] = coadd(dens18_overlap_test13,range_L0,5)

#------

timespan1 = 20 * 239 
 #np.where(utc_L2 == '00:00')
timespan2 =  21 * 239 
#np.where(utc_L2 == '03:00')

d18_night_test14 = d14[(timespan1):(timespan2),:]
d18_nightsum_test14 = np.mean(d18_night_test14,0)

noise11 = np.mean(d18_nightsum_test14[333:511])
print(noise11)
d18_nightsum_test14 = d18_nightsum_test14 + noise11

for f in overlap:
  dens18_overlap_test14 = d18_nightsum_test14 * overlap
#If overlap is needed:
[xxx14,yyy14] = coadd(dens18_overlap_test14,range_L0,5)

#--------

timespan1 = 23  * 239 
#np.where(utc_L2 == '06:00')
timespan2 = 24 * 239 
#np.where(utc_L2 == '12:00')

d18_night_test15 = d15[(timespan1):(timespan2),:]
d18_nightsum_test15 = np.mean(d18_night_test15,0)

noise11 = np.mean(d18_nightsum_test15[950:1024])
print(noise11)
d18_nightsum_test15 = d18_nightsum_test15 + noise11

for f in overlap:
  dens18_overlap_test15 = d18_nightsum_test15 * overlap
#If overlap is needed:
[xxx15,yyy15] = coadd(dens18_overlap_test15,range_L0,5)

#------

#--------

timespan1 = 0  * 239 
#np.where(utc_L2 == '06:00')
timespan2 =8 * 239 
#np.where(utc_L2 == '12:00')

d18_night_test16 = d16[(timespan1):(timespan2),:]
d18_nightsum_test16 = np.mean(d18_night_test16,0)

noise11 = np.mean(d18_nightsum_test16[950:1024])
print(noise11)
d18_nightsum_test16 = d18_nightsum_test16 + noise11

for f in overlap:
  dens18_overlap_test16 = d18_nightsum_test16 * overlap
#If overlap is needed:
[xxx16,yyy16] = coadd(dens18_overlap_test16,range_L0,5)

#------




#To quality check individual averaged plots: 

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3 = fig.add_subplot(111)
ax4 = fig.add_subplot(111)
ax5 = fig.add_subplot(111)
ax6 = fig.add_subplot(111)
ax7 = fig.add_subplot(111)

#ax3.semilogx(xxx19, yy1, label='April 12, 2019')



#ax1.semilogx(xxx11, yyy1, label='May 2, 2018')
ax2.plot(xxx2, yyy1,lw=0.8, label='June 4, 2018')
ax3.plot(xxx5, yyy1, lw=0.8, label='June 11, 2018')
ax4.plot(xxx7, yyy1,lw=0.8,  label='June 13, 2019')
ax5.plot(xxx8, yyy1, lw=0.8, label='June 14, 2018')
ax4.plot(comb_mean20T, y1, lw=2, label='June 2020 average')

ax5.scatter(xxx2, pbl18plot_2[0:204],marker='.',c='b', s=2,label='Aerosol layer - June 4')
ax5.scatter(xxx5, pbl18plot_5[0:204],marker='.',c='g',s=2, label='Aerosol layer - June 11')
#ax5.scatter(xxx7, pbl18plot_7[0:204],marker='.',c='y', label='Aerosol layer - May 7')
ax5.scatter(xxx8, pbl18plot_8[0:204],marker='.',c='y', s=2,label='Aerosol layer - June 14')

#ax6.semilogx(xxx9, yy1, label='')
#ax7.semilogx(xxx16, yy1, label='April 27, 2018')

ax1.set_xlim([1000,300000])
ax1.set_ylim([0, 8000])
ax1.legend(fontsize='small')
ax1.set_xlabel(' Range-Corrected Backscatter Power (a.u.)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Averaged, Range-Corrected Signal Power: June 2018')

#xxx2, xxx5, xxx7, xxx9 higher below 2000m
#ecxclude xxx3, exclyde xxx10, maybe exclude xxx11
#xxx15, xxx16 1000m spike 

#%%
#-----------------------------------------------------------
#Calculate profiles for 2017: 


timespan1 = 0 * 239  
#np.where(utc_L2 == '00:00')
timespan2 = 3* 239   
#np.where(utc_L2 == '06:00')

d17_night_test1 = e1[(timespan1):(timespan2),:]
d17_nightsum_test1 = np.mean(d17_night_test1,0)

pbl17plot_1 = pbl17_1[:,0][timespan1:timespan2]

noise11 = np.mean(d17_nightsum_test1[950:1024])
print(noise11)
d18_nightsum_test1 = d17_nightsum_test1 + noise11

for f in overlap:
  dens17_overlap_test1 = d17_nightsum_test1 * overlap
#If overlap is needed:
[xxxx1,yyyy1] = coadd(dens17_overlap_test1,range_L0,5)

#--------

timespan1 = 22 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 24 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test2 = e2[(timespan1):(timespan2),:]
d17_nightsum_test2 = np.mean(d17_night_test2,0)

pbl17plot_2 = pbl17_2[:,0][timespan1:timespan2]

cbh17plot_2 = cbh17_2[:,0][timespan1:timespan2]
cbh17_time2 = realtime_L0[timespan1:timespan2]

noise11 = np.mean(d17_nightsum_test2[950:1024])
print(noise11)
d18_nightsum_test2 = d17_nightsum_test1 + noise11

for f in overlap:
  dens17_overlap_test2 = d17_nightsum_test2 * overlap
#If overlap is needed:
[xxxx2,yyyy2] = coadd(dens17_overlap_test2,range_L0,5)

#--------



timespan1 = 0 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 1 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test3 = e3[(timespan1):(timespan2),:]
d17_nightsum_test3 = np.mean(d17_night_test3,0)

pbl17plot_3 = pbl17_3[:,0][timespan1:timespan2]

noise11 = np.mean(d17_nightsum_test3[950:1024])
print(noise11)
d18_nightsum_test3 = d17_nightsum_test3 + noise11

for f in overlap:
  dens17_overlap_test3 = d17_nightsum_test3 * overlap
#If overlap is needed:
[xxxx3,yyyy3] = coadd(dens17_overlap_test3,range_L0,5)

#--------
timespan1 = 15 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 16 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test4 = e4[(timespan1):(timespan2),:]
d17_nightsum_test4 = np.mean(d17_night_test4,0)

pbl17plot_4 = pbl17_4[:,0][timespan1:timespan2]

noise11 = np.mean(d17_nightsum_test4[500:1024])
print(noise11)
d18_nightsum_test4 = d17_nightsum_test4 - noise11

for f in overlap:
  dens17_overlap_test4 = d17_nightsum_test4 * overlap
#If overlap is needed:
[xxxx4,yyyy4] = coadd(dens17_overlap_test4,range_L0,5)

#--------

timespan1 = 5* 239 
#np.where(utc_L2 == '00:00')
timespan2 = 6* 239 
#np.where(utc_L2 == '06:00')

d17_night_test5 = e5[(timespan1):(timespan2),:]
d17_nightsum_test5 = np.mean(d17_night_test5,0)

pbl17plot_5 = pbl17_5[:,0][timespan1:timespan2]

noise11 = np.mean(d17_nightsum_test5[950:1024])
print(noise11)
d18_nightsum_test5 = d17_nightsum_test5 + noise11

for f in overlap:
  dens17_overlap_test5 = d17_nightsum_test5 * overlap
#If overlap is needed:
[xxxx5,yyyy5] = coadd(dens17_overlap_test5,range_L0,5)

#--------
timespan1 = 0 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 4 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test6 = e6[(timespan1):(timespan2),:]
d17_nightsum_test6 = np.mean(d17_night_test6,0)

pbl17plot_6 = pbl17_6[:,0][timespan1:timespan2]

noise11 = np.mean(d17_nightsum_test6[950:1024])
print(noise11)
d18_nightsum_test6 = d17_nightsum_test6 + noise11

for f in overlap:
  dens17_overlap_test6 = d17_nightsum_test6 * overlap
#If overlap is needed:
[xxxx6,yyyy6] = coadd(dens17_overlap_test6,range_L0,5)

#--------
timespan1 =   13 * 239  
#np.where(utc_L2 == '00:00')
timespan2 = 15 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test7 = e7[(timespan1):(timespan2),:]
d17_nightsum_test7 = np.mean(d17_night_test7,0)

pbl17plot_7 = pbl17_7[:,0][timespan1:timespan2]

noise11 = np.mean(d17_nightsum_test7[950:1024])
print(noise11)
d18_nightsum_test7 = d17_nightsum_test7 + noise11

for f in overlap:
  dens17_overlap_test7 = d17_nightsum_test7 * overlap
#If overlap is needed:
[xxxx7,yyyy7] = coadd(dens17_overlap_test7,range_L0,5)

#--------

timespan1 = 1 * 239   
#np.where(utc_L2 == '00:00')
timespan2 =  3 *239 
#np.where(utc_L2 == '06:00')

d17_night_test8 = e8[(timespan1):(timespan2),:]
d17_nightsum_test8 = np.mean(d17_night_test8,0)

pbl17plot_8 = pbl17_8[:,0][timespan1:timespan2]

noise11 = np.mean(d17_nightsum_test8[950:1024])
print(noise11)
d18_nightsum_test8 = d17_nightsum_test8 + noise11

for f in overlap:
  dens17_overlap_test8 = d17_nightsum_test8 * overlap
#If overlap is needed:
[xxxx8,yyyy8] = coadd(dens17_overlap_test8,range_L0,5)

#--------

#--------

timespan1 = 13* 239   
#np.where(utc_L2 == '00:00')
timespan2 =  14*239 
#np.where(utc_L2 == '06:00')

d17_night_test9= e9[(timespan1):(timespan2),:]
d17_nightsum_test9 = np.mean(d17_night_test9,0)

pbl17plot_9 = pbl17_9[:,0][timespan1:timespan2]

noise11 = np.mean(d17_nightsum_test9[950:1024])
print(noise11)
d18_nightsum_test9 = d17_nightsum_test9 + noise11

for f in overlap:
  dens17_overlap_test9= d17_nightsum_test9 * overlap
#If overlap is needed:
[xxxx9,yyyy9] = coadd(dens17_overlap_test9,range_L0,5)

#--------

#--------

timespan1 = 12 * 239   
#np.where(utc_L2 == '00:00')
timespan2 =  13 *239 
#np.where(utc_L2 == '06:00')

d17_night_test10 = e10[(timespan1):(timespan2),:]
d17_nightsum_test10= np.mean(d17_night_test10,0)

pbl17plot_10 = pbl17_10[:,0][timespan1:timespan2]

noise11 = np.mean(d17_nightsum_test10[950:1024])
print(noise11)
d18_nightsum_test10 = d17_nightsum_test10 + noise11

for f in overlap:
  dens17_overlap_test10 = d17_nightsum_test10 * overlap
#If overlap is needed:
[xxxx10,yyyy10] = coadd(dens17_overlap_test10,range_L0,5)

#--------
#To quality check individual averaged plots: 


fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3 = fig.add_subplot(111)
ax4 = fig.add_subplot(111)
ax5 = fig.add_subplot(111)
ax6 = fig.add_subplot(111)
ax7 = fig.add_subplot(111)

#ax3.semilogx(xxx19, yy1, label='April 12, 2019')



#ax1.semilogx(xxxx8, yy1, label='April 17, 2017')
ax2.semilogx(xxxx8, yyyy1)
#ax1.scatter(xxxx5, pbl17plot_5[0:204], marker='+',c='r', label='Aerosol Layer height in PBL')
#ax3.scatter(xxxx3, cbh17plot_5[0:204], marker='*',c='g', label='Cloud Base Height')

#ax3.semilogx(xxxx11, yy1, label='April 24, 2017')
#ax4.semilogx(xxx8, yy1, label='April 23, 2019')
#ax5.semilogx(xxxx13, yy1, label='April 28, 2017')
#ax6.semilogx(xxx9, yy1, label='')
#ax7.semilogx(xxx16, yy1, label='April 27, 2018')

ax1.set_xlim([1000,1000000])
ax1.set_ylim([0, 10000])
ax1.legend()
ax1.set_xlabel('Logarithm of Range-Corrected Backscatter Power (a.u.)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Averaged, Range-Corrected Signal Power: May 9, 2017 ')

#xxxx4 spike 2000m


# %%



#--------------MAKE AVERAGE PLOT OF ALL MONTHS -----------------


fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3 = fig.add_subplot(111)
ax4 = fig.add_subplot(111)

combx = np.array([ [x1],  [x3], [x5], [x6], [x7],  [x9], [x12], [x13], [x14],  [x16], [x17], [x19], [x20], [x21], [x22], [x23]])
combx19 = np.array([ [xx1], [xx2],  [xx4], [xx6],[xx8], [xx9], [xx10],[xx11],[xx12],[xx13],[xx14], [xx15], [xx16]])
combx18 = np.array([ [xxx1], [xxx2], [xxx4], [xxx6], [xxx7], [xxx8], [xxx9], [xxx10],[xxx12],[xxx13], [xxx16]])
combx17 = np.array([ [xxxx1],[xxxx2] , [xxxx3], [xxxx5], [xxxx6],[xxxx7],[xxxx8]])


comb_mean20 = np.mean(combx, axis=0)
comb_mean19 = np.mean(combx19, axis=(0))
comb_mean18 = np.mean(combx18, axis=(0))
comb_mean17 = np.mean(combx17, axis=(0))


comb_mean20T = np.transpose(comb_mean20)
comb_mean19T = np.transpose(comb_mean19)
comb_mean18T = np.transpose(comb_mean18)
comb_mean17T = np.transpose(comb_mean17)

#ax1.semilogx(comb_mean20T, y5, label='June 2020 average')
#ax2.semilogx(comb_mean19T, y5, label='June 2019 average')
#ax3.semilogx(comb_mean18T, y5, label='June 2018 average')
#ax4.semilogx(comb_mean17T, y5, label='June 2017 average')

ax1.plot(comb_mean20T, y5, label='June 2020 average')
ax2.plot(comb_mean19T, y5, label='June 2019 average')
ax3.plot(comb_mean18T, y5, label='June 2018 average')
ax4.plot(comb_mean17T, y5, label='June 2017 average')
#ax1.semilogx(comb_mean20T[0:102:], y1, label='April 2020 average')
#ax2.semilogx(comb_mean19T[0:102:], y1, label='April 2019 average')
#ax3.semilogx(comb_mean18T[0:102:], y1, label='April 2018 average')

ax1.set_xlim([10000,300000])
ax1.set_ylim([0, 7000])
ax1.legend()
ax1.set_xlabel('Range-Corrected Backscatter Power (a.u.)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Averaged, Range-Corrected Signal Power over 1 month: June')

plt.show()
# %%
