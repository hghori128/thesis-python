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
    print(l)
    q = q[1:l+1]
    z = z[1:l+1]
    # Reshape q and z so that the bins from each layer are in the
    # same column
    qc = np.reshape(q, (layer,int(l/layer)), order='F')
    zc = np.reshape(z, (layer,int(l/layer)), order='F')
    print(np.shape(q))
    qc = (np.sum(qc,0))
    zc = (np.median(zc,0))
    return [qc, zc]

#%%
# OPEN FILES

# read multiple files (wildcard)
files = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/01/14/20200114_YXU-Cronyn_CHM160155_*.nc') 
files1 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/14/20190114_YXU-Cronyn_CHM160155_*.nc') 

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

ytime = np.linspace(0, 14301.546, 5740)

#for i in (time):

#NOTE I should make the time be the same dimensions as the altitude 


#ax.scatter(time, plotalt[:,1], c=logdensity[:,1], cmap = 'seismic')
#ax.scatter(time, y[0,:], c=logdensity/1024, cmap = 'RdBu')

#plt.show()



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
plt.title('Day Average Signal: 2019/09/01')
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




files_L0_1 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/01/01/20200101_YXU-Cronyn_CHM160155_*.nc') 
files_L0_2 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/01/02/20200102_YXU-Cronyn_CHM160155_*.nc') 
files_L0_3 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/01/06/20200106_YXU-Cronyn_CHM160155_*.nc') 
files_L0_4 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/01/07/20200107_YXU-Cronyn_CHM160155_*.nc') 
files_L0_5 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/01/17/20200117_YXU-Cronyn_CHM160155_*.nc') 
files_L0_6 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/01/19/20200119_YXU-Cronyn_CHM160155_*.nc') 
files_L0_7 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/01/20/20200120_YXU-Cronyn_CHM160155_*.nc') 
files_L0_8 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2020/01/22/20200122_YXU-Cronyn_CHM160155_*.nc') 



files_L0_list = [files_L0_1, files_L0_2, files_L0_3, files_L0_4, files_L0_5, files_L0_6,files_L0_7, files_L0_8]
#            files_L0_9, files_L0_10, files_L0_11, files_L0_12, files_L0_13, files_L0_14, files_L0_15, files_L0_16,
#            files_L0_17, files_L0_18,files_L0_19, files_L0_20, files_L0_21, files_L0_22, files_L0_23]

time_test = files_L0_1.variables['time']


#%%
#2019:

#L0 data



files19_L0_1 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/04/20190104_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_2 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/05/20190105_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_3 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/13/20190113_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_4 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/14/20190114_YXU-Cronyn_CHM160155_*.nc') 


files19_L0_5 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/17/20190117_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_6 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/20/20190120_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_7 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/21/20190121_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_8 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/22/20190122_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_9 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/26/20190126_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_10 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/27/20190127_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_11 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/28/20190128_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_12 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/29/20190129_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_13 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/30/20190130_YXU-Cronyn_CHM160155_*.nc') 
files19_L0_14 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/01/31/20190131_YXU-Cronyn_CHM160155_*.nc') 


files19_L0_list = [files19_L0_1, files19_L0_2, files19_L0_3, files19_L0_4, files19_L0_5, files19_L0_6,files19_L0_7, files19_L0_8, 
            files19_L0_9, files19_L0_10, files19_L0_11, files19_L0_12, files19_L0_13, files19_L0_14]



#%%
#2018:


#L0

files18_L0_1 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/01/20180101_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_2 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/04/20180104_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_3 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/09/20180109_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_4 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/10/20180110_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_5 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/13/20180113_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_6 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/14/20180114_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_7 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/17/20180117_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_8 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/18/20180118_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_9 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/26/20180126_YXU-Cronyn_CHM160155_*.nc') 

files18_L0_10 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/28/20180128_YXU-Cronyn_CHM160155_*.nc') 
files18_L0_11 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2018/01/29/20180129_YXU-Cronyn_CHM160155_*.nc') 


files18_L0_list = [files18_L0_1, files18_L0_2, files18_L0_3, files18_L0_4, files18_L0_5, files18_L0_6,files18_L0_7, files18_L0_8, 
            files18_L0_9, files18_L0_10, files18_L0_11]

#%%
#2017:

files17_L0_1 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/02/20170102_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_2 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/05/20170105_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_3 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/08/20170108_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_4 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/09/20170109_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_5 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/14/20170114_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_6 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/15/20170115_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_7 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/16/20170116_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_8 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/17/20170117_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_9 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/18/20170118_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_10 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/23/20170123_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_11 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/24/20170124_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_12 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/26/20170126_YXU-Cronyn_CHM160155_*.nc') 
files17_L0_13 = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/01/28/20170128_YXU-Cronyn_CHM160155_*.nc') 

files17_L0_list = [files17_L0_1, files17_L0_2, files17_L0_3, files17_L0_4, files17_L0_5, files17_L0_6,files17_L0_7, files17_L0_8, 
            files17_L0_9, files17_L0_10, files17_L0_11, files17_L0_12, files17_L0_13]

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

time_L0 = files_L0_1.variables['time'][:]
range_L0 = files_L0_1.variables['range'][:]


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
pbl19_11 = files19_L0_11['pbl'][:]



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

pbl18_1 = files18_L0_1['pbl'][:]
pbl18_2 = files18_L0_2['pbl'][:]
pbl18_3 = files18_L0_3['pbl'][:]
pbl18_4 = files18_L0_4['pbl'][:]
pbl18_5 = files18_L0_5['pbl'][:]
pbl18_6 = files18_L0_6['pbl'][:]
pbl18_7 = files18_L0_7['pbl'][:]
pbl18_8 = files18_L0_8['pbl'][:]
pbl18_9 = files18_L0_9['pbl'][:]
pbl18_10 = files18_L0_10['pbl'][:]
pbl18_11 = files18_L0_11['pbl'][:]

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
e11 = files17_L0_11.variables['beta_raw'][:]
e12 = files17_L0_12.variables['beta_raw'][:]
e13 = files17_L0_13.variables['beta_raw'][:]


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


customdate_L0 = datetime.datetime(year=1901, month=1, day=1, hour=0, second=0)
realtime_L0 = [   datetime.timedelta(seconds=f) + customdate_L0 for f in (int(time_L0))]

utc_L0 = np.array([f.strftime('%H:%M') for f in (realtime_L0)])

#total_cloud_L2 = files_L2.variables['cloud_amount'][:]

#for l in np.where(total_cloud_L2 == 0):
#  print(utc_L2[l])

#%%

#------------------PROCESS ALL 2020

a = int(len(b1)/2)
print(a)


##------

#5740/24 hours = 239

timespan1 = 18* 239 
timespan2 = 19 * 239 
#np.where(utc_L2 == '21:00')
#timespan2 = np.where(utc_L2 == '23:55' )

#d_night_test1 = b1[:,248:250] 
d_night_test1 = b1[(timespan1):(timespan2),:] 

d_nightsum_test1 = np.mean(d_night_test1,0)

noise1 = np.mean(d_nightsum_test1[950:1024])
print(noise1)
d_nightsum_test1 = d_nightsum_test1 + noise1

for f in overlap:
  dens_overlap_test1 = d_nightsum_test1 * overlap
#If overlap is needed:
[x1,y1] = coadd(dens_overlap_test1,range_L0,5)

##------

timespan1 = 0 
#np.where(utc_L2 == '00:00')
timespan2 =  3 * 239  
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

timespan1 = 15 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 24 * 239 
#np.where(utc_L2 == '03:00' )

d_night_test3 = b3[(timespan1):(timespan2),:] 
d_nightsum_test3 = np.mean(d_night_test3,0)

noise3 = np.mean(d_nightsum_test3[950:1024])
print(noise3)
d_nightsum_test3 = d_nightsum_test3 + noise3

for f in overlap:
  dens_overlap_test3 = d_nightsum_test3 * overlap
#If overlap is needed:
[x3,y3] = coadd(dens_overlap_test3,range_L0,5)

##------

timespan1 = 0* 239 
#np.where(utc_L2 == '22:00')
timespan2 = 3 *239
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

timespan1 = 21*239
#np.where(utc_L2 == '00:00')
timespan2 = 23 * 239
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

timespan1 = 21 * 239 
#np.where(utc_L2 == 00:00')
timespan2 = 24 * 239 
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

timespan1 = 0 * 239 
#np.where(utc_L2 == '22:30')
timespan2 = 15 * 239 
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

timespan1 = 2 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 6 * 239 
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




plt.plot(x8, y2)
#plt.semilogx(x1[0:70:], y1[0:70])

plt.xlim([0,500000])
plt.ylim([0, 4000])

#check   x10 (2000m), x11 (3000m),  x18 nothing showing up , 



#Step at 2000m starts at x7, x8, x9  

#%%

#-------PROCESS ALL 2019

#------

timespan1 = 18*239 
#np.where(utc_L2 == '13:30')
timespan2 = 24 * 239 
#np.where(utc_L2 == '14:30' )


d19_night_test1 = c1[(timespan1):(timespan2),:]  
d19_nightsum_test1 = np.mean(d19_night_test1,0)

noise11 = np.mean(d19_nightsum_test1[800:1024])
print(noise11)
d19_nightsum_test1 = d19_nightsum_test1 - noise11

for f in overlap:
  dens19_overlap_test1 = d19_nightsum_test1 * overlap
#If overlap is needed:
[xx1,yy1] = coadd(dens19_overlap_test1,range_L0,5)

#------

timespan1 = 0 * 239 
#np.where(utc_L2 == '08:30')
timespan2 = 9 * 239 
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

timespan1 = 21 * 239 
#np.where(utc_L2 == '12:30')
timespan2 = 24 * 239 
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
timespan1 = 18* 239 
#np.where(utc_L2 == '07:00')
timespan2 = 24 * 239 
#np.where(utc_L2 == '08:30' )


d19_night_test4 = c4[(timespan1):(timespan2),:] 
d19_nightsum_test4 = np.mean(d19_night_test4,0)

pbl19plot_4 = pbl19_4[:,0][timespan1:timespan2]


noise44 = np.mean(d19_nightsum_test4[950:1024])
print(noise44)
d19_nightsum_test4 = d19_nightsum_test4 + noise44

for f in overlap:
  dens19_overlap_test4 = d19_nightsum_test4 * overlap
#If overlap is needed:
[xx4,yy4] = coadd(dens19_overlap_test4,range_L0,5)

#--------

timespan1 = 6 * 239 
#np.where(utc_L2 == '16:30')
timespan2 = 9 * 239 
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

timespan1 = 15 * 239 
#np.where(utc_L2 == '01:20')
timespan2 = 24 * 239 
# np.where(utc_L2 == '03:00' )


d19_night_test6 = c6[(timespan1):(timespan2),:]  
d19_nightsum_test6 = np.mean(d19_night_test6,0)

pbl19plot_6 = pbl19_6[:,0][timespan1:timespan2]



noise66 = np.mean(d19_nightsum_test6[960:1024])
print(noise66)
d19_nightsum_test6 = d19_nightsum_test6 + noise66

for f in overlap:
  dens19_overlap_test6 = d19_nightsum_test6 * overlap
#If overlap is needed:
[xx6,yy6] = coadd(dens19_overlap_test6,range_L0,5)

#--------

timespan1 = 0*129
#np.where(utc_L2 == '23:00')
timespan2 = 9*239
#np.where(utc_L2 == '23:55' )

d19_night_test7 = c7[(timespan1):(timespan2),:] 
d19_nightsum_test7 = np.mean(d19_night_test7,0)

pbl19plot_7 = pbl19_7[:,0][timespan1:timespan2]

noise77 = np.mean(d19_nightsum_test7[960:1024])
print(noise77)
d19_nightsum_test7 = d19_nightsum_test7 + noise77

for f in overlap:
  dens19_overlap_test7 = d19_nightsum_test7 * overlap
#If overlap is needed:
[xx7,yy7] = coadd(dens19_overlap_test7,range_L0,5)

#------

timespan1 = 18*239
#np.where(utc_L2 == '06:30')
timespan2 = 21*239
#np.where(utc_L2 == '08:30' )

d19_night_test8 = c8[(timespan1):(timespan2),:] 
d19_nightsum_test8 = np.mean(d19_night_test8,0)

noise88 = np.mean(d19_nightsum_test8[950:1024])
print(noise88)
d19_nightsum_test8 = d19_nightsum_test8 + noise88

for f in overlap:
  dens19_overlap_test8 = d19_nightsum_test8 * overlap
#If overlap is needed:
[xx8,yy8] = coadd(dens19_overlap_test8,range_L0,5)


#-------

timespan1 =6 * 239 
 #np.where(utc_L2 == '00:00')
timespan2 = 9*239 
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
d19_nightsum_test10 = d19_nightsum_test10 - noise99

for f in overlap:
  dens19_overlap_test10 = d19_nightsum_test10 * overlap
#If overlap is needed:
[xx10,yy10] = coadd(dens19_overlap_test10,range_L0,5)

#--------

timespan1 = 0
# np.where(utc_L2 == '00:00')
timespan2 = 6 *239
#np.where(utc_L2 == '02:00' )

d19_night_test11 = c11[(timespan1):(timespan2),:] 
d19_nightsum_test11 = np.mean(d19_night_test11,0)

noise99 = np.mean(d19_nightsum_test11[960:1024])
d19_nightsum_test11 = d19_nightsum_test11 + noise99

for f in overlap:
  dens19_overlap_test11 = d19_nightsum_test11 * overlap
#If overlap is needed:
[xx11,yy11] = coadd(dens19_overlap_test11,range_L0,5)

#------

timespan1 = 22*239
#np.where(utc_L2 == '15:30')
timespan2 = 24*239
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

timespan1 = 0 * 239
#np.where(utc_L2 == '03:00')
timespan2 =  12*239 
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

timespan1 = 0 
#np.where(utc_L2 == '00:00')
timespan2 = 7 * 239
#np.where(utc_L2 == '01:00' )

d19_night_test14 = c14[(timespan1):(timespan2),:] 
d19_nightsum_test14 = np.mean(d19_night_test14,0)

#pbl19plot_14 = pbl19_14[:,0][timespan1:timespan2]

noise99 = np.mean(d19_nightsum_test14[950:1024])
d19_nightsum_test14 = d19_nightsum_test14 + noise99

for f in overlap:
  dens19_overlap_test14 = d19_nightsum_test14 * overlap
#If overlap is needed:
[xx14,yy14] = coadd(dens19_overlap_test14,range_L0,5)

#------
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3 = fig.add_subplot(111)
ax4 = fig.add_subplot(111)
ax5 = fig.add_subplot(111)

#ax5.scatter(xx4, pbl19plot_4[0:201],marker='.',c='b', s=5, label='Aerosol layer - April 7')
#ax5.scatter(xx6, pbl19plot_6[0:201],marker='.',c='y',s=5, label='Aerosol layer - April 9')
#ax5.scatter(xx9, pbl19plot_9[0:201],marker='.',c='g', s=5, label='Aerosol layer - April 14')
#ax5.scatter(xx14, pbl19plot_4[0:201],marker='.',c='g', s=5, label='Aerosol layer - April 7')
ax1.plot(xx14, yy1, label='April 7, 2019')

#ax2.semilogx(xx6, yy1, label='April 9, 2019')
#ax3.semilogx(xx7, yy1, label='April 12, 2019')
#ax4.semilogx(xx9, yy1, label='April 14, 2019')
#ax5.semilogx(xx14, yy1, label='April 23, 2019')

ax1.set_xlim([10,500000])
ax1.set_ylim([0, 5000])
ax1.legend(loc='upper right', fontsize='small')
ax1.set_xlabel('Logarithm of Range-Corrected Backscatter Power (a.u.)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Averaged, Range-Corrected Signal Power: April ')

#4000m spike in: xx2, xx4, xx13
#1000m spike in: xx5
#higher average in xx7

#xx5 high, xx17 high at 1000m, 

#%%
#--------------------------------------------------
#Calculate all profiles for 2018:




#-------

timespan1 = (6* 239)  
# np.where(utc_L2 == '10:30')
timespan2 = (8*239) 
#np.where(utc_L2 == '11:30')


d18_night_test1 = d1[(timespan1):(timespan2),:]
d18_nightsum_test1 = np.mean(d18_night_test1,0)

noise11 = np.mean(d18_nightsum_test1[950:1024])
print(noise11)
d18_nightsum_test1 = d18_nightsum_test1 + noise11

for f in overlap:
  dens18_overlap_test1 = d18_nightsum_test1 * overlap
#If overlap is needed:
[xxx1,yyy1] = coadd(dens18_overlap_test1,range_L0,5)

#------

timespan1 = 18* 239 
#np.where(ut_L2 == '09:00')
timespan2 = 22 * 239 
#np.where(utc_L2 == '12:00')

d18_night_test2 = d2[(timespan1):(timespan2),:]
d18_nightsum_test2 = np.mean(d18_night_test2,0)

noise11 = np.mean(d18_nightsum_test2[950:1024])
print(noise11)
d18_nightsum_test2 = d18_nightsum_test2 + noise11

for f in overlap:
  dens18_overlap_test2 = d18_nightsum_test2 * overlap
#If overlap is needed:
[xxx2,yyy2] = coadd(dens18_overlap_test2,range_L0,5)

#-------

timespan1 = (7 * 239) 
#np.where(utc_L2 == '01:30')
timespan2 = (9* 239) 
#np.where(utc_L2 == '05:30')

d18_night_test3 = d3[(timespan1):(timespan2),:]
d18_nightsum_test3 = np.mean(d18_night_test3,0)

noise11 = np.mean(d18_nightsum_test3[950:1024])
print(noise11)
d18_nightsum_test3 = d18_nightsum_test3 - noise11

for f in overlap:
  dens18_overlap_test3 = d18_nightsum_test3 * overlap
#If overlap is needed:
[xxx3,yyy3] = coadd(dens18_overlap_test3,range_L0,5)

#------

timespan1 = 3 * 239 
 #np.where(utc_L2 == '00:00')
timespan2 = (6 *239 ) 
 #np.where(utc_L2 == '01:30')

d18_night_test4 = d4[(timespan1):(timespan2),:] 
d18_nightsum_test4 = np.mean(d18_night_test4,0)

pbl18plot_4 = pbl18_4[:,0][timespan1:timespan2]


noise11 = np.mean(d18_nightsum_test4[950:1024])
print(noise11)
d18_nightsum_test4 = d18_nightsum_test4 + noise11

for f in overlap:
  dens18_overlap_test4 = d18_nightsum_test4 * overlap
#If overlap is needed:
[xxx4,yyy4] = coadd(dens18_overlap_test4,range_L0,5)

#------

timespan1 = 15 * 239
 #np.where(utc_L2 == '09:00')
timespan2 = 18* 239 
#np.where(utc_L2 == '11:30')

d18_night_test5 = d5[(timespan1):(timespan2),:]
d18_nightsum_test5 = np.mean(d18_night_test5,0)

pbl18plot_5 = pbl18_5[:,0][timespan1:timespan2]

noise11 = np.mean(d18_nightsum_test5[950:1024])
print(noise11)
d18_nightsum_test5 = d18_nightsum_test5 + noise11

for f in overlap:
  dens18_overlap_test5 = d18_nightsum_test5 * overlap
#If overlap is needed:
[xxx5,yyy5] = coadd(dens18_overlap_test5,range_L0,5)

#-------

timespan1 =15 * 239 
#np.where(utc_L2 == '12:00')
timespan2 = 18 * 239 
#np.where(utc_L2 == '13:00')

d18_night_test6 = d6[(timespan1):(timespan2),:]
d18_nightsum_test6 = np.mean(d18_night_test6,0)

noise11 = np.mean(d18_nightsum_test6[960:1024])
print(noise11)
d18_nightsum_test6 = d18_nightsum_test6 + noise11

for f in overlap:
  dens18_overlap_test6 = d18_nightsum_test6 * overlap
#If overlap is needed:
[xxx6,yyy6] = coadd(dens18_overlap_test6,range_L0,5)

#-----

timespan1 = (11 * 239) 
#np.where(utc_L2 == '22:30')
timespan2 = (14 * 239) 
#np.where(utc_L2 == '23:55')

d18_night_test7 = d7[(timespan1):(timespan2),:]
d18_nightsum_test7 = np.mean(d18_night_test7,0)

pbl18plot_7 = pbl18_7[:,0][timespan1:timespan2]


noise11 = np.mean(d18_nightsum_test7[950:1024])
print(noise11)
d18_nightsum_test7 = d18_nightsum_test7 + noise11

for f in overlap:
  dens18_overlap_test7 = d18_nightsum_test7 * overlap
#If overlap is needed:
[xxx7,yyy7] = coadd(dens18_overlap_test7,range_L0,5)

#-----

timespan1 = 4 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 7 * 239 
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

timespan1 = 2 * 239  
#np.where(utc_L2 == '00:00')
timespan2 = (239*4)
#np.where(utc_L2 == '01:30')

d18_night_test9 = d9[(timespan1):(timespan2),:]
d18_nightsum_test9 = np.mean(d18_night_test9,0)

pbl18plot_9 = pbl18_9[:,0][timespan1:timespan2]

noise11 = np.mean(d18_nightsum_test9[950:1024])
print(noise11)
d18_nightsum_test9 = d18_nightsum_test9 + noise11

for f in overlap:
  dens18_overlap_test9 = d18_nightsum_test9 * overlap
#If overlap is needed:
[xxx9,yyy9] = coadd(dens18_overlap_test9,range_L0,5)

#------

timespan1 = 3*239
#np.where(utc_L2 == '02:00')
timespan2 =9*239
 #np.where(utc_L2 == '03:00')

d18_night_test10 = d10[(timespan1):(timespan2),:]
d18_nightsum_test10 = np.mean(d18_night_test10,0)

noise11 = np.mean(d18_nightsum_test10[950:1024])
print(noise11)
d18_nightsum_test10 = d18_nightsum_test10 + noise11

for f in overlap:
  dens18_overlap_test10 = d18_nightsum_test10 * overlap
#If overlap is needed:
[xxx10,yyy10] = coadd(dens18_overlap_test10,range_L0,5)

#------

timespan1 = 0 * 239 
#np.where(utc_L2 == '20:00')
timespan2 = 3 * 239 
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



ax1.plot(xxx12, yyy1, label='April 6, 2018')
#ax2.semilogx(xxx7, yy1, label='April 9, 2018')
#ax5.semilogx(xxx9, yy1, label='April 12, 2018')
#ax3.semilogx(xxx10, yy1, label='April 23, 2018')
#ax4.semilogx(xxx8, yy1, label='April 23, 2019')
#ax5.scatter(xxx14, pbl18plot_14[0:201],marker='.',c='r', s=5, label='Aerosol layer - April 23')

#ax6.semilogx(xxx9, yy1, label='')
#ax7.semilogx(xxx16, yy1, label='April 27, 2018')

ax1.set_xlim([1,1000000])
ax1.set_ylim([0, 5000])
ax1.legend(fontsize='small')
ax1.set_xlabel('Logarithm of Range-Corrected Backscatter Power (a.u.)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Averaged, Range-Corrected Signal Power: April 2018 ')

#xxx3 has spike 2000m
#xxx6 has spikes 1000m and 1900m
#xxx7 has overall higher starting 2000m
#xxx8 has spoke 2000m
#xxx9 has higher from 1500m to 1000m, same xxx10
#xxx14 bump to 2000m, xxx16, xxx5

#xxx1, xxx3, xxx8

#%%
#-----------------------------------------------------------
#Calculate profiles for 2017: 


timespan1 = 3 * 239
#np.where(utc_L2 == '00:00')
timespan2 = 6 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test1 = e1[(timespan1):(timespan2),:]
d17_nightsum_test1 = np.mean(d17_night_test1,0)

noise11 = np.mean(d17_nightsum_test1[950:1024])
print(noise11)
d18_nightsum_test1 = d17_nightsum_test1 + noise11

for f in overlap:
  dens17_overlap_test1 = d17_nightsum_test1 * overlap
#If overlap is needed:
[xxxx1,yyyy1] = coadd(dens17_overlap_test1,range_L0,5)

#--------

timespan1 = 9 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 10 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test2 = e2[(timespan1):(timespan2),:]
d17_nightsum_test2 = np.mean(d17_night_test2,0)

noise11 = np.mean(d17_nightsum_test2[950:1024])
print(noise11)
d18_nightsum_test2 = d17_nightsum_test1 + noise11

for f in overlap:
  dens17_overlap_test2 = d17_nightsum_test2 * overlap
#If overlap is needed:
[xxxx2,yyyy2] = coadd(dens17_overlap_test2,range_L0,5)

#--------



timespan1 = 6 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 9 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test3 = e3[(timespan1):(timespan2),:]
d17_nightsum_test3 = np.mean(d17_night_test3,0)

noise11 = np.mean(d17_nightsum_test3[950:1024])
print(noise11)
d18_nightsum_test3 = d17_nightsum_test3 + noise11

for f in overlap:
  dens17_overlap_test3 = d17_nightsum_test3 * overlap
#If overlap is needed:
[xxxx3,yyyy3] = coadd(dens17_overlap_test3,range_L0,5)

#--------
timespan1 = 2 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 9 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test4 = e4[(timespan1):(timespan2),:]
d17_nightsum_test4 = np.mean(d17_night_test4,0)

noise11 = np.mean(d17_nightsum_test4[950:1024])
print(noise11)
d18_nightsum_test4 = d17_nightsum_test4 + noise11

for f in overlap:
  dens17_overlap_test4 = d17_nightsum_test4 * overlap
#If overlap is needed:
[xxxx4,yyyy4] = coadd(dens17_overlap_test4,range_L0,5)

#--------

timespan1 = 9 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 10 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test5 = e5[(timespan1):(timespan2),:]
d17_nightsum_test5 = np.mean(d17_night_test5,0)

noise11 = np.mean(d17_nightsum_test5[950:1024])
print(noise11)
d18_nightsum_test5 = d17_nightsum_test5 + noise11

for f in overlap:
  dens17_overlap_test5 = d17_nightsum_test5 * overlap
#If overlap is needed:
[xxxx5,yyyy5] = coadd(dens17_overlap_test5,range_L0,5)

#--------
timespan1 = 19 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 21 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test6 = e6[(timespan1):(timespan2),:]
d17_nightsum_test6 = np.mean(d17_night_test6,0)

noise11 = np.mean(d17_nightsum_test6[950:1024])
print(noise11)
d18_nightsum_test6 = d17_nightsum_test6 + noise11

for f in overlap:
  dens17_overlap_test6 = d17_nightsum_test6 * overlap
#If overlap is needed:
[xxxx6,yyyy6] = coadd(dens17_overlap_test6,range_L0,5)

#--------
timespan1 =   0 
#np.where(utc_L2 == '00:00')
timespan2 = 239 
#np.where(utc_L2 == '06:00')

d17_night_test7 = e7[(timespan1):(timespan2),:]
d17_nightsum_test7 = np.mean(d17_night_test7,0)

noise11 = np.mean(d17_nightsum_test7[950:1024])
print(noise11)
d18_nightsum_test7 = d17_nightsum_test7 + noise11

for f in overlap:
  dens17_overlap_test7 = d17_nightsum_test7 * overlap
#If overlap is needed:
[xxxx7,yyyy7] = coadd(dens17_overlap_test7,range_L0,5)

#--------

timespan1 = 6*239  
#np.where(utc_L2 == '00:00')
timespan2 =  21*239 
#np.where(utc_L2 == '06:00')

d17_night_test8 = e8[(timespan1):(timespan2),:]
d17_nightsum_test8 = np.mean(d17_night_test8,0)

noise11 = np.mean(d17_nightsum_test8[950:1024])
print(noise11)
d18_nightsum_test8 = d17_nightsum_test8 + noise11

for f in overlap:
  dens17_overlap_test8 = d17_nightsum_test8 * overlap
#If overlap is needed:
[xxxx8,yyyy8] = coadd(dens17_overlap_test8,range_L0,5)

#--------


timespan1 = 16 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 18 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test9 = e9[(timespan1):(timespan2),:]
d17_nightsum_test9 = np.mean(d17_night_test9,0)

noise11 = np.mean(d17_nightsum_test9[950:1024])
print(noise11)
d18_nightsum_test9 = d17_nightsum_test9 + noise11

for f in overlap:
  dens17_overlap_test9 = d17_nightsum_test9 * overlap
#If overlap is needed:
[xxxx9,yyyy9] = coadd(dens17_overlap_test9,range_L0,5)

#--------


timespan1 = 3 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 6 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test10 = e10[(timespan1):(timespan2),:]
d17_nightsum_test10 = np.mean(d17_night_test10,0)

noise11 = np.mean(d17_nightsum_test10[950:1024])
print(noise11)
d18_nightsum_test10 = d17_nightsum_test10 + noise11

for f in overlap:
  dens17_overlap_test10 = d17_nightsum_test10 * overlap
#If overlap is needed:
[xxxx10,yyyy10] = coadd(dens17_overlap_test10,range_L0,5)

#--------

timespan1 = 15* 239 
#np.where(utc_L2 == '00:00')
timespan2 = 24 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test11 = e11[(timespan1):(timespan2),:]
d17_nightsum_test11 = np.mean(d17_night_test11,0)

noise11 = np.mean(d17_nightsum_test11[950:1024])
print(noise11)
d18_nightsum_test11 = d17_nightsum_test11 + noise11

for f in overlap:
  dens17_overlap_test11 = d17_nightsum_test11 * overlap
#If overlap is needed:
[xxxx11,yyyy11] = coadd(dens17_overlap_test11,range_L0,5)

#--------
timespan1 = 9 * 239 
#np.where(utc_L2 == '00:00')
timespan2 = 12 * 239 
#np.where(utc_L2 == '06:00')

d17_night_test12 = e12[(timespan1):(timespan2),:]
d17_nightsum_test12 = np.mean(d17_night_test12,0)

noise11 = np.mean(d17_nightsum_test12[950:1024])
print(noise11)
d18_nightsum_test12 = d17_nightsum_test12 + noise11

for f in overlap:
  dens17_overlap_test12 = d17_nightsum_test12 * overlap
#If overlap is needed:
[xxxx12,yyyy12] = coadd(dens17_overlap_test12,range_L0,5)

#--------

timespan1 = 0 * 239 
#np.where(utc_L2 == '00:00')
timespan2 =  4* 239 +119

#np.where(utc_L2 == '06:00') 0-4am

d17_night_test13 = e13[(timespan1):(timespan2),:]
d17_nightsum_test13 = np.mean(d17_night_test13,0)

noise11 = np.mean(d17_nightsum_test13[950:1024])
print(noise11)
d18_nightsum_test13 = d17_nightsum_test13 + noise11

for f in overlap:
  dens17_overlap_test13 = d17_nightsum_test13 * overlap
#If overlap is needed:
[xxxx13,yyyy13] = coadd(dens17_overlap_test13,range_L0,5)

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
#ax2.semilogx(xxxx9, yy1, label='April 18, 2017')
ax3.semilogx(xxxx4, yy1, label='April 24, 2017')
#ax4.semilogx(xxx8, yy1, label='April 23, 2019')
#ax5.semilogx(xxxx13, yy1, label='April 28, 2017')
#ax6.semilogx(xxx9, yy1, label='')
#ax7.semilogx(xxx16, yy1, label='April 27, 2018')

ax1.set_xlim([10000,10000000])
ax1.set_ylim([0, 5000])
ax1.legend()
ax1.set_xlabel('Logarithm of Range-Corrected Backscatter Power (a.u.)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Averaged, Range-Corrected Signal Power: April ')

#spike xxxx2 EXCLUDE, spike xxxx5 EXCLUDE, xxxx12

#xxxx8 2000m spike, xxxx9 spike 2000m, xxxx11, xxxx13

# %%

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3 = fig.add_subplot(111)
ax4 = fig.add_subplot(111)
ax5 = fig.add_subplot(111)
ax6 = fig.add_subplot(111)
ax7 = fig.add_subplot(111)

#ax3.semilogx(xxxx11, yy1, label='April 24, 2017')
ax3.plot(xx5, yy1, lw = 0.6, label='April 8,2019')
#ax4.plot(xx7, yy1,lw = 0.6, label='April 19, 2019')
ax4.plot(xx14, yy1,lw = 0.6, label='April 23, 2019')

#ax5.plot(xx16, yy1, lw = 0.6,label='June 27, 2019')

ax2.plot(xxx7, yy1, lw = 0.6,label='April 9, 2018')
ax5.plot(xxx9, yy1,lw = 0.6, label='April 12, 2018')
ax3.plot(xxx14, yy1,lw = 0.6, label='April 23, 2018')

ax1.plot(xxxx8, yy1,lw = 0.6, label='April 17, 2017')
ax2.plot(xxxx9, yy1, lw = 0.6,label='April 18, 2017')

#ax1.plot(xxx4, yy1, label='April 6, 2018')


#ax5.plot(xxxx8, yy1, lw=0.6,label='June 25, 2017')

#ax1.semilogx(xx7, yy1, label='April 9, 2019')
#ax5.plot(xx15, yy1, lw = 0.8,label='June 26, 2019')
ax1.plot(comb_mean20T, y5, lw=2, label='April 2020 average')

ax1.set_xlim([10000,500000])
ax1.set_ylim([0, 7000])
ax1.legend(fontsize='small')
ax1.set_xlabel('Range-Corrected Backscatter Power (a.u.)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Averaged, Range-Corrected Signal Power over 1 month: April')
#%%


#--------------MAKE AVERAGE PLOT OF ALL MONTHS -----------------


fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3 = fig.add_subplot(111)
ax4 = fig.add_subplot(111)

combx = np.array([ [x1], [x2], [x3],[x4], [x5], [x6], [x7],  [x8]])
combx19 = np.array([ [xx1], [xx2], [xx3], [xx4], [xx5],[xx6], [xx7], [xx8], [xx9], [xx10],[xx11],[xx12],[xx13],[xx14]])
combx18 = np.array([ [xxx1], [xxx2], [xxx3], [xxx4], [xxx5],[xxx6], [xxx7], [xxx8], [xxx9], [xxx10],[xxx11]])
#combx17 = np.array([ [xxxx1],  [xxxx3], [xxxx4],  [xxxx6],[xxxx7],[xxxx8],[xxxx9], [xxxx10], [xxxx11], [xxxx12], [xxxx13]])


comb_mean20 = np.mean(combx, axis=0)
comb_mean19 = np.mean(combx19, axis=(0))
comb_mean18 = np.mean(combx18, axis=(0))
#comb_mean17 = np.mean(daylist3, axis=(0))


comb_mean20T = np.transpose(comb_mean20)
comb_mean19T = np.transpose(comb_mean19)
comb_mean18T = np.transpose(comb_mean18)
#comb_mean17T = np.transpose(comb_mean17)

#ax1.plot(xx11, y5, label='April 2020 average')

ax1.plot(comb_mean20T, y5, label='April 2020 average')
ax2.plot(comb_mean19T, y5, label='April 2019 average')
ax3.plot(comb_mean18T, y5, label='April 2018 average')
#ax4.plot(comb_mean17T, y5, label='April 2017 average')

#ax1.semilogx(comb_mean20T[0:102:], y1, label='April 2020 average')
#ax2.semilogx(comb_mean19T[0:102:], y1, label='April 2019 average')
#ax3.semilogx(comb_mean18T[0:102:], y1, label='April 2018 average')

ax1.set_xlim([1,1000000])
ax1.set_ylim([0, 5000])
ax1.legend()
ax1.set_xlabel('Logarithm of Range-Corrected Backscatter Power (a.u.)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Averaged, Range-Corrected Signal Power over 1 month: April')

plt.show()
## %%

# %%
# RMS ERROR SECTION HERE -----------------------------------

#take mean of all years other than 2020:




comb_17_19 = np.array([ [xx1], [xx2],  [xx4], [xx6], [xx7], [xx8], [xx9], [xx14], [xx15], [xx16], [xx17],
[xxx1], [xxx2], [xxx3], [xxx4], [xxx6], [xxx7], [xxx8], [xxx9], [xxx10],[xxx12],[xxx13], [xxx16], [xxx18],
  [xxxx4],  [xxxx6],[xxxx7],[xxxx8],[xxxx9],  [xxxx11], [xxxx12], [xxxx13]])

#comb_mean_17_19 = np.mean(comb_17_19, axis=0)

#comb_mean_17_19T  = np.transpose(comb_mean_17_19)

#observed and predicted values:
#these are the averages for each period of time in question
#comb_mean_17_19 = np.mean(comb_17_19, axis=0)   #expected values
combx1 = np.array([ [x1], [x2], [x3],[x4], [x5], [x6], [x12], [x13], [x14], [x15], [x16],   [x21], [x22]])

comb_mean_17_19 = np.sum(comb_17_19, axis=0)/32   #expected values
comb_mean20 = (np.sum(combx1, axis=0))/13       #observed values (actual)




comb_mean_17_19T  = np.transpose(comb_mean_17_19)

#Compute the standard deviation via rms: 
comb_mean_17_19_square = comb_mean20**2


rms_mean = np.sum(comb_mean20[:,0:200], axis=1)/200
rms_mean_square = rms_mean**2

#square the number, square mean, subtract them and take square root 

rms_diff =  comb_mean20[:,0:200]- rms_mean 
rmss = np.sum((rms_diff**2))/200


rms = np.sqrt(rmss) 
print(rms)

percent_20 = rms/rms_mean

comb_mean20_error = comb_mean20 * percent_20


comb_mean20T  = np.transpose(comb_mean20)




#rms = np.sqrt(resid_mean)



test = comb_mean_17_19[:,0:200] - rms
test2 = comb_mean_17_19[:,0:200]+ rms

error = np.squeeze(comb_mean20_error[:,0:60])


fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)

#plt.axvline(x=rms, color='r', linestyle='-.', linewidth=1,label='Upper limit height of aerosol layer 1')
#ax1.errorbar(comb_mean_17_19T, y5, xerr=rms, fmt='-')
daylist = [[x1], [x2], [x3],[x4], [x5], [x6], [x12], [x13], [x14], [x15], [x16],   [x21], [x22]]
daylist1 = [[xx1], [xx2],  [xx4], [xx6], [xx7],  [xx9], [xx14], [xx15], [xx16], [xx17]]
daylist2 = [ [xxx1], [xxx2], [xxx3], [xxx4], [xxx6], [xxx7], [xxx8], [xxx9], [xxx10],[xxx12],[xxx13], [xxx16], [xxx18]]
daylist3 = [  [xxxx4],  [xxxx6],[xxxx7],[xxxx8],[xxxx9],  [xxxx11], [xxxx12], [xxxx13]]
#take out xx11
for i in daylist3:
  plt.plot(np.transpose(i), y1, lw=0.6)



#ax1.errorbar(comb_mean20T, y5,'none', np.squeeze(comb_mean20*percent_20))

ax1.plot(comb_mean20T, y5, c='b', label='April 2020 average')
ax1.plot(comb_mean_17_19T, y5,lw=2.0,c='r' ,label='April 2017-2019 average (39 days)')

#ax1.plot(np.transpose(rms), y5, label='m')
ax1.plot((np.transpose(test)), y5[0:200], linestyle = '--',lw=1.3, c='r',label='Data within 1 standard dev of April 2020')
ax1.plot((np.transpose(test2)), y5[0:200],linestyle = '--', lw=1.3,c='r')


ax1.set_xlim([10000,250000])
ax1.set_ylim([0, 3000])
ax1.legend(fontsize='small', loc='upper right')
ax1.set_xlabel(' Range-Corrected Backscatter Power (a.u.)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Range-Corrected Signal Power over April 2017 (8 clear days)')
plt.savefig('shade.png', dpi=300)

plt.show()
# %%


#calculate mean at each data point: 

rms_mean19 = np.mean(comb_mean19[:,0:40], axis=1)/40

#square the number, square mean, subtract them and take square root 

rms_diff19 =  comb_mean19[:,0:40]- rms_mean19 
rmss19 = np.sum((rms_diff19**2))/40


rms19 = np.sqrt(rmss19) 
print(rms)
comb_mean19t = np.mean(comb_mean19, axis=0)

percent_19 = rms19/(comb_mean19)
#comb_mean19_error = comb_mean19 * percent_19

#----------


rms_mean20 = np.sum(comb_mean20[:,0:70], axis=1)/70
rms_diff20 = comb_mean20[:,0:70] - rms_mean20
rmss20 = np.sum(rms_diff20**2)/70
rms20 = np.sqrt(rmss20)


percent_20 = rms20/(comb_mean20)

#comb_mean20_error = comb_mean20 * percent_20

#------

rms_mean18 = np.sum(comb_mean18[:,0:200], axis=1)/200
rms_diff18 = comb_mean18[:,0:200] - rms_mean18
rmss18 = np.sum(rms_diff18**2)/200
rms18 = np.sqrt(rmss18)


percent_18 = rms18/(comb_mean18)

#comb_mean18_error = comb_mean18 * percent_18
#------
rms_mean17 = np.sum(comb_mean17[:,0:200], axis=1)/200
rms_diff17 = comb_mean17[:,0:200] - rms_mean20
rmss17 = np.sum(rms_diff17**2)/200
rms17 = np.sqrt(rmss17)


percent_17 = rms17/(comb_mean17)

#comb_mean17_error = comb_mean17 * percent_17





fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax1.plot(np.transpose((percent_20*100)[:,0:70]), y5[0:70], label='RMS of April 2020')
ax1.plot(np.transpose((percent_19*100)[:,0:60]), y5[0:60], label='RMS of April 2019')
ax1.plot(np.transpose((percent_18*100)[:,0:70]), y5[0:70], label='RMS of April 2018')

ax1.plot(np.transpose((percent_17*100)[:,0:70]), y5[0:70], label='RMS of April 2017')





#plt.axvline(x=percent_20*100, color='b', linestyle='-.', linewidth=1,label='RMS percent value for 2020')
#plt.axvline(x=percent_19*100, color='y', linestyle='-.', linewidth=1,label='RMS percent value for 2019')
#plt.axvline(x=percent_18*100, color='g', linestyle='-.', linewidth=1,label='RMS percent value for 2018')
#plt.axvline(x=percent_17*100, color='r', linestyle='-.', linewidth=1,label='RMS percent value for 2017')



ax1.set_xlim([0,150])
ax1.set_ylim([0, 3000])
ax1.legend(fontsize='small', loc='upper right')
ax1.set_xlabel('Root mean square variation (%)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Percent change in RMS for backscatter power in April')



plt.show()

# %%
from scipy import stats


print(stats.levene(np.squeeze(comb_mean17),np.squeeze(comb_mean18)))

stats.ttest_ind(comb_mean20,comb_mean19,axis=1, equal_var=True)

# %%
