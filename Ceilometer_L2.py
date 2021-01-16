#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.matlib as matlib


from matplotlib import cm

import pandas as pd
import netCDF4 

import xarray as xr

# Iniatalize coadd function to be used below 

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
#----------------------------
#Following function recieved from http://xarray.pydata.org/en/stable/io.html and modified as such
# Which should work sufficiently well for large datasets
#-REMOVED-





#-------XARRAY LOAD FILES---------------------

datatest = xr.open_mfdataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/01/L2_*.nc', concat_dim="time", data_vars='minimal', coords='minimal', compat='override')

beta_att = datatest.variables['attenuated_backscatter_0']
range = datatest.variables['altitude']
station_alt = datatest.variables['station_altitude']

time = datatest.variables['time']

#lt.plot(beta_att, Range)

print(np.shape(beta_att))
print(np.shape(range))
print(np.shape(time))

plt.scatter(beta_att[0:512], range)




# %%
# Attempting to use the xarray dataframes to make a normalized, averaged signal plot
# did not end up working correctly so see below

zenith2 = 89
print(zenith2)
plotalt = np.matlib.repmat(range*np.sin(zenith2)+station_alt, len(time), 1)

a = int(len(beta_att_test)/2)
print(a)
d_day = beta_att.iloc[:,a:a*2] 
d_daysum = np.mean(d_day,1)
print(d_daysum)

[x,y] = coadd(d_daysum,range,10)
#x = np.abs(x)
#logprofile = np.log(x)

# %%
#----------------------
# Attempting using the netcdf4 module instead of xarray 
# This should be easier as this will load in a numpy like array BUT
# does not handle multiple datasets in the netcdf4 format



t = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/01/L2_0-20000-0-73009_A20200401.nc")
beta_att_test = t.variables['attenuated_backscatter_0']
range_test = t.variables['altitude']
time_test = t.variables['time']

plt.plot(beta_att_test, range_test)



# %%
#-------------------------WORKING--------------------
# Make night averaged, normalized plot in a similar fashion to the 
# way that it is created for the L0 data. 
# The further corrections done to beta_raw to obtain beta_att should allow 
# for the same manipulation to normalize and average 

#The overlap file is an array of 1024 values. 
# Resize this array to be 511 values in order to multiply with beta_att

overlap = np.loadtxt('data.txt',dtype = float)
print(overlap)
overlap_0 = np.resize(overlap,[511,2])
overlap_atten = overlap_0[:, 0]


a = int(len(beta_att_test)/2)
print(a)
d_night_test = beta_att_test[:,a:a*2] 
d_nightsum_test = np.mean(d_night_test,1)
for f in overlap_atten:
  dens_overlap_test = d_nightsum_test * overlap_atten
print(dens_overlap_test)

print(d_nightsum_test)

[x,y] = coadd(dens_overlap_test,range_test,5)
plt.semilogx(x[0:55],y[0:55])
plt.xlabel('Normalized, Range-Corrected Attenuated Backscatter')
plt.ylabel('Height (m)')
plt.title('Day Average Signal: 2020/04/01')

# %%
#-------------Attempt at making attenuated backscatter colour plot

# Set any value less than or equal to 0 in the beta_att values to 1
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

beta = []
for f in beta_att_test:
  if np.any(f >= 0):
    beta.append(f)
      
  else:
    beta.append(1)


levels = MaxNLocator(nbins=15).tick_values(1, np.max(np.log(beta_abs)))

cmap = plt.get_cmap('RdYlBu')
#cmap = set_color('jet')

norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

betaT = np.transpose(beta)
#fig, (ax0) = plt.subplots(nrows=2)
beta_abs = np.abs(beta)
im = plt.pcolormesh(time, range, np.log10(beta_abs), cmap=cmap, vmin=None, vmax=None, shading='flat')
fig.colorbar(im)
plt.title('Normalized, Attenuated Backscatter Power')
plt.xlabel('Time UT [h]')
plt.ylabel('Altitude a.s.l. (m)')
plt.show()

# %%
print(beta)
# %%
