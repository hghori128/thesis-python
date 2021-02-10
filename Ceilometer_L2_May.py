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

#---------------------COADD-----------
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



#-------XARRAY LOAD FILES---------------------

#-------------LOAD RAYLEIGH CORRECTION FILES------------------------
t11 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis-python/air2020.nc")
f11 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis-python/hgt2020.nc")

pressure = f11.variables['hgt']
print(pressure)
temp = t11.variables['level'][:]


files_test = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/01/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')

backscatter_test = files_test.variables['attenuated_backscatter_0']

# %%
#----------------------
# Attempting using the netcdf4 module instead of xarray 
# This should be easier as this will load in a numpy like array BUT
# does not handle multiple datasets in the netcdf4 format


#%%
files_test2 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/11/20190511_YXU-Cronyn_CHM160155_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')

files_L2 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/05/L2_0-20000-0-73009_A20200505.nc")


#------------------OPEN FILES 2020, 2019 --------------

#2020:


#L2 data


files_L2_1 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/01/L2_0-20000-0-73009_A20200501.nc")
files_L2_2 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/06/L2_0-20000-0-73009_A20200506.nc")
files_L2_3 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/07/L2_0-20000-0-73009_A20200507.nc")
files_L2_4 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/08/L2_0-20000-0-73009_A20200508.nc")
files_L2_5 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/09/L2_0-20000-0-73009_A20200509.nc")
files_L2_6 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/10/L2_0-20000-0-73009_A20200510.nc")
files_L2_7 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/12/L2_0-20000-0-73009_A20200512.nc")
files_L2_8 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/13/L2_0-20000-0-73009_A20200513.nc")
files_L2_9 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/16/L2_0-20000-0-73009_A20200516.nc")
files_L2_10 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/20/L2_0-20000-0-73009_A20200520.nc")


files_L2_11 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/21/L2_0-20000-0-73009_A20200521.nc")
files_L2_12 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/22/L2_0-20000-0-73009_A20200522.nc")
files_L2_13 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/24/L2_0-20000-0-73009_A20200524.nc")
files_L2_14 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/25/L2_0-20000-0-73009_A20200525.nc")
files_L2_15 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/26/L2_0-20000-0-73009_A20200526.nc")
files_L2_16 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/27/L2_0-20000-0-73009_A20200527.nc")
files_L2_17 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/29/L2_0-20000-0-73009_A20200529.nc")
files_L2_18 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/05/31/L2_0-20000-0-73009_A20200531.nc")


#%%
#2019:


#L2
files19_L2_1 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/02/L2_0-20000-0-73009_A20190502.nc")
files19_L2_2 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/05/L2_0-20000-0-73009_A20190505.nc")
files19_L2_3 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/06/L2_0-20000-0-73009_A20190506.nc")
files19_L2_4 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/07/L2_0-20000-0-73009_A20190507.nc")
files19_L2_5 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/08/L2_0-20000-0-73009_A20190508.nc")
files19_L2_6 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/11/L2_0-20000-0-73009_A20190511.nc")
files19_L2_7 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/14/L2_0-20000-0-73009_A20190514.nc")
files19_L2_8 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/15/L2_0-20000-0-73009_A20190515.nc")
files19_L2_9 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/16/L2_0-20000-0-73009_A20190516.nc")
files19_L2_10 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/18/L2_0-20000-0-73009_A20190518.nc")
files19_L2_11 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/20/L2_0-20000-0-73009_A20190520.nc")
files19_L2_12 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/21/L2_0-20000-0-73009_A20190521.nc")
files19_L2_13 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/23/L2_0-20000-0-73009_A20190523.nc")
files19_L2_14 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/24/L2_0-20000-0-73009_A20190524.nc")
files19_L2_15 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/05/26/L2_0-20000-0-73009_A20190526.nc")
#%%

#2018

files18_L2_1 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/01/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_2 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/02/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_3 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/05/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_4 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/06/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
#%%
files18_L2_5 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/05/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_6 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/08/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_7 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/09/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
#%%
files18_L2_8 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/10/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_9 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/12/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_10 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/13/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
#%%
files18_L2_11 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/19/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_12 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/20/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_13 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/21/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_14 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/23/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
#%%
files18_L2_15= xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/26/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_16 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/27/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_17 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/29/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
files18_L2_18 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/30/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')


#2017
files17_L2_1 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2017/05/01/L2_0-20000-0-73009_A20170501.nc")





#%%
#files_L2 = nc.MFDataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/05/01/L2_73009_*.nc")
#beta_att_test = files_test.variables['attenuated_backscatter_0'][:]
b = files_test2.variables['beta_raw']

# Initialize 2020 variables ---------------------------------------

# L2 attenuated backscatter: 
beta_att_test = files_L2.variables['attenuated_backscatter_0'][:]
b1 = files_L2_1.variables['attenuated_backscatter_0'][:]
b2 = files_L2_2.variables['attenuated_backscatter_0'][:]
b3 = files_L2_3.variables['attenuated_backscatter_0'][:]
b4 = files_L2_4.variables['attenuated_backscatter_0'][:]
b5 = files_L2_5.variables['attenuated_backscatter_0'][:]
b6 = files_L2_6.variables['attenuated_backscatter_0'][:]
b7 = files_L2_7.variables['attenuated_backscatter_0'][:]
b8 = files_L2_8.variables['attenuated_backscatter_0'][:]
b9 = files_L2_9.variables['attenuated_backscatter_0'][:]

b10 = files_L2_10.variables['attenuated_backscatter_0'][:]
b11 = files_L2_11.variables['attenuated_backscatter_0'][:]
b12 = files_L2_12.variables['attenuated_backscatter_0'][:]
b13 = files_L2_13.variables['attenuated_backscatter_0'][:]
b14 = files_L2_14.variables['attenuated_backscatter_0'][:]
b15 = files_L2_15.variables['attenuated_backscatter_0'][:]
b16 = files_L2_16.variables['attenuated_backscatter_0'][:]
b17 = files_L2_17.variables['attenuated_backscatter_0'][:]
b18 = files_L2_18.variables['attenuated_backscatter_0'][:]
b19 = files_L2_19.variables['attenuated_backscatter_0'][:]
b20 = files_L2_20.variables['attenuated_backscatter_0'][:]
b21 = files_L2_21.variables['attenuated_backscatter_0'][:]
b22 = files_L2_22.variables['attenuated_backscatter_0'][:]
b23 = files_L2_23.variables['attenuated_backscatter_0'][:]


#L2 total cloud cover (octas)
octa20_1 = files_L2_1.variables['cloud_amount'][:]
octa20_2 = files_L2_2.variables['cloud_amount'][:]
octa20_3 = files_L2_3.variables['cloud_amount'][:]
octa20_4 = files_L2_4.variables['cloud_amount'][:]
octa20_5 = files_L2_5.variables['cloud_amount'][:]
octa20_6 = files_L2_6.variables['cloud_amount'][:]
octa20_7 = files_L2_7.variables['cloud_amount'][:]
octa20_8 = files_L2_8.variables['cloud_amount'][:]
octa20_9 = files_L2_9.variables['cloud_amount'][:]

octa20_10 = files_L2_10.variables['cloud_amount'][:]
octa20_11 = files_L2_11.variables['cloud_amount'][:]
octa20_12 = files_L2_12.variables['cloud_amount'][:]
octa20_13 = files_L2_13.variables['cloud_amount'][:]
octa20_14 = files_L2_14.variables['cloud_amount'][:]
octa20_15 = files_L2_15.variables['cloud_amount'][:]
octa20_16 = files_L2_16.variables['cloud_amount'][:]
octa20_17 = files_L2_17.variables['cloud_amount'][:]
octa20_18 = files_L2_18.variables['cloud_amount'][:]
octa20_19 = files_L2_19.variables['cloud_amount'][:]
octa20_20 = files_L2_20.variables['cloud_amount'][:]
octa20_21 = files_L2_21.variables['cloud_amount'][:]
octa20_22 = files_L2_22.variables['cloud_amount'][:]
octa20_23 = files_L2_23.variables['cloud_amount'][:]


#%%
#Initialize 2019 variables ----------------------------------------

#L2 attenuated backscatter
c1 = files19_L2_1.variables['attenuated_backscatter_0'][:]
c2 = files19_L2_2.variables['attenuated_backscatter_0'][:]
#c3 = files19_L2_3.variables['attenuated_backscatter_0'][:]
#c4 = files19_L2_4.variables['attenuated_backscatter_0'][:]
c5 = files19_L2_5.variables['attenuated_backscatter_0'][:]
#c6 = files19_L2_6.variables['attenuated_backscatter_0'][:]
c7 = files19_L2_7.variables['attenuated_backscatter_0'][:]
c8 = files19_L2_8.variables['attenuated_backscatter_0'][:]
c9 = files19_L2_9.variables['attenuated_backscatter_0'][:]

c10 = files19_L2_10.variables['attenuated_backscatter_0'][:]
c11 = files19_L2_11.variables['attenuated_backscatter_0'][:]
c12 = files19_L2_12.variables['attenuated_backscatter_0'][:]
c13 = files19_L2_13.variables['attenuated_backscatter_0'][:]
c14 = files19_L2_14.variables['attenuated_backscatter_0'][:]
c15 = files19_L2_15.variables['attenuated_backscatter_0'][:]
c16 = files19_L2_16.variables['attenuated_backscatter_0'][:]
c17 = files19_L2_17.variables['attenuated_backscatter_0'][:]
c18 = files19_L2_18.variables['attenuated_backscatter_0'][:]
c19 = files19_L2_19.variables['attenuated_backscatter_0'][:]
c20 = files19_L2_20.variables['attenuated_backscatter_0'][:]
c21 = files19_L2_21.variables['attenuated_backscatter_0'][:]
c22 = files19_L2_22.variables['attenuated_backscatter_0'][:]

#L2 total cloud cover, octas
octa19_1 = files19_L2_1.variables['cloud_amount'][:]
octa19_2 = files19_L2_2.variables['cloud_amount'][:]
octa19_3 = files19_L2_3.variables['cloud_amount'][:]
octa19_4 = files19_L2_4.variables['cloud_amount'][:]
octa19_5 = files19_L2_5.variables['cloud_amount'][:]
octa19_6 = files19_L2_6.variables['cloud_amount'][:]
octa19_7 = files19_L2_7.variables['cloud_amount'][:]
octa19_8 = files19_L2_8.variables['cloud_amount'][:]
octa19_9 = files19_L2_9.variables['cloud_amount'][:]
octa19_10 = files19_L2_10.variables['cloud_amount'][:]
octa19_11 = files19_L2_11.variables['cloud_amount'][:]
octa19_12 = files19_L2_12.variables['cloud_amount'][:]
octa19_13 = files19_L2_13.variables['cloud_amount'][:]
octa19_14 = files19_L2_14.variables['cloud_amount'][:]
octa19_15 = files19_L2_15.variables['cloud_amount'][:]
octa19_16 = files19_L2_16.variables['cloud_amount'][:]
octa19_17 = files19_L2_17.variables['cloud_amount'][:]
octa19_18 = files19_L2_18.variables['cloud_amount'][:]
octa19_19 = files19_L2_19.variables['cloud_amount'][:]
octa19_20 = files19_L2_20.variables['cloud_amount'][:]
octa19_21 = files19_L2_21.variables['cloud_amount'][:]
octa19_22 = files19_L2_22.variables['cloud_amount'][:]

#%%
#Initialize 2018 variables ----------------------------------------

#L2 attenuated backscatter
d1 = np.array(files18_L2_1.variables['attenuated_backscatter_0'][:])
d2 = np.array(files18_L2_2.variables['attenuated_backscatter_0'][:])
d3 = np.array(files18_L2_3.variables['attenuated_backscatter_0'][:])
d4 = np.array(files18_L2_4.variables['attenuated_backscatter_0'][:])
d5 = np.array(files18_L2_5.variables['attenuated_backscatter_0'][:])
d6 = np.array(files18_L2_6.variables['attenuated_backscatter_0'][:])
d7 = np.array(files18_L2_7.variables['attenuated_backscatter_0'][:])
d8 = np.array(files18_L2_8.variables['attenuated_backscatter_0'][:])
d9 = np.array(files18_L2_9.variables['attenuated_backscatter_0'][:])
d10 = np.array(files18_L2_10.variables['attenuated_backscatter_0'][:])
d11 = np.array(files18_L2_11.variables['attenuated_backscatter_0'][:])
d12 = np.array(files18_L2_12.variables['attenuated_backscatter_0'][:])
d13 = np.array(files18_L2_13.variables['attenuated_backscatter_0'][:])
d14 = np.array(files18_L2_14.variables['attenuated_backscatter_0'][:])
d15 = np.array(files18_L2_15.variables['attenuated_backscatter_0'][:])
d16 = np.array(files18_L2_16.variables['attenuated_backscatter_0'][:])
d17 = np.array(files18_L2_17.variables['attenuated_backscatter_0'][:])
d18 = np.array(files18_L2_18.variables['attenuated_backscatter_0'][:])


#L2 total cloud amount in octas
octas18_1 = np.array(files18_L2_1.variables['cloud_amount'][:])
octas18_2 = np.array(files18_L2_2.variables['cloud_amount'][:])
octas18_3 = np.array(files18_L2_3.variables['cloud_amount'][:])
octas18_4 = np.array(files18_L2_4.variables['cloud_amount'][:])
octas18_5 = np.array(files18_L2_5.variables['cloud_amount'][:])
octas18_6 = np.array(files18_L2_6.variables['cloud_amount'][:])
octas18_7 = np.array(files18_L2_7.variables['cloud_amount'][:])
octas18_8 = np.array(files18_L2_8.variables['cloud_amount'][:])
octas18_9 = np.array(files18_L2_9.variables['cloud_amount'][:])
octas18_10 = np.array(files18_L2_10.variables['cloud_amount'][:])
octas18_11 = np.array(files18_L2_11.variables['cloud_amount'][:])
octas18_12 = np.array(files18_L2_12.variables['cloud_amount'][:])
octas18_13 = np.array(files18_L2_13.variables['cloud_amount'][:])
octas18_14 = np.array(files18_L2_14.variables['cloud_amount'][:])
octas18_15 = np.array(files18_L2_15.variables['cloud_amount'][:])
octas18_16 = np.array(files18_L2_16.variables['cloud_amount'][:])
octas18_17 = np.array(files18_L2_17.variables['cloud_amount'][:])
octas18_18 = np.array(files18_L2_18.variables['cloud_amount'][:])

#%%

#Initialize 2017 variables -----------------------------------------

e1 = files17_L2_1.variables['attenuated_backscatter_0'][:]
e2 = files17_L2_2.variables['attenuated_backscatter_0'][:]
e3 = files17_L2_3.variables['attenuated_backscatter_0'][:]
e4 = files17_L2_4.variables['attenuated_backscatter_0'][:]
e5 = files17_L2_5.variables['attenuated_backscatter_0'][:]
e6 = files17_L2_7.variables['attenuated_backscatter_0'][:]
e7 = files17_L2_8.variables['attenuated_backscatter_0'][:]
e8 = files17_L2_9.variables['attenuated_backscatter_0'][:]
e9 = files17_L2_10.variables['attenuated_backscatter_0'][:]

e10 = files17_L2_11.variables['attenuated_backscatter_0'][:]
e11 = files17_L2_12.variables['attenuated_backscatter_0'][:]
e12 = files17_L2_13.variables['attenuated_backscatter_0'][:]
e13 = files17_L2_14.variables['attenuated_backscatter_0'][:]
e14 = files17_L2_15.variables['attenuated_backscatter_0'][:]
e15 = files17_L2_16.variables['attenuated_backscatter_0'][:]
e16 = files17_L2_17.variables['attenuated_backscatter_0'][:]
e17 = files17_L2_18.variables['attenuated_backscatter_0'][:]

octas17_1 = files17_L2_1.variables['cloud_amount'][:]
octas17_2 = files17_L2_2.variables['cloud_amount'][:]
octas17_3 = files17_L2_3.variables['cloud_amount'][:]
octas17_4 = files17_L2_4.variables['cloud_amount'][:]
octas17_5 = files17_L2_5.variables['cloud_amount'][:]
octas17_6 = files17_L2_6.variables['cloud_amount'][:]
octas17_7 = files17_L2_7.variables['cloud_amount'][:]
octas17_8 = files17_L2_8.variables['cloud_amount'][:]
octas17_9 = files17_L2_9.variables['cloud_amount'][:]
octas17_10 = files17_L2_10.variables['cloud_amount'][:]
octas17_11 = files17_L2_11.variables['cloud_amount'][:]
octas17_12 = files17_L2_12.variables['cloud_amount'][:]
octas17_13 = files17_L2_13.variables['cloud_amount'][:]
octas17_14 = files17_L2_14.variables['cloud_amount'][:]
octas17_15 = files17_L2_15.variables['cloud_amount'][:]
octas17_16 = files17_L2_16.variables['cloud_amount'][:]
octas17_17 = files17_L2_17.variables['cloud_amount'][:]
#%%


range_L2 = files_L2_1.variables['altitude'][:]
range = files.variables['range'][:]
#%%

#----------------------CHECK CLEAR SKY HERE--------------------
files_L2 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/01/L2_0-20000-0-73009_A20200401.nc")

files = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/06/17/20190617_YXU-Cronyn_CHM160155_*.nc') 

#files = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/05/01/20170501_YXU-Cronyn_CHM160155_000.nc")


clouds = files_L2.variables['cloud_base_height'][:]
#visib = files_L2.variables['vertical_visibility'][:]
max_height = files.variables['mxd'][:]
vor = files.variables['vor'][:]
base_cloud = files.variables['bcc'][:]

total_cloud = files.variables['tcc'][:]

range_beta_raw = files.variables['range']
cloud_height = files.variables['cbh'][:]
pbl = files.variables['pbl'][:]
base = files.variables['base'][:]
time_raw = files.variables['time'][:]

cloud_height_L2 = files_L2.variables['cloud_base_height'][:]


customdate = datetime.datetime(year=1905, month=1, day=1, hour=0,second=0)
realtime = [ customdate + datetime.timedelta(seconds=i) for i in (time_raw)]

utc = np.array([f.strftime('%H:%M') for f in (realtime)])
print(utc)



print(np.shape(utc))
#utc = []
#for l in realtime:
#    ff = l.strftime('%H:%M')
#    utc.append(ff)

#print(utc)

#Plotting for the reasons of quality checking and deterimining
# whether it is clear sky or not

#Plot the visibility
plt.plot(realtime, vor)
plt.title('Vertical optical visibility: 2019/05/05')
plt.xlabel('Time in seconds after 00:00')
plt.ylabel('Height (m)')
plt.show()

#clear = np.where(base_cloud == 0)[0]
#plt.scatter(time_raw-(3.668*10**9), total_cloud)

#Plot the total clour cover
plt.scatter(realtime, total_cloud)
plt.title('Amount of cloud cover in octas: 2020/05/03')
plt.xlabel('Time in seconds after 00:00')
plt.ylabel('Octa (0 = clear sky, 8 = total cloud cover)')
plt.ylim(0, 8)
plt.show()

#Plot the max detection height
plt.plot(realtime, max_height)
plt.title('Maximum detection height: 2019/05/05')
plt.xlabel('Time in seconds after 00:00')
plt.ylabel('Height (m)')
plt.show()

max_height = np.array(max_height)
#print(max_height)
print(np.max(max_height))
print(np.min(max_height))
print(np.mean(max_height))

# %%
# Make night averaged, normalized plot in a similar fashion to the 
# way that it is created for the L0 data. 
# The further corrections done to beta_raw to obtain beta_att should allow 
# for the same manipulation to normalize and average 

#-----------------------OVERLAP---------------------------------

#The overlap file is an array of 1024 values. 
# Resize this array to be 511 values in order to multiply with beta_att


overlap = np.loadtxt('data.txt',dtype = float)
print(overlap)
overlap_0 = np.resize(overlap,[511,2])
overlap_atten = overlap_0[:, 0]


#%%----------------------------------------------
# Calcultate all profiles for 2020:

#Determine exact times at which there is clear sky rating, i.e. octa = 0
time_L2 = files_L2_1.variables['time'][:]


customdate_L2 = datetime.datetime(year=1970, month=1, day=1, hour=0)
realtime_L2 = [ customdate_L2 + datetime.timedelta(days=i) for i in (time_L2)]

utc_L2 = np.array([f.strftime('%H:%M') for f in (realtime_L2)])

total_cloud_L2 = files_L2.variables['cloud_amount'][:]

for l in np.where(total_cloud_L2 == 0):
  print(utc_L2[l])
#%%
a = int(len(b1)/2)
print(a)

range_L2 = files_L2_1.variables['altitude']

##------
timespan1 = np.where(utc_L2 == '21:00')
timespan2 = np.where(utc_L2 == '22:00' )

#d_night_test1 = b1[:,248:250] 
d_night_test1 = b1[:,int(timespan1[0]):int(timespan2[0])] 

d_nightsum_test1 = np.mean(d_night_test1,1)

noise1 = np.mean(d_nightsum_test1[333:511])
print(noise1)
d_nightsum_test1 = d_nightsum_test1 - noise1

for f in overlap_atten:
  dens_overlap_test1 = d_nightsum_test1 * overlap_atten
#If overlap is needed:
[x1,y1] = coadd(dens_overlap_test1,range_L2,5)

##------

timespan1 = np.where(utc_L2 == '23:00')
timespan2 = np.where(utc_L2 == '23:55' )
print(timespan1, timespan2)


d_night_test2 = b2[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test2 = np.mean(d_night_test2,1)

noise2 = np.mean(d_nightsum_test2[333:511])
print(noise2)
d_nightsum_test2 = d_nightsum_test2 + noise2

for f in overlap_atten:
  dens_overlap_test2 = d_nightsum_test2 * overlap_atten
#If overlap is needed:
[x2,y2] = coadd(dens_overlap_test2,range_L2,5)

#-------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '05:00' )

a = int(len(beta_att_test)/2)
print(a)
d_night_test3 = b3[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test3 = np.mean(d_night_test3,1)

noise3 = np.mean(d_nightsum_test3[333:511])
print(noise3)
d_nightsum_test3 = d_nightsum_test3 + noise3

for f in overlap_atten:
  dens_overlap_test3 = d_nightsum_test3 * overlap_atten
#If overlap is needed:
[x3,y3] = coadd(dens_overlap_test3,range_L2,5)

##------

timespan1 = np.where(utc_L2 == '22:00')
timespan2 = np.where(utc_L2 == '23:55' )

a = int(len(beta_att_test)/2)
print(a)
d_night_test4 = b4[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test4 = np.mean(d_night_test4,1)

noise4 = np.mean(d_nightsum_test4[400:511])
print(noise4)
d_nightsum_test4 = d_nightsum_test4 + noise4

for f in overlap_atten:
  dens_overlap_test4 = d_nightsum_test4 * overlap_atten
#If overlap is needed:
[x4,y4] = coadd(dens_overlap_test4,range_L2,5)

##------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '01:00' )

a = int(len(beta_att_test)/2)
print(a)
d_night_test5 = b5[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test5 = np.mean(d_night_test5,1)

noise5 = np.mean(d_nightsum_test5[333:511])
print(noise5)
d_nightsum_test5 = d_nightsum_test5 + noise5

for f in overlap_atten:
  dens_overlap_test5 = d_nightsum_test5 * overlap_atten
#If overlap is needed:
[x5,y5] = coadd(dens_overlap_test5,range_L2,5)

#-------

timespan1 = np.where(utc_L2 == '01:30')
timespan2 = np.where(utc_L2 == '03:30' )

a = int(len(beta_att_test)/2)
print(a)
d_night_test6 = b6[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test6 = np.mean(d_night_test6,1)

noise6 = np.mean(d_nightsum_test6[333:511])
print(noise6)
d_nightsum_test6 = d_nightsum_test6 + noise6

for f in overlap_atten:
  dens_overlap_test6 = d_nightsum_test6 * overlap_atten
#If overlap is needed:
[x6,y6] = coadd(dens_overlap_test6,range_L2,5)

#--------

timespan1 = np.where(utc_L2 == '05:30')
timespan2 = np.where(utc_L2 == '07:00' )

d_night_test7 = b7[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test7 = np.mean(d_night_test7,1)

noise7 = np.mean(d_nightsum_test7[333:511])
print(noise7)
d_nightsum_test7 = d_nightsum_test7 + noise7

for f in overlap_atten:
  dens_overlap_test7 = d_nightsum_test7 * overlap_atten
#If overlap is needed:
[x7,y7] = coadd(dens_overlap_test7,range_L2,5)

#-----

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '03:00' )

d_night_test8 = b8[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test8 = np.mean(d_night_test8,1)

noise8 = np.mean(d_nightsum_test8[333:511])
print(noise8)
d_nightsum_test8 = d_nightsum_test8 + noise8

for f in overlap_atten:
  dens_overlap_test8 = d_nightsum_test8 * overlap_atten
#If overlap is needed:
[x8,y8] = coadd(dens_overlap_test8,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '03:00' )

d_night_test9 = b9[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test9 = np.mean(d_night_test9,1)

noise9 = np.mean(d_nightsum_test9[333:511])
print(noise9)
d_nightsum_test9 = d_nightsum_test9 + noise9

for f in overlap_atten:
  dens_overlap_test9 = d_nightsum_test9 * overlap_atten
#If overlap is needed:
[x9,y9] = coadd(dens_overlap_test9,range_L2,5)


#------


timespan1 = np.where(utc_L2 == '19:00')
timespan2 = np.where(utc_L2 == '21:00' )

d_night_test10 = b10[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test10 = np.mean(d_night_test10,1)

noise10 = np.mean(d_nightsum_test10[333:511])
print(noise10)
d_nightsum_test10 = d_nightsum_test10 + noise10

for f in overlap_atten:
  dens_overlap_test10 = d_nightsum_test10 * overlap_atten
#If overlap is needed:
[x10,y10] = coadd(dens_overlap_test10,range_L2,5)


#------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '03:00' )

d_night_test11 = b11[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test11 = np.mean(d_night_test11,1)

noise11 = np.mean(d_nightsum_test11[333:511])
print(noise10)
d_nightsum_test11 = d_nightsum_test11 + noise11

for f in overlap_atten:
  dens_overlap_test11 = d_nightsum_test11 * overlap_atten
#If overlap is needed:
[x11,y11] = coadd(dens_overlap_test11,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '06:00')
timespan2 = np.where(utc_L2 == '09:00' )

d_night_test12 = b12[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test12 = np.mean(d_night_test12,1)

noise12 = np.mean(d_nightsum_test12[333:511])
print(noise10)
d_nightsum_test12 = d_nightsum_test12 + noise12

for f in overlap_atten:
  dens_overlap_test12 = d_nightsum_test12 * overlap_atten
#If overlap is needed:
[x12,y12] = coadd(dens_overlap_test12,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '20:00')
timespan2 = np.where(utc_L2 == '22:00' )

d_night_test13 = b13[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test13 = np.mean(d_night_test13,1)

noise13 = np.mean(d_nightsum_test13[440:511])
print(noise13)
d_nightsum_test13 = d_nightsum_test13 - noise13

for f in overlap_atten:
  dens_overlap_test13 = d_nightsum_test13 * overlap_atten
#If overlap is needed:
[x13,y13] = coadd(dens_overlap_test13,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '21:00')
timespan2 = np.where(utc_L2 == '22:00' )

d_night_test14 = b14[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test14 = np.mean(d_night_test14,1)

noise14 = np.mean(d_nightsum_test14[333:511])
print(noise10)
d_nightsum_test14 = d_nightsum_test14 + noise14

for f in overlap_atten:
  dens_overlap_test14 = d_nightsum_test14 * overlap_atten
#If overlap is needed:
[x14,y14] = coadd(dens_overlap_test14,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '18:00')
timespan2 = np.where(utc_L2 == '21:00' )

d_night_test15 = b15[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test15 = np.mean(d_night_test15,1)

noise15 = np.mean(d_nightsum_test15[333:511])
print(noise10)
d_nightsum_test15 = d_nightsum_test15 + noise15

for f in overlap_atten:
  dens_overlap_test15 = d_nightsum_test15 * overlap_atten
#If overlap is needed:
[x15,y15] = coadd(dens_overlap_test15,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '03:01')
timespan2 = np.where(utc_L2 == '06:00' )

d_night_test16 = b16[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test16 = np.mean(d_night_test16,1)

noise16 = np.mean(d_nightsum_test16[333:511])
print(noise10)
d_nightsum_test16 = d_nightsum_test16 + noise16

for f in overlap_atten:
  dens_overlap_test16 = d_nightsum_test16 * overlap_atten
#If overlap is needed:
[x16,y16] = coadd(dens_overlap_test16,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '03:00')
timespan2 = np.where(utc_L2 == '04:00' )

d_night_test17 = b17[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test17 = np.mean(d_night_test17,1)

noise17 = np.mean(d_nightsum_test17[333:511])
print(noise10)
d_nightsum_test17 = d_nightsum_test17 + noise17

for f in overlap_atten:
  dens_overlap_test17 = d_nightsum_test17 * overlap_atten
#If overlap is needed:
[x17,y17] = coadd(dens_overlap_test17,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '05:00')
timespan2 = np.where(utc_L2 == '08:00' )

d_night_test18 = b18[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test18 = np.mean(d_night_test18,1)

noise18 = np.mean(d_nightsum_test18[333:511])
print(noise10)
d_nightsum_test18 = d_nightsum_test18 + noise18

for f in overlap_atten:
  dens_overlap_test18 = d_nightsum_test18 * overlap_atten
#If overlap is needed:
[x18,y18] = coadd(dens_overlap_test18,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '00:30')
timespan2 = np.where(utc_L2 == '03:00' )

d_night_test19 = b19[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test19 = np.mean(d_night_test19,1)

noise19 = np.mean(d_nightsum_test19[333:511])
print(noise10)
d_nightsum_test19 = d_nightsum_test19 + noise19

for f in overlap_atten:
  dens_overlap_test19 = d_nightsum_test19 * overlap_atten
#If overlap is needed:
[x19,y19] = coadd(dens_overlap_test19,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '06:00')
timespan2 = np.where(utc_L2 == '09:00' )

d_night_test20 = b20[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test20 = np.mean(d_night_test20,1)

noise20 = np.mean(d_nightsum_test20[333:511])
print(noise10)
d_nightsum_test20 = d_nightsum_test20 + noise20

for f in overlap_atten:
  dens_overlap_test20 = d_nightsum_test20 * overlap_atten
#If overlap is needed:
[x20,y20] = coadd(dens_overlap_test20,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '21:01')
timespan2 = np.where(utc_L2 == '23:56' )

d_night_test21 = b21[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test21 = np.mean(d_night_test21,1)

noise21 = np.mean(d_nightsum_test21[333:511])
d_nightsum_test21 = d_nightsum_test21 + noise21

for f in overlap_atten:
  dens_overlap_test21 = d_nightsum_test21 * overlap_atten
#If overlap is needed:
[x21,y21] = coadd(dens_overlap_test21,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '08:00' )

d_night_test22 = b22[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test22 = np.mean(d_night_test22,1)

noise22 = np.mean(d_nightsum_test22[333:511])

d_nightsum_test22 = d_nightsum_test22 - noise22

for f in overlap_atten:
  dens_overlap_test22 = d_nightsum_test22 * overlap_atten
#If overlap is needed:
[x22,y22] = coadd(dens_overlap_test22,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '09:00')
timespan2 = np.where(utc_L2 == '13:01' )

d_night_test23 = b23[:,int(timespan1[0]):int(timespan2[0])] 
d_nightsum_test23 = np.mean(d_night_test23,1)

noise23 = np.mean(d_nightsum_test23[333:511])
d_nightsum_test23 = d_nightsum_test23 + noise23

for f in overlap_atten:
  dens_overlap_test23 = d_nightsum_test23 * overlap_atten
#If overlap is needed:
[x23,y23] = coadd(dens_overlap_test23,range_L2,5)



plt.semilogx(x22[0:70:], y1[0:70])
plt.xlim([0.001,20])
plt.ylim([0, 10000])

#INVESTIGATE:
#x11 big spike, 
# X12 is higher overall average, 
# exclude x13, 
#x17 4000m spike
#x18 spikes
#x22 soike
#x23


#excluse x11,   x17

#x5, x11

# x3 why NAN, x11 spikes, x13 not correct profile, x15 spikes, x17 spike, x18, x22, x23
#%%

#%%------------------------------
# Calcultate all profiles for 2019:

#------

timespan1 = np.where(utc_L2 == '13:00')
timespan2 = np.where(utc_L2 == '15:00' )

a = int(len(beta_att_test)/2)
print(a)
d19_night_test1 = c1[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test1 = np.mean(d19_night_test1,1)

noise11 = np.mean(d19_nightsum_test1[333:511])
print(noise11)
d19_nightsum_test1 = d19_nightsum_test1 + noise11

for f in overlap_atten:
  dens19_overlap_test1 = d19_nightsum_test1 * overlap_atten
#If overlap is needed:
[xx1,yy1] = coadd(dens19_overlap_test1,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '14:00')
timespan2 = np.where(utc_L2 == '17:00' )

d19_night_test2 = c2[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test2 = np.mean(d19_night_test2,1)

noise22 = np.mean(d19_nightsum_test2[333:511])
print(noise22)
d19_nightsum_test2 = d19_nightsum_test2 + noise22

for f in overlap_atten:
  dens19_overlap_test2 = d19_nightsum_test2 * overlap_atten
#If overlap is needed:
[xx2,yy2] = coadd(dens19_overlap_test2,range_L2,5)

#------

#-----


#--------

timespan1 = np.where(utc_L2 == '03:00')
timespan2 = np.where(utc_L2 == '06:00' )

a = int(len(beta_att_test)/2)
print(a)
d19_night_test5 = c5[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test5 = np.mean(d19_night_test5,1)

noise55 = np.mean(d19_nightsum_test5[333:511])
print(noise55)
d19_nightsum_test5 = d19_nightsum_test5 + noise55

for f in overlap_atten:
  dens19_overlap_test5 = d19_nightsum_test5 * overlap_atten
#If overlap is needed:
[xx5,yy5] = coadd(dens19_overlap_test5,range_L2,5)

#-------



#--------

timespan1 = np.where(utc_L2 == '22:00')
timespan2 = np.where(utc_L2 == '23:00' )

d19_night_test7 = c7[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test7 = np.mean(d19_night_test7,1)

noise77 = np.mean(d19_nightsum_test7[333:511])
print(noise77)
d19_nightsum_test7 = d19_nightsum_test7 + noise77

for f in overlap_atten:
  dens19_overlap_test7 = d19_nightsum_test7 * overlap_atten
#If overlap is needed:
[xx7,yy7] = coadd(dens19_overlap_test7,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '15:00')
timespan2 = np.where(utc_L2 == '16:30' )

d19_night_test8 = c8[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test8 = np.mean(d19_night_test8,1)

noise88 = np.mean(d19_nightsum_test8[333:511])
print(noise88)
d19_nightsum_test8 = d19_nightsum_test8 + noise88

for f in overlap_atten:
  dens19_overlap_test8 = d19_nightsum_test8 * overlap_atten
#If overlap is needed:
[xx8,yy8] = coadd(dens19_overlap_test8,range_L2,5)


#-------

timespan1 = np.where(utc_L2 == '03:00')
timespan2 = np.where(utc_L2 == '06:00' )

d19_night_test9 = c9[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test9 = np.mean(d19_night_test9,1)

noise99 = np.mean(d19_nightsum_test9[333:511])
d19_nightsum_test9 = d19_nightsum_test9 + noise99

for f in overlap_atten:
  dens19_overlap_test9 = d19_nightsum_test9 * overlap_atten
#If overlap is needed:
[xx9,yy9] = coadd(dens19_overlap_test9,range_L2,5)

#-----

timespan1 = np.where(utc_L2 == '18:00')
timespan2 = np.where(utc_L2 == '21:00' )

d19_night_test10 = c10[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test10 = np.mean(d19_night_test10,1)

noise99 = np.mean(d19_nightsum_test10[333:511])
d19_nightsum_test10 = d19_nightsum_test10 + noise99

for f in overlap_atten:
  dens19_overlap_test10 = d19_nightsum_test10 * overlap_atten
#If overlap is needed:
[xx10,yy10] = coadd(dens19_overlap_test10,range_L2,5)

#--------

timespan1 = np.where(utc_L2 == '07:00')
timespan2 = np.where(utc_L2 == '08:00' )

d19_night_test11 = c11[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test11 = np.mean(d19_night_test11,1)

noise99 = np.mean(d19_nightsum_test11[333:511])
d19_nightsum_test11 = d19_nightsum_test11 + noise99

for f in overlap_atten:
  dens19_overlap_test11 = d19_nightsum_test11 * overlap_atten
#If overlap is needed:
[xx11,yy11] = coadd(dens19_overlap_test11,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '05:00')
timespan2 = np.where(utc_L2 == '07:00' )

d19_night_test12 = c12[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test12 = np.mean(d19_night_test12,1)

noise99 = np.mean(d19_nightsum_test12[333:511])
d19_nightsum_test12 = d19_nightsum_test12 + noise99

for f in overlap_atten:
  dens19_overlap_test12 = d19_nightsum_test12 * overlap_atten
#If overlap is needed:
[xx12,yy12] = coadd(dens19_overlap_test12,range_L2,5)

#--------hhhHHHHHHHHHHHHHHH

timespan1 = np.where(utc_L2 == '03:00')
timespan2 = np.where(utc_L2 == '04:00' )

d19_night_test13 = c13[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test13 = np.mean(d19_night_test13,1)

noise99 = np.mean(d19_nightsum_test13[333:511])
d19_nightsum_test13 = d19_nightsum_test13 + noise99

for f in overlap_atten:
  dens19_overlap_test13 = d19_nightsum_test13 * overlap_atten
#If overlap is needed:
[xx13,yy13] = coadd(dens19_overlap_test13,range_L2,5)

#---------

timespan1 = np.where(utc_L2 == '05:00')
timespan2 = np.where(utc_L2 == '06:00' )

d19_night_test14 = c14[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test14 = np.mean(d19_night_test14,1)

noise99 = np.mean(d19_nightsum_test14[333:511])
d19_nightsum_test14 = d19_nightsum_test14 + noise99

for f in overlap_atten:
  dens19_overlap_test14 = d19_nightsum_test14 * overlap_atten
#If overlap is needed:
[xx14,yy14] = coadd(dens19_overlap_test14,range_L2,5)

#-------

timespan1 = np.where(utc_L2 == '03:00')
timespan2 = np.where(utc_L2 == '09:00' )

d19_night_test15 = c15[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test15 = np.mean(d19_night_test15,1)

noise99 = np.mean(d19_nightsum_test15[333:511])
d19_nightsum_test15 = d19_nightsum_test15 + noise99

for f in overlap_atten:
  dens19_overlap_test15 = d19_nightsum_test15 * overlap_atten
#If overlap is needed:
[xx15,yy15] = coadd(dens19_overlap_test15,range_L2,5)

#-----
timespan1 = np.where(utc_L2 == '22:00')
timespan2 = np.where(utc_L2 == '23:55' )

d19_night_test16 = c16[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test16 = np.mean(d19_night_test16,1)

noise99 = np.mean(d19_nightsum_test16[333:511])
d19_nightsum_test16 = d19_nightsum_test16 + noise99

for f in overlap_atten:
  dens19_overlap_test16 = d19_nightsum_test16 * overlap_atten
#If overlap is needed:
[xx16,yy16] = coadd(dens19_overlap_test16,range_L2,5)

#-----
timespan1 = np.where(utc_L2 == '03:00')
timespan2 = np.where(utc_L2 == '06:00' )

d19_night_test17 = c17[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test17 = np.mean(d19_night_test17,1)

noise99 = np.mean(d19_nightsum_test17[333:511])
d19_nightsum_test17 = d19_nightsum_test17 + noise99

for f in overlap_atten:
  dens19_overlap_test17 = d19_nightsum_test17 * overlap_atten
#If overlap is needed:
[xx17,yy17] = coadd(dens19_overlap_test17,range_L2,5)

#------
timespan1 = np.where(utc_L2 == '03:00')
timespan2 = np.where(utc_L2 == '03:00' )

d19_night_test18 = c18[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test18 = np.mean(d19_night_test18,1)

noise99 = np.mean(d19_nightsum_test18[333:511])
d19_nightsum_test18 = d19_nightsum_test18 + noise99

for f in overlap_atten:
  dens19_overlap_test18 = d19_nightsum_test18 * overlap_atten
#If overlap is needed:
[xx18,yy18] = coadd(dens19_overlap_test18,range_L2,5)

#-----
timespan1 = np.where(utc_L2 == '03:00')
timespan2 = np.where(utc_L2 == '06:00')

d19_night_test19 = c19[:,int(timespan1[0]):int(timespan2[0])] 
d19_nightsum_test19 = np.mean(d19_night_test19,1)

noise99 = np.mean(d19_nightsum_test19[333:511])
d19_nightsum_test19 = d19_nightsum_test19 + noise99

for f in overlap_atten:
  dens19_overlap_test19 = d19_nightsum_test19 * overlap_atten
#If overlap is needed:
[xx19,yy19] = coadd(dens19_overlap_test19,range_L2,5)



#------
plt.semilogx(xx21[0:102:], yy1)
plt.xlim([0.01,20])
plt.ylim([0, 15000])

#xx12 is strange

#%%
#--------------------------------------------------
#Calculate all profiles for 2018:




#-------

timespan1 = np.where(utc_L2 == '10:30')
timespan2 = np.where(utc_L2 == '11:30')

a = int(len(beta_att_test)/2)
print(a)
d18_night_test1 = d1[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test1 = np.mean(d18_night_test1,1)

noise11 = np.mean(d18_nightsum_test1[333:511])
print(noise11)
d18_nightsum_test1 = d18_nightsum_test1 + noise11

for f in overlap_atten:
  dens18_overlap_test1 = d18_nightsum_test1 * overlap_atten
#If overlap is needed:
[xxx1,yyy1] = coadd(dens18_overlap_test1,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '09:00')
timespan2 = np.where(utc_L2 == '12:00')

d18_night_test2 = d2[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test2 = np.mean(d18_night_test2,1)

noise11 = np.mean(d18_nightsum_test2[333:511])
print(noise11)
d18_nightsum_test2 = d18_nightsum_test2 + noise11

for f in overlap_atten:
  dens18_overlap_test2 = d18_nightsum_test2 * overlap_atten
#If overlap is needed:
[xxx2,yyy2] = coadd(dens18_overlap_test2,range_L2,5)

#-------

timespan1 = np.where(utc_L2 == '05:30')
timespan2 = np.where(utc_L2 == '05:30')

d18_night_test3 = d3[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test3 = np.mean(d18_night_test3,1)

noise11 = np.mean(d18_nightsum_test3[300:511])
print(noise11)
d18_nightsum_test3 = d18_nightsum_test3 - noise11

for f in overlap_atten:
  dens18_overlap_test3 = d18_nightsum_test3 * overlap_atten
#If overlap is needed:
[xxx3,yyy3] = coadd(dens18_overlap_test3,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '01:30')

d18_night_test4 = d4[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test4 = np.mean(d18_night_test4,1)

noise11 = np.mean(d18_nightsum_test4[333:511])
print(noise11)
d18_nightsum_test4 = d18_nightsum_test4 + noise11

for f in overlap_atten:
  dens18_overlap_test4 = d18_nightsum_test4 * overlap_atten
#If overlap is needed:
[xxx4,yyy4] = coadd(dens18_overlap_test4,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '09:00')
timespan2 = np.where(utc_L2 == '11:30')

d18_night_test5 = d5[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test5 = np.mean(d18_night_test5,1)

noise11 = np.mean(d18_nightsum_test5[333:511])
print(noise11)
d18_nightsum_test5 = d18_nightsum_test5 + noise11

for f in overlap_atten:
  dens18_overlap_test5 = d18_nightsum_test5 * overlap_atten
#If overlap is needed:
[xxx5,yyy5] = coadd(dens18_overlap_test5,range_L2,5)

#-------

timespan1 = np.where(utc_L2 == '12:00')
timespan2 = np.where(utc_L2 == '13:00')

d18_night_test6 = d6[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test6 = np.mean(d18_night_test6,1)

noise11 = np.mean(d18_nightsum_test6[333:511])
print(noise11)
d18_nightsum_test6 = d18_nightsum_test6 + noise11

for f in overlap_atten:
  dens18_overlap_test6 = d18_nightsum_test6 * overlap_atten
#If overlap is needed:
[xxx6,yyy6] = coadd(dens18_overlap_test6,range_L2,5)

#-----

timespan1 = np.where(utc_L2 == '22:30')
timespan2 = np.where(utc_L2 == '23:55')

d18_night_test7 = d7[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test7 = np.mean(d18_night_test7,1)

noise11 = np.mean(d18_nightsum_test7[333:511])
print(noise11)
d18_nightsum_test7 = d18_nightsum_test7 + noise11

for f in overlap_atten:
  dens18_overlap_test7 = d18_nightsum_test7 * overlap_atten
#If overlap is needed:
[xxx7,yyy7] = coadd(dens18_overlap_test7,range_L2,5)

#-----

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '02:00')

d18_night_test8 = d8[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test8 = np.mean(d18_night_test8,1)

noise11 = np.mean(d18_nightsum_test8[333:511])
print(noise11)
d18_nightsum_test8 = d18_nightsum_test8 + noise11

for f in overlap_atten:
  dens18_overlap_test8 = d18_nightsum_test8 * overlap_atten
#If overlap is needed:
[xxx8,yyy8] = coadd(dens18_overlap_test8,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '01:30')

d18_night_test9 = d9[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test9 = np.mean(d18_night_test9,1)

noise11 = np.mean(d18_nightsum_test9[333:511])
print(noise11)
d18_nightsum_test9 = d18_nightsum_test9 + noise11

for f in overlap_atten:
  dens18_overlap_test9 = d18_nightsum_test9 * overlap_atten
#If overlap is needed:
[xxx9,yyy9] = coadd(dens18_overlap_test9,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '02:00')
timespan2 = np.where(utc_L2 == '03:00')

d18_night_test10 = d10[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test10 = np.mean(d18_night_test10,1)

noise11 = np.mean(d18_nightsum_test10[333:511])
print(noise11)
d18_nightsum_test10 = d18_nightsum_test10 + noise11

for f in overlap_atten:
  dens18_overlap_test10 = d18_nightsum_test10 * overlap_atten
#If overlap is needed:
[xxx10,yyy10] = coadd(dens18_overlap_test10,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '20:00')
timespan2 = np.where(utc_L2 == '21:00')

d18_night_test11 = d11[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test11 = np.mean(d18_night_test11,1)

noise11 = np.mean(d18_nightsum_test11[333:511])
print(noise11)
d18_nightsum_test11 = d18_nightsum_test11 + noise11

for f in overlap_atten:
  dens18_overlap_test11 = d18_nightsum_test11 * overlap_atten
#If overlap is needed:
[xxx11,yyy11] = coadd(dens18_overlap_test11,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '12:00')

d18_night_test12 = d12[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test12 = np.mean(d18_night_test12,1)

noise11 = np.mean(d18_nightsum_test12[333:511])
print(noise11)
d18_nightsum_test12 = d18_nightsum_test12 + noise11

for f in overlap_atten:
  dens18_overlap_test12 = d18_nightsum_test12 * overlap_atten
#If overlap is needed:
[xxx12,yyy12] = coadd(dens18_overlap_test12,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '05:00')

d18_night_test13 = d13[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test13 = np.mean(d18_night_test13,1)

noise11 = np.mean(d18_nightsum_test13[333:511])
print(noise11)
d18_nightsum_test13 = d18_nightsum_test13 + noise11

for f in overlap_atten:
  dens18_overlap_test13 = d18_nightsum_test13 * overlap_atten
#If overlap is needed:
[xxx13,yyy13] = coadd(dens18_overlap_test13,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '03:00')

d18_night_test14 = d14[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test14 = np.mean(d18_night_test14,1)

noise11 = np.mean(d18_nightsum_test14[333:511])
print(noise11)
d18_nightsum_test14 = d18_nightsum_test14 + noise11

for f in overlap_atten:
  dens18_overlap_test14 = d18_nightsum_test14 * overlap_atten
#If overlap is needed:
[xxx14,yyy14] = coadd(dens18_overlap_test14,range_L2,5)

#--------

timespan1 = np.where(utc_L2 == '06:00')
timespan2 = np.where(utc_L2 == '12:00')

d18_night_test15 = d15[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test15 = np.mean(d18_night_test15,1)

noise11 = np.mean(d18_nightsum_test15[333:511])
print(noise11)
d18_nightsum_test15 = d18_nightsum_test15 + noise11

for f in overlap_atten:
  dens18_overlap_test15 = d18_nightsum_test15 * overlap_atten
#If overlap is needed:
[xxx15,yyy15] = coadd(dens18_overlap_test15,range_L2,5)

#------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '05:00')

d18_night_test16 = d16[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test16 = np.mean(d18_night_test16,1)

noise11 = np.mean(d18_nightsum_test16[333:511])
print(noise11)
d18_nightsum_test16 = d18_nightsum_test16 + noise11

for f in overlap_atten:
  dens18_overlap_test16 = d18_nightsum_test16 * overlap_atten
#If overlap is needed:
[xxx16,yyy16] = coadd(dens18_overlap_test16,range_L2,5)

#-----

timespan1 = np.where(utc_L2 == '15:00')
timespan2 = np.where(utc_L2 == '18:00')

d18_night_test17 = d17[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test17 = np.mean(d18_night_test17,1)

noise11 = np.mean(d18_nightsum_test17[333:511])
print(noise11)
d18_nightsum_test17 = d18_nightsum_test17 + noise11

for f in overlap_atten:
  dens18_overlap_test17 = d18_nightsum_test17 * overlap_atten
#If overlap is needed:
[xxx17,yyy17] = coadd(dens18_overlap_test17,range_L2,5)

#---------

timespan1 = np.where(utc_L2 == '00:00')
timespan2 = np.where(utc_L2 == '06:00')

d18_night_test18 = d18[:,int(timespan1[0]):int(timespan2[0])] 
d18_nightsum_test18 = np.mean(d18_night_test18,1)

noise11 = np.mean(d18_nightsum_test18[333:511])
print(noise11)
d18_nightsum_test18 = d18_nightsum_test18 + noise11

for f in overlap_atten:
  dens18_overlap_test18 = d18_nightsum_test18 * overlap_atten
#If overlap is needed:
[xxx18,yyy18] = coadd(dens18_overlap_test18,range_L2,5)



#To quality check individual averaged plots: 

plt.semilogx(xxx18[0:102:], yy1)
plt.xlim([0.01,20])
plt.ylim([0, 15000])

#%%
#-----------------------------------------------------------
#Calculate profiles for 2017: 


d17_night_test1 = e1[:,156:175] 
d17_nightsum_test1 = np.mean(d17_night_test1,1)

noise11 = np.mean(d17_nightsum_test1[333:511])
print(noise11)
d17_nightsum_test1 = d17_nightsum_test1 + noise11

for f in overlap_atten:
  dens17_overlap_test1 = d17_nightsum_test1 * overlap_atten
#If overlap is needed:
[xxxx1,yyyy1] = coadd(dens17_overlap_test1,range_L2,5)

d17_night_test2 = e2[:,156:175] 
d17_nightsum_test2 = np.mean(d17_night_test2,1)

noise11 = np.mean(d17_nightsum_test2[333:511])
print(noise11)
d17_nightsum_test2= d17_nightsum_test2 + noise11

for f in overlap_atten:
  dens17_overlap_test2 = d17_nightsum_test2 * overlap_atten
#If overlap is needed:
[xxxx2,yyyy2] = coadd(dens17_overlap_test2,range_L2,5)

d17_night_test3 = e3[:,156:175] 
d17_nightsum_test3 = np.mean(d17_night_test3,1)

noise11 = np.mean(d17_nightsum_test3[333:511])
print(noise11)
d17_nightsum_test3 = d17_nightsum_test3 + noise11

for f in overlap_atten:
  dens17_overlap_test3 = d17_nightsum_test3 * overlap_atten
#If overlap is needed:
[xxxx3,yyyy3] = coadd(dens17_overlap_test3,range_L2,5)

d17_night_test4 = e4[:,156:175] 
d17_nightsum_test4 = np.mean(d17_night_test4,1)

noise11 = np.mean(d17_nightsum_test4[333:511])
print(noise11)
d17_nightsum_test4 = d17_nightsum_test4 + noise11

for f in overlap_atten:
  dens17_overlap_test4 = d17_nightsum_test4 * overlap_atten
#If overlap is needed:
[xxxx4,yyyy4] = coadd(dens17_overlap_test4,range_L2,5)

d17_night_test5 = e5[:,156:175] 
d17_nightsum_test5 = np.mean(d17_night_test5,1)

noise11 = np.mean(d17_nightsum_test5[333:511])
print(noise11)
d17_nightsum_test5 = d17_nightsum_test5 + noise11

for f in overlap_atten:
  dens17_overlap_test5 = d17_nightsum_test5 * overlap_atten
#If overlap is needed:
[xxxx5,yyyy5] = coadd(dens17_overlap_test5,range_L2,5)

d17_night_test6 = e6[:,156:175] 
d17_nightsum_test6 = np.mean(d17_night_test6,1)

noise11 = np.mean(d17_nightsum_test6[333:511])
print(noise11)
d17_nightsum_test6 = d17_nightsum_test6 + noise11

for f in overlap_atten:
  dens17_overlap_test6 = d17_nightsum_test6 * overlap_atten
#If overlap is needed:
[xxxx6,yyyy6] = coadd(dens17_overlap_test6,range_L2,5)

d17_night_test7 = e7[:,156:175] 
d17_nightsum_test1 = np.mean(d17_night_test7,1)

noise11 = np.mean(d17_nightsum_test7[333:511])
print(noise11)
d17_nightsum_test7 = d17_nightsum_test7 + noise11

for f in overlap_atten:
  dens17_overlap_test7 = d17_nightsum_test7 * overlap_atten
#If overlap is needed:
[xxxx7,yyyy7] = coadd(dens17_overlap_test7,range_L2,5)

d17_night_test8 = e8[:,156:175] 
d17_nightsum_test8 = np.mean(d17_night_test8,1)

noise11 = np.mean(d17_nightsum_test8[333:511])
print(noise11)
d17_nightsum_test8 = d17_nightsum_test8 + noise11

for f in overlap_atten:
  dens17_overlap_test8 = d17_nightsum_test8 * overlap_atten
#If overlap is needed:
[xxxx8,yyyy8] = coadd(dens17_overlap_test8,range_L2,5)

d17_night_test9 = e9[:,156:175] 
d17_nightsum_test9 = np.mean(d17_night_test9,1)

noise11 = np.mean(d17_nightsum_test9[333:511])
print(noise11)
d17_nightsum_test9 = d17_nightsum_test9 + noise11

for f in overlap_atten:
  dens17_overlap_test9 = d17_nightsum_test9 * overlap_atten
#If overlap is needed:
[xxxx9,yyyy9] = coadd(dens17_overlap_test9,range_L2,5)

d17_night_test10 = e10[:,156:175] 
d17_nightsum_test10 = np.mean(d17_night_test10,1)

noise11 = np.mean(d17_nightsum_test10[333:511])
print(noise11)
d17_nightsum_test10 = d17_nightsum_test10 + noise11

for f in overlap_atten:
  dens17_overlap_test10 = d17_nightsum_test10 * overlap_atten
#If overlap is needed:
[xxxx10,yyyy10] = coadd(dens17_overlap_test10,range_L2,5)

d17_night_test11 = e11[:,156:175] 
d17_nightsum_test11 = np.mean(d17_night_test11,1)

noise11 = np.mean(d17_nightsum_test11[333:511])
print(noise11)
d17_nightsum_test11 = d17_nightsum_test11 + noise11

for f in overlap_atten:
  dens17_overlap_test11 = d17_nightsum_test11 * overlap_atten
#If overlap is needed:
[xxxx11,yyyy11] = coadd(dens17_overlap_test11,range_L2,5)

d17_night_test12 = e1[:,156:175] 
d17_nightsum_test12 = np.mean(d17_night_test12,1)

noise11 = np.mean(d17_nightsum_test12[333:511])
print(noise11)
d17_nightsum_test12 = d17_nightsum_test12 + noise11

for f in overlap_atten:
  dens17_overlap_test12 = d17_nightsum_test12 * overlap_atten
#If overlap is needed:
[xxxx12,yyyy12] = coadd(dens17_overlap_test12,range_L2,5)

d17_night_test13 = e13[:,156:175] 
d17_nightsum_test13 = np.mean(d17_night_test13,1)

noise11 = np.mean(d17_nightsum_test13[333:511])
print(noise11)
d17_nightsum_test13 = d17_nightsum_test13 + noise11

for f in overlap_atten:
  dens17_overlap_test13 = d17_nightsum_test13 * overlap_atten
#If overlap is needed:
[xxxx13,yyyy13] = coadd(dens17_overlap_test13,range_L2,5)


d17_night_test14 = e14[:,156:175] 
d17_nightsum_test14 = np.mean(d17_night_test14,1)

noise11 = np.mean(d17_nightsum_test14[333:511])
print(noise11)
d17_nightsum_test14 = d17_nightsum_test14 + noise11

for f in overlap_atten:
  dens17_overlap_test14 = d17_nightsum_test14 * overlap_atten
#If overlap is needed:
[xxxx14,yyyy14] = coadd(dens17_overlap_test14,range_L2,5)

d17_night_test15 = e15[:,156:175] 
d17_nightsum_test15 = np.mean(d17_night_test15,1)

noise11 = np.mean(d17_nightsum_test15[333:511])
print(noise11)
d17_nightsum_test15 = d17_nightsum_test15 + noise11

for f in overlap_atten:
  dens17_overlap_test15 = d17_nightsum_test15 * overlap_atten
#If overlap is needed:
[xxxx15,yyyy15] = coadd(dens17_overlap_test15,range_L2,5)

d17_night_test16 = e16[:,156:175] 
d17_nightsum_test16 = np.mean(d17_night_test16,1)

noise11 = np.mean(d17_nightsum_test16[333:511])
print(noise11)
d17_nightsum_test16 = d17_nightsum_test16 + noise11

for f in overlap_atten:
  dens17_overlap_test16 = d17_nightsum_test16 * overlap_atten
#If overlap is needed:
[xxxx16,yyyy16] = coadd(dens17_overlap_test16,range_L2,5)

d17_night_test17 = e17[:,156:175] 
d17_nightsum_test17 = np.mean(d17_night_test17,1)

noise11 = np.mean(d17_nightsum_test17[333:511])
print(noise11)
d17_nightsum_test17 = d17_nightsum_test17 + noise11

for f in overlap_atten:
  dens17_overlap_test17 = d17_nightsum_test17 * overlap_atten
#If overlap is needed:
[xxxx17,yyyy17] = coadd(dens17_overlap_test17,range_L2,5)



#To quality check individual averaged plots: 

plt.semilogx(xxxx1[0:102:], yyyy1)
plt.xlim([0.01,20])
plt.ylim([0, 15000])
#%%----------
#-------------------------MONTH AVERAGE PLOT: ALL YEARS--------------------------------------


fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3 = fig.add_subplot(111)

combx = np.array([ [x1], [x2], [x4], [x5], [x6], [x7],  [x9], [x12], [x13], [x14], [x15], [x16], [x17],  [x19], [x20], [x21], [x22]])
combx19 = np.array([ [xx1], [xx2],  [xx5], [xx7], [xx8], [xx9], [xx10],[xx11],[xx12],[xx13],[xx14], [xx15], [xx16], [xx17], [xx18]])
#combx18 = np.array([ [xxx1], [xxx2], [xxx3], [xxx4], [xxx6], [xxx7], [xxx8], [xxx9], [xxx10],[xxx11],[xxx12],[xxx13],[xxx14], [xxx15], [xxx16], [xxx17], [xxx18]])


comb_mean20 = np.mean(combx, axis=0)
comb_mean19 = np.mean(combx19, axis=(0))
#comb_mean18 = np.mean(combx18, axis=(0))


comb_mean20T = np.transpose(comb_mean20)
comb_mean19T = np.transpose(comb_mean19)
#comb_mean18T = np.transpose(comb_mean18)

ax1.semilogx(comb_mean20T[0:102:], y1, label='April 2020 average')
ax2.semilogx(comb_mean19T[0:102:], y1, label='April 2019 average')
#ax3.semilogx(comb_mean18T[0:102:], y1, label='April 2018 average')

ax1.set_xlim([0.03,20])
ax1.set_ylim([0, 12000])
ax1.legend()
ax1.set_xlabel('Range-Corrected Attenuated Backscatter (m^-1.sr^-1)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Averaged, Range-Corrected Signal over 1 month: April')

plt.show()

#%%
#-------------------------------------PLOT ONLY 1 DAY

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
#ax3 = fig.add_subplot(111)


ax1.semilogx(x1[0:200] ,y1[0:200], label = 'Corrected Data')
ax1.set_xlabel('Normalized, Range-Corrected Attenuated Backscatter (m^-1.sr^-1)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Night Averaged Signal: April 24')
ax1.set_xlim([0.09,20])
ax1.set_ylim([0, 15000])

#ax2.semilogx(xx5[0:200] ,y5[0:200], label = 'April 14, 2019')

#ax3.semilogx(x4[0:200] ,y4[0:200], label = 'April 1, 2020')

ax1.legend()

plt.show()



# %%
#-------------Attempt at making attenuated backscatter colour plot
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

#First place data into bins so that plotting doesnt take as much time
a = int(len(time_L2)/30)
bins = np.arange(0, len(time_L2), a)
beta_bins = []
time_bins = []
for item in bins:
  d_night_bin = beta_att_test[:,0:item] 
  t2 = np.mean(d_night_bin, 1)
  beta_bins.append(t2)
  times = time_L2[0:item]
  t3 = np.mean(times)
  time_bins.append(times)





# Set any value less than or equal to 0 in the beta_att values to 1

# Set any values less than 1 equal to 1

beta_bins = np.array(beta_bins)

beta = np.where(beta_bins <= 0, 1, beta_bins) 



#levels = MaxNLocator(nbins=15).tick_values(1, np.max(np.log(beta_abs)))

cmap = plt.get_cmap('viridis')
#cmap = plt.get_cmap('BrBG')

#cmap = set_color('jet')

#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

beta_binsT = np.transpose(beta_bins)
betaT = np.transpose(beta)
fig, (ax0) = plt.subplots()
beta_abs = np.abs(beta)
im = plt.pcolormesh(time_bins, range_L2, np.log10(betaT), cmap=cmap, vmax=None,vmin=None)
fig.colorbar(im)
plt.title('Normalized, Attenuated Backscatter Power')
plt.xlabel('Time UT [h]')
plt.ylabel('Altitude a.s.l. (m)')
plt.clim(0, -2)
plt.ylim(0, 3000)
plt.show()




# %%

print(bins)
test = np.empty(30)
for i in len(test): 
  test[i] = np.mean(beta_att_test[0:bin[i],:]) 


d_nightsum_test = np.mean(d_night_test,1)
for f in overlap_atten:
  dens_overlap_test = d_nightsum_test * overlap_atten


[x,y] = coadd(beta,range,10)


# %%


np.shape(beta_att_test)
b_time = beta_att_test[:,0:288]
print(np.shape(b_time))
t1 = np.histogram2d(beta_att_test, 30)

# %%
