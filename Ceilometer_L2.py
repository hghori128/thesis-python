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


files_test = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/01/L2_73009_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')

backscatter_test = files_test.variables['attenuated_backscatter_0']

# %%
#----------------------
# Attempting using the netcdf4 module instead of xarray 
# This should be easier as this will load in a numpy like array BUT
# does not handle multiple datasets in the netcdf4 format


#%%
files_test2 = xr.open_mfdataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/04/06/20190406_YXU-Cronyn_CHM160155_*.nc", concat_dim="time", data_vars='minimal', coords='minimal', compat='override')

files_L2 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/04/L2_0-20000-0-73009_A20200404.nc")


#------------------OPEN FILES 2020, 2019 --------------

#2020:
files_L2_11 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/24/L2_0-20000-0-73009_A20200424.nc")

files_L2_1 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/01/L2_0-20000-0-73009_A20200401.nc")
files_L2_2 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/02/L2_0-20000-0-73009_A20200402.nc")
files_L2_3 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/03/L2_0-20000-0-73009_A20200403.nc")
files_L2_4 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/05/L2_0-20000-0-73009_A20200405.nc")
files_L2_5 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/06/L2_0-20000-0-73009_A20200406.nc")
files_L2_6 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/08/L2_0-20000-0-73009_A20200408.nc")
files_L2_7 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/09/L2_0-20000-0-73009_A20200409.nc")
files_L2_8 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/10/L2_0-20000-0-73009_A20200410.nc")
files_L2_9 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2020/04/11/L2_0-20000-0-73009_A20200411.nc")

#2019:

files19_L2_1 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/01/L2_0-20000-0-73009_A20190401.nc")
files19_L2_2 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/03/L2_0-20000-0-73009_A20190403.nc")
files19_L2_3 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/06/L2_0-20000-0-73009_A20190406.nc")
files19_L2_4 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/07/L2_0-20000-0-73009_A20190407.nc")
files19_L2_5 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/08/L2_0-20000-0-73009_A20190408.nc")
files19_L2_6 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/09/L2_0-20000-0-73009_A20190409.nc")
files19_L2_7 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/12/L2_0-20000-0-73009_A20190412.nc")
files19_L2_8 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/13/L2_0-20000-0-73009_A20190413.nc")
files19_L2_9 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/14/L2_0-20000-0-73009_A20190414.nc")

files19_L2_10 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/15/L2_0-20000-0-73009_A20190415.nc")
files19_L2_11 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/16/L2_0-20000-0-73009_A20190416.nc")
files19_L2_12 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/17/L2_0-20000-0-73009_A20190417.nc")
files19_L2_13 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/22/L2_0-20000-0-73009_A20190422.nc")
files19_L2_14 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/23/L2_0-20000-0-73009_A20190423.nc")
files19_L2_15 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/24/L2_0-20000-0-73009_A20190424.nc")
files19_L2_16 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/25/L2_0-20000-0-73009_A20190425.nc")
files19_L2_17 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/27/L2_0-20000-0-73009_A20190427.nc")
files19_L2_18 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/28/L2_0-20000-0-73009_A20190428.nc")
files19_L2_19 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2019/04/29/L2_0-20000-0-73009_A20190429.nc")

#2018

files18_L2_1 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/01/L2_0-20000-0-73009_A20190401.nc")
files18_L2_2 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/02/L2_0-20000-0-73009_A20190402.nc")
files18_L2_3 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/05/L2_0-20000-0-73009_A20190405.nc")
files18_L2_4 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/06/L2_0-20000-0-73009_A20190406.nc")
files18_L2_5 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/07/L2_0-20000-0-73009_A20190407.nc")
files18_L2_6 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/08/L2_0-20000-0-73009_A20190408.nc")
files18_L2_7 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/09/L2_0-20000-0-73009_A20190409.nc")
files18_L2_8 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/10/L2_0-20000-0-73009_A20190410.nc")
files18_L2_9 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/12/L2_0-20000-0-73009_A20190412.nc")

files18_L2_10 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/13/L2_0-20000-0-73009_A20190413.nc")
files18_L2_11 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/19/L2_0-20000-0-73009_A20190419.nc")
files18_L2_12 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/20/L2_0-20000-0-73009_A20190420.nc")
files18_L2_13 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/21/L2_0-20000-0-73009_A20190421.nc")
files18_L2_14 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/23/L2_0-20000-0-73009_A20190423.nc")
files18_L2_15 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/26/L2_0-20000-0-73009_A20190426.nc")
files18_L2_16 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/27/L2_0-20000-0-73009_A20190427.nc")
files18_L2_17 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/29/L2_0-20000-0-73009_A20190429.nc")
files18_L2_18 = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/30/L2_0-20000-0-73009_A20190430.nc")


#beta_att_test = []

#files_L2 = nc.MFDataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2018/04/01/L2_73009_*.nc")
#beta_att_test = files_test.variables['attenuated_backscatter_0'][:]
b = files_test2.variables['beta_raw']

# Initialize 2020 variables

beta_att_test = files_L2.variables['attenuated_backscatter_0'][:]
b1 = files_L2_11.variables['attenuated_backscatter_0'][:]
b2 = files_L2_2.variables['attenuated_backscatter_0'][:]
b3 = files_L2_3.variables['attenuated_backscatter_0'][:]
b4 = files_L2_4.variables['attenuated_backscatter_0'][:]
b5 = files_L2_5.variables['attenuated_backscatter_0'][:]
b6 = files_L2_6.variables['attenuated_backscatter_0'][:]
b7 = files_L2_7.variables['attenuated_backscatter_0'][:]
b8 = files_L2_8.variables['attenuated_backscatter_0'][:]
b9 = files_L2_9.variables['attenuated_backscatter_0'][:]

#Initialize 2019 variables
c1 = files19_L2_1.variables['attenuated_backscatter_0'][:]
c2 = files19_L2_2.variables['attenuated_backscatter_0'][:]
c3 = files19_L2_3.variables['attenuated_backscatter_0'][:]
c4 = files19_L2_4.variables['attenuated_backscatter_0'][:]
c5 = files19_L2_5.variables['attenuated_backscatter_0'][:]
c6 = files19_L2_6.variables['attenuated_backscatter_0'][:]
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

#Initialize 2018 variables
d1 = files18_L2_1.variables['attenuated_backscatter_0'][:]
d2 = files18_L2_2.variables['attenuated_backscatter_0'][:]
d3 = files18_L2_3.variables['attenuated_backscatter_0'][:]
d4 = files18_L2_4.variables['attenuated_backscatter_0'][:]
d5 = files18_L2_5.variables['attenuated_backscatter_0'][:]
d6 = files18_L2_7.variables['attenuated_backscatter_0'][:]
d7 = files18_L2_8.variables['attenuated_backscatter_0'][:]
d8 = files18_L2_9.variables['attenuated_backscatter_0'][:]
d9 = files18_L2_10.variables['attenuated_backscatter_0'][:]

d10 = files18_L2_11.variables['attenuated_backscatter_0'][:]
d11 = files18_L2_12.variables['attenuated_backscatter_0'][:]
d12 = files18_L2_13.variables['attenuated_backscatter_0'][:]
d13 = files18_L2_14.variables['attenuated_backscatter_0'][:]
d14 = files18_L2_15.variables['attenuated_backscatter_0'][:]
d15 = files18_L2_16.variables['attenuated_backscatter_0'][:]
d16 = files18_L2_17.variables['attenuated_backscatter_0'][:]
d17 = files18_L2_18.variables['attenuated_backscatter_0'][:]


#Initialize 2017 variables



range_L2 = files_L2.variables['altitude'][:]
range = files.variables['range'][:]
#%%
#----------------------CHECK CLEAR SKY HERE--------------------
files = nc.MFDataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2019/04/03/20190403_YXU-Cronyn_CHM160155_*.nc') 
#files = nc.Dataset("/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/2017/04/01/20170401_YXU-Cronyn_CHM160155_000.nc")

time_L2 = files_L2.variables['time'][:]
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


customdate = datetime.datetime(year=1904, month=1, day=1, hour=0,second=0)
realtime = [ customdate + datetime.timedelta(seconds=i) for i in (time_raw)]

#utc = []
#for l in realtime:
#    ff = l.strftime('%H:%M')
#    utc.append(ff)

#print(utc)

#Plotting for the reasons of quality checking and deterimining
# whether it is clear sky or not

#Plot the visibility
plt.plot(realtime, vor)
plt.title('Vertical optical visibility: 2019/04/04')
plt.xlabel('Time in seconds after 00:00')
plt.ylabel('Height (m)')
plt.show()

#clear = np.where(base_cloud == 0)[0]
#plt.scatter(time_raw-(3.668*10**9), total_cloud)

#Plot the total clour cover
plt.scatter(realtime, total_cloud)
plt.title('Amount of cloud cover in octas: 2020/04/03')
plt.xlabel('Time in seconds after 00:00')
plt.ylabel('Octa (0 = clear sky, 8 = total cloud cover)')
plt.ylim(0, 8)
plt.show()

#Plot the max detection height
plt.plot(realtime, max_height)
plt.title('Maximum detection height: 2019/04/04')
plt.xlabel('Time in seconds after 00:00')
plt.ylabel('Height (m)')
plt.show()

max_height = np.array(max_height)
print(max_height)
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

a = int(len(beta_att_test)/2)
print(a)
#9pm - 11:59pm
d_night_test1 = b1[:,248:250] 
d_nightsum_test1 = np.mean(d_night_test1,1)

noise1 = np.mean(d_nightsum_test1[333:511])
print(noise1)
d_nightsum_test1 = d_nightsum_test1 + noise1

for f in overlap_atten:
  dens_overlap_test1 = d_nightsum_test1 * overlap_atten
#If overlap is needed:
[x1,y1] = coadd(dens_overlap_test1,range_L2,5)

a = int(len(beta_att_test)/2)
print(a)
d_night_test2 = b2[:,0:120] 
d_nightsum_test2 = np.mean(d_night_test2,1)

noise2 = np.mean(d_nightsum_test2[333:511])
print(noise2)
d_nightsum_test2 = d_nightsum_test2 + noise2

for f in overlap_atten:
  dens_overlap_test2 = d_nightsum_test2 * overlap_atten
#If overlap is needed:
[x2,y2] = coadd(dens_overlap_test2,range_L2,5)

a = int(len(beta_att_test)/2)
print(a)
d_night_test3 = b3[:,0:40] 
d_nightsum_test3 = np.mean(d_night_test3,1)

noise3 = np.mean(d_nightsum_test3[333:511])
print(noise3)
d_nightsum_test3 = d_nightsum_test3 + noise3

for f in overlap_atten:
  dens_overlap_test3 = d_nightsum_test3 * overlap_atten
#If overlap is needed:
[x3,y3] = coadd(dens_overlap_test3,range_L2,5)

a = int(len(beta_att_test)/2)
print(a)
d_night_test4 = b4[:,200:a] 
d_nightsum_test4 = np.mean(d_night_test4,1)

noise4 = np.mean(d_nightsum_test4[333:511])
print(noise4)
d_nightsum_test4 = d_nightsum_test4 + noise4

for f in overlap_atten:
  dens_overlap_test4 = d_nightsum_test4 * overlap_atten
#If overlap is needed:
[x4,y4] = coadd(dens_overlap_test4,range_L2,5)

a = int(len(beta_att_test)/2)
print(a)
d_night_test5 = b5[:,0:a] 
d_nightsum_test5 = np.mean(d_night_test5,1)

noise5 = np.mean(d_nightsum_test5[333:511])
print(noise5)
d_nightsum_test5 = d_nightsum_test5 + noise5

for f in overlap_atten:
  dens_overlap_test5 = d_nightsum_test5 * overlap_atten
#If overlap is needed:
[x5,y5] = coadd(dens_overlap_test5,range_L2,5)

a = int(len(beta_att_test)/2)
print(a)
d_night_test6 = b6[:,200:a] 
d_nightsum_test6 = np.mean(d_night_test6,1)

noise6 = np.mean(d_nightsum_test6[333:511])
print(noise6)
d_nightsum_test6 = d_nightsum_test6 + noise6

for f in overlap_atten:
  dens_overlap_test6 = d_nightsum_test6 * overlap_atten
#If overlap is needed:
[x6,y6] = coadd(dens_overlap_test6,range_L2,5)

d_night_test7 = b7[:,0:24] 
d_nightsum_test7 = np.mean(d_night_test7,1)

noise7 = np.mean(d_nightsum_test7[333:511])
print(noise7)
d_nightsum_test7 = d_nightsum_test7 + noise7

for f in overlap_atten:
  dens_overlap_test7 = d_nightsum_test7 * overlap_atten
#If overlap is needed:
[x7,y7] = coadd(dens_overlap_test7,range_L2,5)

d_night_test8 = b8[:,220:240] 
d_nightsum_test8 = np.mean(d_night_test8,1)

noise8 = np.mean(d_nightsum_test8[333:511])
print(noise8)
d_nightsum_test8 = d_nightsum_test8 + noise8

for f in overlap_atten:
  dens_overlap_test8 = d_nightsum_test8 * overlap_atten
#If overlap is needed:
[x8,y8] = coadd(dens_overlap_test8,range_L2,5)

d_night_test9 = b9[:,90:114] 
d_nightsum_test9 = np.mean(d_night_test9,1)

noise9 = np.mean(d_nightsum_test9[333:511])
print(noise9)
d_nightsum_test9 = d_nightsum_test9 + noise9

for f in overlap_atten:
  dens_overlap_test9 = d_nightsum_test9 * overlap_atten
#If overlap is needed:
[x9,y9] = coadd(dens_overlap_test9,range_L2,5)

plt.semilogx(x9[0:102:], y1)
plt.xlim([0.001,20])
plt.ylim([0, 15000])

#%%

#%%------------------------------
# Calcultate all profiles for 2019:

a = int(len(beta_att_test)/2)
print(a)
d19_night_test1 = c1[:,156:175] 
d19_nightsum_test1 = np.mean(d19_night_test1,1)

noise11 = np.mean(d19_nightsum_test1[333:511])
print(noise11)
d19_nightsum_test1 = d19_nightsum_test1 + noise11

for f in overlap_atten:
  dens19_overlap_test1 = d19_nightsum_test1 * overlap_atten
#If overlap is needed:
[xx1,yy1] = coadd(dens19_overlap_test1,range_L2,5)

a = int(len(beta_att_test)/2)
print(a)
d19_night_test2 = c2[:,102:114] 
d19_nightsum_test2 = np.mean(d19_night_test2,1)

noise22 = np.mean(d19_nightsum_test2[333:511])
print(noise22)
d19_nightsum_test2 = d19_nightsum_test2 + noise22

for f in overlap_atten:
  dens19_overlap_test2 = d19_nightsum_test2 * overlap_atten
#If overlap is needed:
[xx2,yy2] = coadd(dens19_overlap_test2,range_L2,5)

a = int(len(beta_att_test)/2)
print(a)
d19_night_test3 = c3[:,144:168] 
d19_nightsum_test3 = np.mean(d19_night_test3,1)

noise33 = np.mean(d19_nightsum_test3[333:511])
print(noise33)
d19_nightsum_test3 = d19_nightsum_test3 + noise33

for f in overlap_atten:
  dens19_overlap_test3 = d19_nightsum_test3 * overlap_atten
#If overlap is needed:
[xx3,yy3] = coadd(dens19_overlap_test3,range_L2,5)

a = int(len(beta_att_test)/2)
print(a)
d19_night_test4 = c4[:,90:105] 
d19_nightsum_test4 = np.mean(d19_night_test4,1)

noise44 = np.mean(d19_nightsum_test4[333:511])
print(noise44)
d19_nightsum_test4 = d19_nightsum_test4 + noise44

for f in overlap_atten:
  dens19_overlap_test4 = d19_nightsum_test4 * overlap_atten
#If overlap is needed:
[xx4,yy4] = coadd(dens19_overlap_test4,range_L2,5)

a = int(len(beta_att_test)/2)
print(a)
d19_night_test5 = c5[:,105:120] 
d19_nightsum_test5 = np.mean(d19_night_test5,1)

noise55 = np.mean(d19_nightsum_test5[333:511])
print(noise55)
d19_nightsum_test5 = d19_nightsum_test5 + noise55

for f in overlap_atten:
  dens19_overlap_test5 = d19_nightsum_test5 * overlap_atten
#If overlap is needed:
[xx5,yy5] = coadd(dens19_overlap_test5,range_L2,5)

a = int(len(beta_att_test)/2)
print(a)
d19_night_test6 = c6[:,18:62] 
d19_nightsum_test6 = np.mean(d19_night_test6,1)

noise66 = np.mean(d19_nightsum_test6[333:511])
print(noise66)
d19_nightsum_test6 = d19_nightsum_test6 + noise66

for f in overlap_atten:
  dens19_overlap_test6 = d19_nightsum_test6 * overlap_atten
#If overlap is needed:
[xx6,yy6] = coadd(dens19_overlap_test6,range_L2,5)

d19_night_test7 = c7[:,18:62] 
d19_nightsum_test7 = np.mean(d19_night_test7,1)

noise77 = np.mean(d19_nightsum_test7[333:511])
print(noise77)
d19_nightsum_test7 = d19_nightsum_test7 + noise77

for f in overlap_atten:
  dens19_overlap_test7 = d19_nightsum_test7 * overlap_atten
#If overlap is needed:
[xx7,yy7] = coadd(dens19_overlap_test7,range_L2,5)

d19_night_test8 = c8[:,18:62] 
d19_nightsum_test8 = np.mean(d19_night_test8,1)

noise88 = np.mean(d19_nightsum_test8[333:511])
print(noise88)
d19_nightsum_test8 = d19_nightsum_test8 + noise88

for f in overlap_atten:
  dens19_overlap_test8 = d19_nightsum_test8 * overlap_atten
#If overlap is needed:
[xx8,yy8] = coadd(dens19_overlap_test8,range_L2,5)

d19_night_test9 = c9[:,18:62] 
d19_nightsum_test9 = np.mean(d19_night_test9,1)

noise99 = np.mean(d19_nightsum_test9[333:511])
print(noise66)
d19_nightsum_test9 = d19_nightsum_test9 + noise99

for f in overlap_atten:
  dens19_overlap_test9 = d19_nightsum_test9 * overlap_atten
#If overlap is needed:
[xx9,yy9] = coadd(dens19_overlap_test9,range_L2,5)

d19_night_test10 = c10[:,252:276] 
d19_nightsum_test10 = np.mean(d19_night_test10,1)

noise99 = np.mean(d19_nightsum_test10[333:511])
d19_nightsum_test10 = d19_nightsum_test10 + noise99

for f in overlap_atten:
  dens19_overlap_test10 = d19_nightsum_test10 * overlap_atten
#If overlap is needed:
[xx10,yy10] = coadd(dens19_overlap_test10,range_L2,5)

d19_night_test11 = c11[:,0:24] 
d19_nightsum_test11 = np.mean(d19_night_test11,1)

noise99 = np.mean(d19_nightsum_test11[333:511])
d19_nightsum_test11 = d19_nightsum_test11 + noise99

for f in overlap_atten:
  dens19_overlap_test11 = d19_nightsum_test11 * overlap_atten
#If overlap is needed:
[xx11,yy11] = coadd(dens19_overlap_test11,range_L2,5)

d19_night_test12 = c12[:,186:198] 
d19_nightsum_test12 = np.mean(d19_night_test12,1)

noise99 = np.mean(d19_nightsum_test12[333:511])
d19_nightsum_test12 = d19_nightsum_test12 + noise99

for f in overlap_atten:
  dens19_overlap_test12 = d19_nightsum_test12 * overlap_atten
#If overlap is needed:
[xx12,yy12] = coadd(dens19_overlap_test12,range_L2,5)

d19_night_test13 = c13[:,42:55] 
d19_nightsum_test13 = np.mean(d19_night_test13,1)

noise99 = np.mean(d19_nightsum_test13[333:511])
d19_nightsum_test13 = d19_nightsum_test13 + noise99

for f in overlap_atten:
  dens19_overlap_test13 = d19_nightsum_test13 * overlap_atten
#If overlap is needed:
[xx13,yy13] = coadd(dens19_overlap_test13,range_L2,5)

d19_night_test14 = c14[:,0:12] 
d19_nightsum_test14 = np.mean(d19_night_test14,1)

noise99 = np.mean(d19_nightsum_test14[333:511])
d19_nightsum_test14 = d19_nightsum_test14 + noise99

for f in overlap_atten:
  dens19_overlap_test14 = d19_nightsum_test14 * overlap_atten
#If overlap is needed:
[xx14,yy14] = coadd(dens19_overlap_test14,range_L2,5)

d19_night_test15 = c15[:,108:144] 
d19_nightsum_test15 = np.mean(d19_night_test15,1)

noise99 = np.mean(d19_nightsum_test15[333:511])
d19_nightsum_test15 = d19_nightsum_test15 + noise99

for f in overlap_atten:
  dens19_overlap_test15 = d19_nightsum_test15 * overlap_atten
#If overlap is needed:
[xx15,yy15] = coadd(dens19_overlap_test15,range_L2,5)

d19_night_test16 = c16[:,48:72] 
d19_nightsum_test16 = np.mean(d19_night_test16,1)

noise99 = np.mean(d19_nightsum_test16[333:511])
d19_nightsum_test16 = d19_nightsum_test16 + noise99

for f in overlap_atten:
  dens19_overlap_test16 = d19_nightsum_test16 * overlap_atten
#If overlap is needed:
[xx16,yy16] = coadd(dens19_overlap_test16,range_L2,5)

d19_night_test17 = c17[:,6:30] 
d19_nightsum_test17 = np.mean(d19_night_test17,1)

noise99 = np.mean(d19_nightsum_test17[333:511])
d19_nightsum_test17 = d19_nightsum_test17 + noise99

for f in overlap_atten:
  dens19_overlap_test17 = d19_nightsum_test17 * overlap_atten
#If overlap is needed:
[xx17,yy17] = coadd(dens19_overlap_test17,range_L2,5)

d19_night_test18 = c18[:,180:204] 
d19_nightsum_test18 = np.mean(d19_night_test18,1)

noise99 = np.mean(d19_nightsum_test18[333:511])
d19_nightsum_test18 = d19_nightsum_test18 + noise99

for f in overlap_atten:
  dens19_overlap_test18 = d19_nightsum_test18 * overlap_atten
#If overlap is needed:
[xx18,yy18] = coadd(dens19_overlap_test18,range_L2,5)

d19_night_test19 = c19[:,36:72] 
d19_nightsum_test19 = np.mean(d19_night_test19,1)

noise99 = np.mean(d19_nightsum_test19[333:511])
d19_nightsum_test19 = d19_nightsum_test19 + noise99

for f in overlap_atten:
  dens19_overlap_test19 = d19_nightsum_test19 * overlap_atten
#If overlap is needed:
[xx19,yy19] = coadd(dens19_overlap_test19,range_L2,5)

plt.semilogx(xx18[0:102:], yy1)
plt.xlim([0.01,20])
plt.ylim([0, 15000])

#%%----------
#-------------------------MONTH AVERAGE PLOT: ALL YEARS--------------------------------------


fig = plt.figure()

ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)

combx = np.array([ [x2], [x3], [x4], [x5],[x6], [x7], [x8], [x9]])
combx19 = np.array([[xx1], [xx2], [xx3], [xx4], [xx5],[xx6], [xx7], [xx8], [xx9], [xx10],[xx11],[xx12],[xx13],[xx14], [xx15], [xx16], [xx17], [xx18],[xx19]])

comb_mean20 = np.mean(combx, axis=0)
comb_mean19 = np.mean(combx19, axis=0)

comb_mean20T = np.transpose(comb_mean20)
comb_mean19T = np.transpose(comb_mean19)

ax1.semilogx(comb_mean20T[0:102:], y1, label='April 2020 average')
ax2.semilogx(comb_mean19T[0:102:], y1, label='April 2019 average')

ax1.set_xlim([0.03,20])
ax1.set_ylim([0, 16000])
ax1.legend()
ax1.set_xlabel('Range-Corrected Attenuated Backscatter (m^-1.sr^-1)')
ax1.set_ylabel('Height (m)')
ax1.set_title('Night Averaged Signal for 9 days: April')

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
