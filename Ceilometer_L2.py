#%%
import numpy as np
import matplotlib.pyplot as plt


import xarray as xr

#%%
#----------------------------
#Following function recieved from http://xarray.pydata.org/en/stable/io.html and modified as such
# Which should work sufficiently well for large datasets
from glob import glob

def read_netcdfs(files, dim, transform_func=None):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(glob(files))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined

# here we suppose we only care about the combined mean of each file;
# you might also use indexing operations like .sel to subset datasets
dataset = read_netcdfs('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2017/07/30/L2_*.nc', dim='time',transform_func=lambda ds: ds.mean())
#print(combined)
beta_att = dataset.variables['beta_att_raw']
print(beta_att)


# %%
import xarray as xr

xr.open_mfdataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2017/07/30/L2_*.nc')

dataset =  xr.open_mfdataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2017/07/30/L2_*.nc', concat_dim="time", data_vars='minimal', coords='minimal', compat='override')

#%%

#----------------------------
#Following function recieved from http://xarray.pydata.org/en/stable/io.html and modified as such

from glob import glob

def read_netcdfs(files, dim, transform_func=None):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(glob(files))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined

# here we suppose we only care about the combined mean of each file;
# you might also use indexing operations like .sel to subset datasets
#%%
combined = read_netcdfs('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2017/07/30/L2_*.nc', dim='time',transform_func=lambda ds: ds.mean())
print(combined)
datatest = xr.open_mfdataset('/Users/hannanghori/Documents/university shit/4th year/Thesis/Cronyn/L2/2017/07/30/L2_*.nc', concat_dim="time", data_vars='minimal', coords='minimal', compat='override')

beta_att = combined.variables['attenuated_backscatter_0']
Range = datatest.variables['altitude']

plt.plot(beta_att, Range)



#%%

# %%
