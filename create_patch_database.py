# patch_detection_database.py
# Find polar cap patches across a portion of the AMISR database
# Automatically walk the RISR-N database and identify polar cap patches in each experiment
# Patch detection based on the patch finding algorithm presented in Ren et al., 2018
# Algorithm originally developed in MATLAB by Jiaen Ren @ University of Michigan
# Conversion to python by Leslie Lamarche @ SRI International

import numpy as np
import datetime as dt
import scipy.stats as stats
import scipy.signal as sig
import scipy.special as sp
import os
import re
import h5py
from procdbtools.amisr_lookup import AMISR_lookup

starttime = dt.datetime(2009,1,1)
endtime = dt.datetime(2023,1,1)
output_file = 'patch_database_risrn.h5'

patches = {'peak':[],'time':[],'prominence':[],'fidx':[],'lb_idx':[],'rb_idx':[]}

amisrdb = AMISR_lookup('RISR-N')

experiment_list = amisrdb.find_experiments(starttime, endtime)

for exp in experiment_list:
    filename = amisrdb.select_datafile(exp, pulse='lp')
    if not filename:
        continue
    else:
        print(filename)

    # read the file
    with h5py.File(filename, 'r') as f:

        beam = np.argmax(f['BeamCodes'][:,2])    # select beam with highest elevation angle

        # read data into arrays
        utime = f['Time/UnixTime'][:]
        alt = f['NeFromPower/Altitude'][beam,:]
        Ne = f['NeFromPower/Ne_NoTr'][:,beam,:]

    # transform density and altitude arrays into correct
    Ne = np.log10(Ne[:,np.isfinite(alt) & (alt>100000.)])
    alt = alt[np.isfinite(alt) & (alt>100000.)]/1000.

    # Filter out anomolously high density values
    Ne[Ne>=12.5] = np.nan

    # Filter outlier by removing points with density above 11.5 AND more than 3.5 scaled MADs
    #   away from the median on a 17 point sliding window.  This is equivilent to the MATLAB
    #   function isoulier (https://www.mathworks.com/help/matlab/ref/isoutlier.html) called as follows:
    #   isoutlier(Ne,'movmedian',17,'ThresholdFactor',3.5)
    sw = np.squeeze(np.lib.stride_tricks.sliding_window_view(Ne, window_shape=(1,17)))
    fill_shape = (sw.shape[0],8,sw.shape[2])
    sw = np.concatenate((np.full(fill_shape,np.nan), sw, np.full(fill_shape,np.nan)), axis=1)
    mad = stats.median_abs_deviation(sw, scale=-np.sqrt(2)*sp.erfcinv(3/2), axis=-1, nan_policy='omit')
    med = np.nanmedian(sw, axis=-1)
    TF = np.abs(Ne-med)/mad>3.5
    Ne[(Ne>=11.5) & TF] = np.nan

    # calculate the average density in the desired altitude range
    altrange = [250.,400.]
    aidx = [np.argmin(np.abs(a-alt)) for a in altrange]
    Ne_F2 = np.nanmean(Ne[:,aidx[0]:aidx[1]], axis=-1)

    # Apply 3-point median filter if time cadence less than 2.5 minutes
    if np.mean(np.diff(utime[:,0]))<2.5*60.:
        Ne_F2 = sig.medfilt(Ne_F2, kernel_size=3)

    # Find initial peaks for entire time series
    peaks0, _ = sig.find_peaks(Ne_F2, prominence=np.log10(2), width=0.)

    # For each peak idenified, consider a 2 hour window around the peak and make sure it can
    #   still be identified.  If so, save the time the patch occured and the charactristics of
    #   the peak.
    for p in peaks0:
        start = np.argmin(np.abs((utime[p,0]-1.*60.*60.)-utime[:,0]))
        end = np.argmin(np.abs((utime[p,0]+1.*60.*60.)-utime[:,0]))
        peaks, prop = sig.find_peaks(Ne_F2[start:end], prominence=np.log10(2), height=0., width=0.)
        try:
            idx = list(start+peaks).index(p)
            patches['fidx'].append(p)
            patches['time'].append(utime[p,0])
            patches['peak'].append(prop['peak_heights'][idx])
            patches['prominence'].append(prop['prominences'][idx])
            patches['lb_idx'].append(prop['left_bases'][idx]+start)
            patches['rb_idx'].append(prop['right_bases'][idx]+start)
        except ValueError:
            continue

# save list of patch parameters to output hdf5 file
with h5py.File(output_file, 'w') as h5:
    for key, val in patches.items():
        h5.create_dataset(key, data=np.array(val))
