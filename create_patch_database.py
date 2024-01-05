# patch_detection_database.py
# Find polar cap patches across a portion of the AMISR database
# Automatically walk the RISR-N database and identify polar cap patches in each experiment
# Patch detection based on two patch finding algorithms:
# Ren et al., 2018 algorithm originally developed in MATLAB by Jiaen Ren @ University of Michigan
# Perry & St. Maurice, 2018 algorithm originally developed in MATLAB by Gareth Perry @ University of Saskatchewan
# Conversion to python by Leslie Lamarche and Olu Jonah @ SRI International

import numpy as np
import datetime as dt
import scipy.stats as stats
import scipy.signal as sig
import scipy.special as sp
import os
import re
import h5py
from procdbtools.amisr_lookup import AMISR_lookup
import argparse


parser = argparse.ArgumentParser(description='Create a list of polar cap patches from the AMISR database.')
parser.add_argument('start', help='start date for patch detection')
parser.add_argument('end', help='end date for patch detection')
parser.add_argument('-o', dest='output_file', help='output file name (default: patch_database.h5)', default='patch_database.h5')
args = parser.parse_args()

starttime = dt.datetime.fromisoformat(args.start)
endtime = dt.datetime.fromisoformat(args.end)
output_file = args.output_file


def Ren2018_algorithm(filename):

    # read input file
    with h5py.File(filename, 'r') as f:

        beam = np.argmax(f['BeamCodes'][:,2])    # select beam with highest elevation angle

        # read data into arrays
        utime = f['Time/UnixTime'][:]
        alt = f['NeFromPower/Altitude'][beam,:]
        Ne = f['NeFromPower/Ne_NoTr'][:,beam,:]
        fit_Te = f['FittedParams/Fits'][:,beam,:,-1,1] ##### Olu
        fit_Ti = f['FittedParams/Fits'][:,beam,:,0,1] ##### Olu
        fit_Ne = f['FittedParams/Ne'][:,beam,:] ##### Olu
        fit_alt = f['FittedParams/Altitude'][beam,:] ##### Olu

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
    fit_aidx = [np.nanargmin(np.abs(a-fit_alt/1000.)) for a in altrange]

    # Apply 3-point median filter if time cadence less than 2.5 minutes
    if np.mean(np.diff(utime[:,0]))<2.5*60.:
        Ne_F2 = sig.medfilt(Ne_F2, kernel_size=3)

    # Find initial peaks for entire time series
    peaks0, _ = sig.find_peaks(Ne_F2, prominence=np.log10(2), width=0.)

    # For each peak idenified, consider a 2 hour window around the peak and make sure it can
    #   still be identified.  If so, save the time the patch occured and the charactristics of
    #   the peak.
    time = list()
    peak = list()
    prominence = list()
    avgte = list()
    avgti = list()
    avgne = list()
    for p in peaks0:
        start = np.argmin(np.abs((utime[p,0]-1.*60.*60.)-utime[:,0]))
        end = np.argmin(np.abs((utime[p,0]+1.*60.*60.)-utime[:,0]))
        peaks, prop = sig.find_peaks(Ne_F2[start:end], prominence=np.log10(2), height=0., width=0.)
        try:
            idx = list(start+peaks).index(p)
            sidx = prop['left_bases'][idx]+start
            eidx = prop['right_bases'][idx]+start
            time.append([utime[p,0], utime[sidx,0], utime[eidx,0]])
            peak.append(prop['peak_heights'][idx])
            prominence.append(prop['prominences'][idx])
            avgte.append(np.nanmean(fit_Te[sidx:eidx,fit_aidx[0]:fit_aidx[1]]))
            avgti.append(np.nanmean(fit_Ti[sidx:eidx,fit_aidx[0]:fit_aidx[1]]))
            print(fit_Ne[sidx:eidx,fit_aidx[0]:fit_aidx[1]])
            avgne.append(np.nanmean(fit_Ne[sidx:eidx,fit_aidx[0]:fit_aidx[1]]))
        except ValueError:
            continue

    return time, peak, prominence, avgte, avgti, avgne



def Perry2018_algorithm(filename):

    with h5py.File(filename, 'r') as f:
        dens = f['FittedParams']['Ne'][:]
        err = f['FittedParams']['dNe'][:]
        rng = f['FittedParams']['Range'][:]
        alt = f['FittedParams']['Altitude'][:]
        utime = f['Time']['UnixTime'][:]
        bco = f['BeamCodes'][:]
    
        chi2 = f['FittedParams/FitInfo']['chi2'][:]
        fitcode = f['FittedParams/FitInfo']['fitcode'][:]
        
    Nbeam = len(bco[:,0])
    
    # filtering
    dens[dens>1e13]=np.nan
    dens[dens<0]=np.nan
    
    dens[err>dens] = np.nan
    
    dens[(chi2<0.1) & (chi2>10.)] = np.nan
    dens[(fitcode<1) & (fitcode>4)] = np.nan
    
    # determine the average time between records and the number of records in a 30 minute window
    dt = np.mean(utime[1:,0]-utime[:-1,0])
    win = int(30.*60./dt)
    
    # determine which altitude indicesed correspond to 200 and 500 km
    alt_c, alt_r = np.nonzero((alt >= 200*1000.) & (alt <=500*1000.))
    Ngates = len(alt_c)
    
    # find median of points in the correct altitude range for each time
    med_dens = np.nanmedian(dens[:,alt_c, alt_r], axis=1)
    
    pad1 = np.full(int(win/2), np.nan)
    if win % 2 != 0:
        pad2 = pad1
    else:
        pad2 = pad1[:-1]
    pad_med_dens = np.concatenate((pad1, med_dens, pad2))
    sw_dens = np.lib.stride_tricks.sliding_window_view(pad_med_dens, win)
    backgrnd = np.nanmean(sw_dens, axis=1)
    
    patch_index = np.empty(backgrnd.shape)
    
    for tidx in range(len(backgrnd)):
    
        patch_beam = 0.
        for bidx in range(Nbeam):
            condition = ((dens[tidx,bidx,:]>=2*backgrnd[tidx]) & (alt[bidx,:]>=200*1000.) & (alt[bidx,:]<=500*1000.))
            conv = np.convolve(condition.astype(int), np.ones(3), mode='valid')
            patch_beam += np.sum(conv>=3)

        # patch_index[tidx] = patch_beam/Ngates*100.
        patch_index[tidx] = patch_beam

    return utime[:,0], patch_index, Nbeam


 
Ren2018 = dict(time=list(), peak=list(), prominence=list(), avgTe=list(), avgTi=list(), avgNe=list())
Perry2018 = dict(time=list(), patch_index=list(), num_beams=list())

amisrdb = AMISR_lookup('RISR-N')

experiment_list = amisrdb.find_experiments(starttime, endtime)

for exp in experiment_list:
    filename = amisrdb.select_datafile(exp, pulse='lp')
    if not filename:
        continue
    else:
        print(filename)

    time, peak, prom, avgte, avgti, avgne = Ren2018_algorithm(filename)
    Ren2018['time'].extend(time)
    Ren2018['peak'].extend(peak)
    Ren2018['prominence'].extend(prom)
    Ren2018['avgTe'].extend(avgte)
    Ren2018['avgTi'].extend(avgti)
    Ren2018['avgNe'].extend(avgne)
    time, patch_index, Nbeam = Perry2018_algorithm(filename)
    Perry2018['time'].extend(time)
    Perry2018['patch_index'].extend(patch_index)
    Perry2018['num_beams'].extend(np.full(time.shape, Nbeam))


Ren2018_descriptions = {
        'time': 'Unix time (seconds since January 1, 1970) of patch peak, start, and end respectively [Npatches x 3]',
        'peak': 'Peak electron density of patch [Npatches]',
        'prominence': 'Prominence of patch relative to background [Npatches]',
        'avgTe': 'Average electron temperature between the start and end times of the patch and the upper and lower altitude bounds selected to identify patches [Npatches]',
        'avgTi': 'Average ion temperature between the start and end times of the patch and the upper and lower altitude bounds selected to identify patches [Npatches]',
        'avgNe': 'Average electron temperature between the start and end times of the patch and the upper and lower altitude bounds selected to identify patches [Npatches]'
        }

Perry2018_descriptions = {
        'time': 'Unix time (seconds since January 1, 1970) of records [Nrecords]',
        'patch_index': 'Polar Cap Patch Index across the entire Field of View [Nrecords]',
        'num_beams': 'The total number of radar beams available at each time [Nrecords]'
        }



# save list of patch parameters to output hdf5 file
with h5py.File(output_file, 'w') as h5:
    grp = h5.create_group('Ren2018')
    grp.attrs['description'] = 'Patches identified from the Ren et al, 2018 algorithm and some information about each patch.  Arrays have a dimension Npatches indicating the total number of patches in this database.'
    for key, val in Ren2018.items():
        print(key, len(val))
        ds = grp.create_dataset(key, data=np.array(val))
        ds.attrs['description'] = Ren2018_descriptions[key]

    grp = h5.create_group('Perry2018')
    grp.attrs['description'] = 'Polar Cap Patch Index as described in Perry & St. Maurice, 2018 indicating the general degree of "patchyness" of the radar FoV.  Arrays have a dimension Nrecords indicating the total number of time records (radar integration periods) in this database.'
    for key, val in Perry2018.items():
        print(key, len(val))
        ds = grp.create_dataset(key, data=np.array(val))
        ds.attrs['description'] = Perry2018_descriptions[key]


