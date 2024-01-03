# Perry2018_Fig5.py
# Replicate parts of Figure 5 from Perry & St. Maurice, 2018
# NOTE: This does not suceed in fully reproducing the patch counts shown in the bottom panel of Figure 5.
#   There is probably an error in how patch detections are counted in groups.  It is not entirely clear
#   how this was done in the original matlab code and the translation to python is probably imperfect.

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

filename = '/Users/e30737/Desktop/Data/AMISR/RISR-N/2010/20100309.001_lp_3min-cal.h5'

with h5py.File(filename, 'r') as f:
    dens = f['FittedParams']['Ne'][:]
    err = f['FittedParams']['dNe'][:]
    rng = f['FittedParams']['Range'][:]
    alt = f['FittedParams']['Altitude'][:]
    utime = f['Time']['UnixTime'][:]
    bco = f['BeamCodes'][:]
    glat=f['Geomag']['Latitude'][:]
    glon=f['Geomag']['Longitude'][:]

    chi2 = f['FittedParams/FitInfo']['chi2'][:]
    fitcode = f['FittedParams/FitInfo']['fitcode'][:]
    
time = utime.astype('datetime64[s]')
Nbeam = len(bco[:,0])

# filtering
dens[dens>1e13]=np.nan
dens[dens<0]=np.nan

dens[err>dens] = np.nan

dens[(chi2<0.1) & (chi2>10.)] = np.nan
dens[(fitcode<1) & (fitcode>4)] = np.nan

# determine the average time between records and the number of records in a 30 minute window
dt = np.mean(time[1:,0]-time[:-1,0])
win = int(np.timedelta64(30, 'm')/dt)

# determine which altitude indicesed correspond to 200 and 500 km
alt_c, alt_r = np.nonzero((alt >= 200*1000.) & (alt <=500*1000.))
Ngates = len(alt_c)
print(Ngates)

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
# for tidx in range(1010,1020):

    patch_beam = 0.
    for bidx in range(Nbeam):
        # print(dens[tidx,bidx,:].shape, alt[bidx,:].shape)
        condition = ((dens[tidx,bidx,:]>=2*backgrnd[tidx]) & (alt[bidx,:]>=200*1000.) & (alt[bidx,:]<=500*1000.))

        # if sum(condition)>=3:

        conv = np.convolve(condition.astype(int), np.ones(3), mode='valid')
        patch_beam += np.sum(conv>=3)


        # # Count the length of each grouping where the condition is true
        # # Copied from: https://stackoverflow.com/questions/24342047/count-consecutive-occurences-of-values-varying-in-length-in-a-numpy-array
        # group_len = np.diff(np.where(np.concatenate(([condition[0]], condition[:-1] != condition[1:],[True])))[0])[::2]
        # patch_beam += len(group_len>=3)

#             if np.any(condition):
#                 # print(group_len)
#             # if np.any(group_len>=3):
#             #     patch_beam += 1.
#             #     # print(condition, group_len)

#                 print('THRESHOLD', 2*backgrnd[tidx], conv, np.sum(conv>=3.))
#                 for a, d, c in zip(alt[bidx,:], dens[tidx, bidx, :], condition.astype(int)):
#                     print(a, d, c)

    # patch_index[tidx] = patch_beam/Ngates*100.
    patch_index[tidx] = patch_beam


# create plot of electron density of patch count
fig = plt.figure()
ax = fig.add_subplot(211)
c = ax.pcolormesh(time[:,0], alt[22,np.isfinite(alt[22,:])]/1000., dens[:,22,np.isfinite(alt[22,:])].T, cmap='jet', vmin=0., vmax=3.e11, zorder=2)
ax.set_xlim([np.datetime64('2010-03-11T03:00:00'),np.datetime64('2010-03-11T10:00:00')])
ax.set_ylim([200., 500.])
cax = fig.add_axes([0.95,0.5,0.01,0.4])
plt.colorbar(c, cax=cax)

ax = fig.add_subplot(212)
ax.plot(time[:,0], patch_index)
ax.set_xlim([np.datetime64('2010-03-11T03:00:00'),np.datetime64('2010-03-11T10:00:00')])

plt.show()
