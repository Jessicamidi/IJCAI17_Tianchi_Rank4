# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:29:53 2017

@author: Flamingo
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import datetime
from textwrap import wrap
import xgboost as xgb
import copy

## return ralatetive date, from the START_DATE
def Datestr2DGap(ser,START_DATE):
    temp = pd.to_datetime(ser)- pd.to_datetime(START_DATE)
    ser_DGap = [(lambda x:(x.days)) (x) for x in temp ]
    return ser_DGap

## return day of week
def Datestr2DofW(ser):
    ser_DofW = pd.to_datetime(ser).dt.dayofweek
    return ser_DofW
    
def Const_Datestr(START_DATE, DATE_LEN):
    return[str((datetime.datetime.strptime(START_DATE,'%Y-%m-%d') + datetime.timedelta(days=x)).date()) for x in range( DATE_LEN)] 


def Const_Datestr2(START_DATE, END_DATE):
    day_N = (datetime.datetime.strptime(END_DATE,'%Y-%m-%d') - datetime.datetime.strptime(START_DATE,'%Y-%m-%d')).days+1
    return Const_Datestr(START_DATE, day_N)
    
def Const_Datestr3(Prefix, START_DATE, END_DATE):
    day_N = (datetime.datetime.strptime(END_DATE,'%Y-%m-%d') - datetime.datetime.strptime(START_DATE,'%Y-%m-%d')).days+1
    return_string = [(Prefix + str((datetime.datetime.strptime(START_DATE,'%Y-%m-%d') + datetime.timedelta(days=x)).date())) for x in range( day_N)] 
    return return_string
    
    
## datetime series to string series
def Datedate2Datestr(ser):
    return  [ (lambda x: str(x.date()) )(x) for x in ser ] 
    
def str2date(tt):
    return  datetime.datetime.strptime(tt,'%Y-%m-%d').date()
    
    
## generate final result
def Loss_Round(result):
    result_int = np.floor(result)
    if result**2 <= result_int**2+ result_int:
        result_back = np.floor(result)
    else:
        result_back = np.ceil(result)
    return result_back 


## Empirical Mode Decomposition


import numpy as np
import scipy.interpolate
import scipy.signal

def emd(data,max_modes=10):
    """Calculate the Emprical Mode Decomposition of a signal."""
    # initialize modes
    modes=[]
  
    # perform sifts until we have all modes
    residue=data
    while not _done_sifting(residue):
        # perform a sift
        imf,residue = _do_sift(residue)
        
        # append the imf
        modes.append(imf)

        # see if achieved max
        if len(modes) == max_modes:
            # we have all we wanted
            break
            
    # append the residue
    modes.append(residue)

    # return an array of modes
    return np.asarray(modes)

def eemd(data, noise_std=0.2, num_ensembles=100, num_sifts=10):
    """
    Ensemble Empirical Mode Decomposition (EEMD)

    *** Must still add in post-processing with EMD ***
    """
    # get modes to generate
    num_samples = len(data)
    num_modes = int(np.fix(np.log2(num_samples)))-1

    # normalize incomming data
    dstd = data.std()
    y = data/dstd
    
    # allocate for starting value
    all_modes = np.zeros((num_modes+2,num_samples))
    
    # loop over num_ensembles
    for e in range(num_ensembles):
        # perturb starting data
        x0 = y + np.random.randn(num_samples)*noise_std

        # save the starting value
        all_modes[0] += x0
        
        # loop over modes
        for m in range(num_modes):
            # do the sifts
            imf = x0
            for s in range(num_sifts):
                imf = _do_one_sift(imf)

            # save the imf
            all_modes[m+1] += imf
            
            # set the residual
            x0 = x0 - imf

        # save the final residual
        all_modes[-1] += x0
                
    # average everything out and renormalize
    return all_modes*dstd/np.float64(num_ensembles)
    
def _done_sifting(d):
    """We are done sifting is there a monotonic function."""
    return np.sum(_localmax(d))+np.sum(_localmax(-d))<=2

def _do_sift(data):
    """
    This function is modified to use the sifting-stopping criteria
    from Huang et al (2003) (this is the suggestion of Peel et al.,
    2005).  Briefly, we sift until the number of extrema and
    zerocrossings differ by at most one, then we continue sifting
    until the number of extrema and ZCs both remain constant for at
    least five sifts."""

    # save the data (may have to copy)
    imf=data

    # sift until num extrema and ZC differ by at most 1
    while True:
        imf=_do_one_sift(imf)
        numExtrema,numZC = _analyze_imf(imf)
        #print 'numextrema=%d, numZC=%d' %  (numExtrema, numZC) 
        if abs(numExtrema-numZC)<=1:
            break

    # then continue until numExtrema and ZCs are constant for at least
    # 5 sifts (Huang et al., 2003)
    numConstant = 0
    desiredNumConstant = 5
    lastNumExtrema = numExtrema
    lastNumZC = numZC
    while numConstant < desiredNumConstant:
        imf=_do_one_sift(imf)
        numExtrema,numZC = _analyze_imf(imf)
        if numExtrema == lastNumExtrema and \
                numZC == lastNumZC:
            # is the same so increment
            numConstant+=1
        else:
            # different, so reset
            numConstant = 0
        # save the last extrema and ZC
        lastNumExtrema = numExtrema
        lastNumZC = numZC
        
    # FIX THIS
#     while True:
#         imf = _do_one_sift(imf)
#         numExtrema[end+1],numZC[end+1] = _analyze_imf(imf)
#         print 'FINAL STAGE: numextrema=%d, numZC=%d' % (numExtrema(end), numZC(end))
#         if length(numExtrema)>=numConstant & \
#                 all(numExtrema(end-4:end)==numExtrema(end)) & \
#                 all(numZC(end-4:end)==numZC(end)):
#             break

    # calc the residue
    residue=data-imf

    # return the imf and residue
    return imf,residue


def _do_one_sift(data):

    upper=_get_upper_spline(data)
    lower=-_get_upper_spline(-data)
    #upper=jinterp(find(maxes),data(maxes),xs);
    #lower=jinterp(find(mins),data(mins),xs);

    #imf=mean([upper;lower],1)
    imf = (upper+lower)*.5

    detail=data-imf

    # plot(xs,data,'b-',xs,upper,'r--',xs,lower,'r--',xs,imf,'k-')

    return detail # imf


def _get_upper_spline(data):
    """Get the upper spline using the Mirroring algoirthm from Rilling et
al. (2003)."""

    maxInds = np.nonzero(_localmax(data))[0]

    if len(maxInds) == 1:
        # Special case: if there is just one max, then entire spline
        # is that number
        #s=repmat(data(maxInds),size(data));
        s = np.ones(len(data))*data[maxInds]
        return s

    # Start points
    if maxInds[0]==0:
        # first point is a local max
        preTimes=1-maxInds[1]
        preData=data[maxInds[1]]
    else:
        # first point is NOT local max
        preTimes=1-maxInds[[1,0]]
        preData=data[maxInds[[1,0]]]

    # end points
    if maxInds[-1]==len(data)-1:
        # last point is a local max
        postTimes=2*len(data)-maxInds[-2]-1;
        postData=data[maxInds[-2]];
    else:
        # last point is NOT a local max
        postTimes=2*len(data)-maxInds[[-1,-2]];
        postData=data[maxInds[[-1,-2]]]

    # perform the spline fit
    t=np.r_[preTimes,maxInds,postTimes];
    d2=np.r_[preData, data[maxInds], postData];
    #s=interp1(t,d2,1:length(data),'spline');
    # XXX verify the 's' argument
    # needed to change so that fMRI dat would work
    rep = scipy.interpolate.splrep(t,d2,s=.0)
    s = scipy.interpolate.splev(range(len(data)),rep)
    # plot(1:length(data),data,'b-',1:length(data),s,'k-',t,d2,'r--');  

    return s


def _analyze_imf(d):
    numExtrema = np.sum(_localmax(d))+np.sum(_localmax(-d))
    numZC = np.sum(np.diff(np.sign(d))!=0)
    return numExtrema,numZC

# % if debug
# %   clf
# %   a1=subplot(2,1,1);
# %   plot(xs,d,'b-',xs,upper,'k-',xs,lower,'k-');
# %   axis tight;
  
# %   a2=subplot(2,1,2);
# %   plot(xs,stopScore,'b-',[0 length(d)],[thresh1 thresh1],'k--',[0 length(d)],[thresh2 ...
# %                       thresh2],'r--');
# %   axis tight;
# %   xlabel(sprintf('score = %.3g',s));  
# %   linkaxes([a1 a2],'x')
# %   keyboard
  
# % end



# function yi=jinterp(x,y,xi);
# if length(x)==1
#   yi=repmat(y,size(xi));
# else
#   yi=interp1(x,y,xi,'spline');
# end

  

def _localmax(d):
    """Calculate the local maxima of a vector."""

    # this gets a value of -2 if it is an unambiguous local max
    # value -1 denotes that the run its a part of may contain a local max
    diffvec = np.r_[-np.inf,d,-np.inf]
    diffScore=np.diff(np.sign(np.diff(diffvec)))
                     
    # Run length code with help from:
    #  http://home.online.no/~pjacklam/matlab/doc/mtt/index.html
    # (this is all painfully complicated, but I did it in order to avoid loops...)

    # here calculate the position and length of each run
    runEndingPositions=np.r_[np.nonzero(d[0:-1]!=d[1:])[0],len(d)-1]
    runLengths = np.diff(np.r_[-1, runEndingPositions])
    runStarts=runEndingPositions-runLengths + 1

    # Now concentrate on only the runs with length>1
    realRunStarts = runStarts[runLengths>1]
    realRunStops = runEndingPositions[runLengths>1]
    realRunLengths = runLengths[runLengths>1]

    # save only the runs that are local maxima
    maxRuns=(diffScore[realRunStarts]==-1) & (diffScore[realRunStops]==-1)

    # If a run is a local max, then count the middle position (rounded) as the 'max'
    # CHECK THIS
    maxRunMiddles=np.round(realRunStarts[maxRuns]+realRunLengths[maxRuns]/2.)-1

    # get all the maxima
    maxima=(diffScore==-2)
    maxima[maxRunMiddles.astype(np.int32)] = True

    return maxima

#%make sure beginning & end are not local maxes
#%maxima([1 end])=false;


def calc_inst_info(modes,samplerate):
    """
    Calculate the instantaneous frequency, amplitude, and phase of
    each mode.
    """

    amp=np.zeros(modes.shape,np.float32);
    phase=np.zeros(modes.shape,np.float32);
    f=np.zeros(modes.shape,np.float32);

    for m in range(len(modes)):
        h=scipy.signal.hilbert(modes[m]);
        amp[m,:]=np.abs(h);
        phase[m,:]=np.angle(h);
        f[m,:] = np.r_[np.nan, 
                      0.5*(np.angle(-h[2:]*np.conj(h[0:-2]))+np.pi)/(2*np.pi) * samplerate,
                      np.nan]

        #f(m,:) = [nan 0.5*(angle(-h(t+1).*conj(h(t-1)))+pi)/(2*pi) * sr nan];
    
    # calc the freqs (old way)
    #f=np.diff(np.unwrap(phase[:,np.r_[0,0:len(modes[0])]]))/(2*np.pi)*samplerate

    # clip the freqs so they don't go below zero
    #f = f.clip(0,f.max())

    return f,amp,phase


