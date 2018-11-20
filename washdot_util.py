#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:10:21 2018
AUTHOR: Alexander Soloway
DESCRIPTION: utility file containing functions to analyze data collected
during the SR 520 bridge project with WASHDOT.
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import signal
import soundfile as sf
import sys
import os
import glob
import shutil
import pandas as pd

#third octave bands from "Noise and Vibration Control Engineering" 2nd ed.,
#Ver and Beranek, (2005) p.9 Table 1.1
def third_octave():
    L = np.array([14.1,17.8,22.4,28.2,35.5,44.7,56.2,70.8,89.1,112.,141., \
                  178.,224.,282.,355.,447.,562.,708.,891.,1122.,1413.,1778., \
                  2239.,2818.,3548.,4467.,5623.,7079.,8913.,11220.,14130., \
                  17780.])
    C = np.array([16.,20.,25.,31.5,40.,50.,63.,80.,100.,125.,160.,200.,250., \
                  315.,400.,500.,630.,800.,1000.,1250.,1600.,2000.,2500., \
                  3150.,4000.,5000.,6300.,8000.,10000.,12500.,16000.,20000.])
    U = np.array([17.8,22.4,28.2,35.5,44.7,56.2,70.8,89.1,112.,141.,178., \
                  224.,282.,355.,447.,562.,708.,891.,1122.,1413.,1778., \
                  2239.,2818.,3548.,4467.,5623.,7079.,8913.,11220., 14130., \
                  17780.,22390.])
    return L,C,U

#octave bands from "Noise and Vibration Control Engineering" 2nd ed.,
#Ver and Beranek, (2005) p.9 Table 1.1
def octave_band():
    L = np.array([11.,22.,44.,88.,177.,355.,710.,1420.,2840.,5680.,11360.])
    C = np.array([16.,31.5,63.,125.,250.,500.,1000.,2000.,4000.,16000.])
    U = np.array([22.,44.,88.,177.,355.,710.,1420.,2840.,5680.,11360.,22720.])
    return L,C,U

#A-weighting from "Noise and Vibration Control Engineering" 2nd ed.,
#Ver and Beranek, (2005) p.16 Table 1.4
def Aweight(low = 0):
    if low == 1:
        #Ver and Beranek A weight include 10 Hz and 12.5 Hz which are not 
        # typically included in 1/3 octave bamd. Option "low = 1" returns 
        #A-weight with these lower frequencies
        C = np.array([10., 12.5, 16.,20.,25.,31.5,40.,50.,63.,80.,100.,125., \
                      160.,200.,250.,315.,400.,500.,630.,800.,1000.,1250., \
                      1600.,2000.,2500.,3150.,4000.,5000.,6300.,8000.,10000., \
                      12500.,16000.,20000.])
        Aw = np.array([-70.4,-63.7,-56.7,-50.5,-44.7,-39.4,-34.6,-30.2, \
                            -26.2,-22.5,-19.1,-16.1,-13.4,-10.9,-8.6,-6.6, \
                            -4.8,-3.2,-1.9,-0.8,-0.,0.6,1.0,1.2,1.3,1.2,1.0,
                            0.5,-0.1,-1.1,-2.5,-4.3,-6.6,-9.3])
                      
    else:
        C = np.array([16.,20.,25.,31.5,40.,50.,63.,80.,100.,125.,160.,200., \
                      250.,315.,400.,500.,630.,800.,1000.,1250.,1600.,2000., \
                      2500.,3150.,4000.,5000.,6300.,8000.,10000.,12500., \
                      16000.,20000.])
        Aw = np.array([-56.7,-50.5,-44.7,-39.4,-34.6,-30.2, \
                    -26.2,-22.5,-19.1,-16.1,-13.4,-10.9,-8.6,-6.6, \
                    -4.8,-3.2,-1.9,-0.8,-0.,0.6,1.0,1.2,1.3,1.2,1.0,
                    0.5,-0.1,-1.1,-2.5,-4.3,-6.6,-9.3])
    return Aw,C

#C-weighting from "Noise and Vibration Control Engineering" 2nd ed.,
#Ver and Beranek, (2005) p.16 Table 1.4
def Cweight(low = 0):
    if low == 1:
        #Ver and Beranek A weight include 10 Hz and 12.5 Hz which are not 
        # typically included in 1/3 octave bamd. Option "low = 1" returns 
        #A-weight with these lower frequencies
        C = np.array([10., 12.5, 16.,20.,25.,31.5,40.,50.,63.,80.,100.,125., \
                      160.,200.,250.,315.,400.,500.,630.,800.,1000.,1250., \
                      1600.,2000.,2500.,3150.,4000.,5000.,6300.,8000.,10000., \
                      12500.,16000.,20000.])
        Aw = np.array([-14.3,-11.2,-8.5,-6.2,-4.4,-3.0,-2.0,-1.3,-0.8,-0.5, \
                       -0.3,-0.2,-0.1,0,0,0,0,0,0,0,0,0,-0.1,-0.2,-0.3,-0.5, \
                       -0.8,-1.3,-2.0,-3.0,-4.4,-6.2,-8.5,-11.2])
                      
    else:
        C = np.array([16.,20.,25.,31.5,40.,50.,63.,80.,100.,125.,160.,200., \
                      250.,315.,400.,500.,630.,800.,1000.,1250.,1600.,2000., \
                      2500.,3150.,4000.,5000.,6300.,8000.,10000.,12500., \
                      16000.,20000.])
        Aw = np.array([-8.5,-6.2,-4.4,-3.0,-2.0,-1.3,-0.8,-0.5, \
                       -0.3,-0.2,-0.1,0,0,0,0,0,0,0,0,0,-0.1,-0.2,-0.3,-0.5, \
                       -0.8,-1.3,-2.0,-3.0,-4.4,-6.2,-8.5,-11.2])
    return Aw,C

# function to apply A-weighting to a broadband measurement. Interpolates from
# third-octave band A-weigfhting function
def Aweight_interp(f):
    Aw,C = Aweight()
    Awinterp = np.interp(f,C,Aw)
    return Awinterp

#Aweight_broadband() is the equation version of the Aweighting functions
# Instead of interpolation. Source is "Handbook of Noise and Vibration" 
# Crocker (2008),John Wiley & Sons, Inc, pp.459 Eq.(1) and Eq.(2)
def Aweight_broadband(f,weight='A'):
    f1 = 20.60
    f2 = 107.7
    f3 = 737.9
    f4 = 12194.0
    WA1000 = -2.00
    WC1000 = -0.062
    if weight=='A':
        Aw = 20.*np.log10((f4**2*f**4)/((f**2+f1**2)*np.sqrt(f**2+f2**2) \
                            *np.sqrt(f**2+f3**2)*(f**2+f4**2)))-WA1000
    elif weight=='C':
        Aw = Aw = 20.*np.log10((f4**2*f**4)/ \
                               ((f**2+f1**2)*np.sqrt(f**2+f4**2))) - WC10000
    else:
        Aw = np.zeros(np.size(f))
    return Aw

def Cweight_interp(f):
    Cw,C = Cweight()
    Cwinterp = np.interp(f,C,Cw)
    return Cwinterp

def weighting(Lo,F,low = 0):
    #Apply weighting function to frequencies in F
    Aw,C = Aweight()
    Lw = Lo+Aw[np.in1d(C.ravel(), F).reshape(C.shape)]
    return Lw

def rms_calc(dat,unit='flat',Pref=20.):
    #compute rms value
    if unit=='dB':
        rms = 20*np.log10(np.sqrt(np.mean(dat**2))/Pref)
    else:
        rms = np.sqrt(np.mean(dat**2))   
    return rms

def third_octave_calc(Pf2,f,weight=False,Pref=20.):
    L,C,U = third_octave()
    Pthird =[]
    for (isum,fsum) in enumerate(C):
        Pthird.append(np.sum(Pf2[np.logical_and(f>=L[isum],f<=U[isum])])/(U[isum]-L[isum]))
    Lthird = 10.*np.log10(np.array(Pthird)/Pref**2.)
    if weight is True:
        Aw, _ = Aweight()
        Lthird = Lthird+Aw
    return Lthird, C

def calibration_correction(calfile,Lrms = 94.,Pref = 20.,tlim = False):
    #correction to convert from .wav file (normalized to -1/+1) to units of
    #pressure. Lrms is the calibration level and Pref is the reference 
    #pressure
    cal, fs = sf.read(calfile) # load data
    tcal = np.arange(0,np.size(cal))/fs # time vector
    Wn = np.array([900.,1100.])/(fs/2.)
    b, a = signal.butter(2,Wn,btype='bandpass')
    cal = signal.filtfilt(b,a,cal)
    Prms = Pref*10.**(Lrms/20.) #rms pressure for calibrator
    #plot the calibration file to select the range of interest
    if tlim == False:
        fig, ax = plt.subplots(figsize=(5, 3),dpi=200)
        ax.grid(alpha = 1.0,color = 'k',linewidth = '0.25')
        ax.plot(tcal, cal,linewidth = '0.25')
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Time (s)')
        ax.set_xlim(tcal[0],tcal[-1])
        fig.tight_layout()
        print("Select the start then end time used to calculate the correction:")
        tlim = fig.ginput(2,show_clicks=True,mouse_add = 1)   
        if tlim[0][0]>tlim[-1][0]:
            print("start time must be less than end time, try again")
            tlim = fig.ginput(2,show_clicks=True,mouse_add = 1)
        Srms = rms_calc(cal[np.logical_and(tcal>=tlim[0][0],tcal<=tlim[-1][0])])
        plt.close()
    else:
        #Srms = rms_calc(cal[np.logical_and(tcal>=tlim[0],tcal<=tlim[1])])
        caldex = [np.logical_and(tcal>=tlim[0],tcal<=tlim[1])]

        Srms = rms_calc(cal[tuple(caldex)])
    RVS = Prms/Srms #conversion from normalized units to muPa (so that P = S*Prms/Srms)
    return RVS,tlim

def calibration_correction2(CALFILE,tlim,Lrms = 94.,Pref = 20.):
    #correction to convert from .wav file (normalized to -1/+1) to units of
    #pressure. Lrms is the calibration level and Pref is the reference 
    #pressure
    dex=0
    RVS = np.zeros([2])
    for cf in CALFILE:
        Lrms = 94.
        cal, fs = sf.read(cf) # load data
        tcal = np.arange(0,np.size(cal[:,dex]))/fs # time vector
        Wn = np.array([900.,1100.])/(fs/2.)
        b, a = signal.butter(2,Wn,btype='bandpass')
        cal = signal.filtfilt(b,a,cal[:,dex])
        Prms = Pref*10.**(Lrms/20.) #rms pressure for calibrator
        caldex = [np.logical_and(tcal>=tlim[0],tcal<=tlim[1])]
        Srms = rms_calc(cal[tuple(caldex)])
        RVS[dex] = Prms/Srms
        dex+=1
    return RVS

def datload(FILENAME):
    data, fs = sf.read(FILENAME) # load data
    t = np.arange(0,np.shape(data)[0])/fs # time vector
    data = data - np.mean(data)
    return data,t,fs

def zoom_rename(rootDIR):
    #add '/' to end of directory name if not already there
    if rootDIR[-1]!='/':
        rootDIR = rootDIR+'/'
    #return list of directories in main directory
    zoomDIR = []
    for xx in os.listdir(rootDIR):
        if os.path.isdir(rootDIR+'/'+xx)==True:
            zoomDIR.append(xx)
    #for each ZOOM directory in a folder, copy and rename the .WAV file with 
    #the .hprj filename (date+time) then move to root directory   
    for subDIR in zoomDIR:
        for file in os.listdir(rootDIR+subDIR):
            if file.endswith(".wav") or file.endswith(".WAV"):
                newfile = os.path.basename(glob.glob(rootDIR+subDIR+'/'+'*.hprj')[0][0:-5] \
                    .replace('-','_')+'.WAV')
                shutil.copy(rootDIR+subDIR+'/'+file,rootDIR+'/'+newfile)
                
def bk2270_rename(ROOTDIR,SAVEDIR):
    if ROOTDIR[-1]!='/':
        ROOTDIR = ROOTDIR+'/'
    for file in os.listdir(ROOTDIR):
        if file.endswith(".wav") or file.endswith(".WAV"):
            newfile = pd.to_datetime(os.path.getmtime(ROOTDIR+file),unit='s'). \
                tz_localize('UTC').tz_convert('US/Pacific').strftime('%Y_%m_%d_%H_%M_%S')
            shutil.copy(ROOTDIR+file,SAVEDIR+'/'+newfile+'.wav')


if __name__ == "__main__":
    Lt,Ct,Ut = third_octave()
    L,C,U = octave_band()
    Aw,C = Aweight()
    Lo = np.array([0.,0.,0.])
    F = np.array([16.,1000.,20000.])
    Lw = weighting(Lo,F) 
    print(Lw)
    