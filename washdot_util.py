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

# function to apply A-weighting to a broadband measurement. Interpolates from
# third-octave band A-weigfhting function
def Aweight_interp(f):
    Aw,C = Aweight()
    Awinterp = np.interp(f,C,Aw)
    return Awinterp

def weighting(Lo,F,low = 0):
    #Apply weighting function to frequencies in F
    Aw,C = Aweight()
    Lw = Lo+Aw[np.in1d(C.ravel(), F).reshape(C.shape)]
    return Lw

def rms_calc(dat):
    rms = np.sqrt(np.mean(dat**2))
    return rms
    
def calibration_correction(calfile,Lrms = 94.,Pref = 20.):
    #correction to convert from .wav file (normalized to -1/+1) to units of
    #pressure. Lrms is the calibration level and Pref is the reference 
    #pressure
    cal, fs = sf.read(calfile) # load data
    tcal = np.arange(0,np.size(cal))/fs # time vector
    Prms = Pref*10.**(Lrms/20.) #rms pressure for calibrator
    #plot the calibration file to select the range of interest
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
    try:
        Srms = rms_calc(cal[np.logical_and(tcal>=tlim[0][0],tcal<=tlim[-1][0])])
    except KeyError:
        print ('try again doofus' )
        tlim = fig.ginput(2,show_clicks=True,mouse_add = 1)
    plt.close()
    Srms = rms_calc(cal[tcal<=5.])
    RVS = Prms/Srms #conversion from normalied units to muPa (so that P = S*Prms/Srms)
    return RVS,[tlim[0][0],tlim[-1][0]]

if __name__ == "__main__":
    Lt,Ct,Ut = third_octave()
    L,C,U = octave_band()
    Aw,C = Aweight()
    Lo = np.array([0.,0.,0.])
    F = np.array([16.,1000.,20000.])
    Lw = weighting(Lo,F) 
    print(Lw)
    