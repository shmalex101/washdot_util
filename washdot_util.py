#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:10:21 2018
AUTHOR: Alexander Soloway
DESCRIPTION: utility file containing functions to analyze data collected
during the SR 520 bridge project with WASHDOT.
"""
import numpy as np

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


if __name__ == "__main__":
    Lt,Ct,Ut = third_octave()
    L,C,U = octave_band()
    print()