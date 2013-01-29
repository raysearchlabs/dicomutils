# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 09:14:42 2013

@author: rickarad
"""
import matplotlib.pyplot as pp
import numpy as np

import modules

def plot_roi_in_cp(beam, cp, roi_contour, study):
    contours = modules.get_contours_in_bld(beam, roi_contour, study)
    for c in contours[cp.ControlPointIndex]:
        pp.plot(list(c[0]) + [c[0][0]], list(c[1]) + [c[1][0]]) 
    
def plot_cp(beam, cp):
    plot_leaves(modules.getblds(beam.BLDs)['MLCX'].LeafPositionBoundaries,
                modules.getblds(cp.BLDPositions)['MLCX'].LeafJawPositions)
                
def plot_leaves(boundaries, positions):
    b = boundaries
    p = positions
    pp.barh(b[:-1], p[:60]-np.min(p)+1, np.diff(b)-0.1*np.min(np.diff(b)), np.min(p)-1, alpha=0.4)
    pp.barh(b[:-1], p[60:]-np.max(p)-1, np.diff(b)-0.1*np.min(np.diff(b)), np.max(p)+1, alpha=0.4)