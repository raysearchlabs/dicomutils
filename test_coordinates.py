#!/usr/bin/python
from coordinates import *

import nose
import numpy as np

def all_rots(f):
    return all([np.abs(np.linalg.norm(f(i)[:3,:3]) - np.sqrt(3)) < np.sqrt(3)*(2.0**-50) for i in range(-360,360)])

def v(*x):
    if len(x) == 3:
        return np.array([x[0],x[1],x[2],1]).reshape((4,1))
    #elif len(x) == 4:
    #    return np.array([x[0],x[1],x[2],x[3]]).reshape((4,1))
    assert False

def eq(a, b):
    return np.linalg.norm(a-b) < (1 + np.linalg.norm(a) * np.linalg.norm(b)) * 2.0**-40

def test_Mfs():
    assert (Mfs(0) == np.eye(4)).all()
    assert all_rots(lambda i: Mfs(i))
    assert eq(Mfs(90) * v(1,0,0), v(0,-1,0))
    assert eq(Mfs(90) * v(0,1,0), v(1,0,0))

def test_Mse():
    assert (Mse(0,0) == np.eye(4)).all()
    assert all_rots(lambda i: Mse(0, i))
    assert eq(Mse(13.8,0) * v(0,13.8,0), v(0,0,0))
    assert eq(Mse(42.9,90) * v(0,43.9,0), v(1,0,0))

def test_Met():
    assert (Met(0,0,0,0) == np.eye(4)).all()
    assert all_rots(lambda i: Met(0, 0, 0, i))
    assert eq(Met(4,9,25,0) * v(4,9,25), v(0,0,0))
    assert eq(Met(4,9,25,90) * v(4,9,25+1), v(0,1,0))
    assert eq(Met(4,9,25,90) * v(4,9+1,25), v(0,0,-1))
    
def test_Mtp():
    assert (Mtp(0,0,0,0,0,0) == np.eye(4)).all()
    assert all_rots(lambda i: Mtp(0, 0, 0, i, 41, 12))
    assert all_rots(lambda i: Mtp(0, 0, 0, 19, i, 41))
    assert all_rots(lambda i: Mtp(0, 0, 0, -81, 93, i))
    assert eq(Mtp(41,13,95,0,0,0) * v(41, 13, 95), v(0,0,0))
    assert eq(Mtp(41,13,95,90,0,0) * v(41, 13+1, 95), v(0,0,-1))
    assert eq(Mtp(41,13,95,90,0,0) * v(41, 13, 95+1), v(0,1,0))
    assert eq(Mtp(41,13,95,0,90,0) * v(41+1, 13, 95), v(0,0,1))
    assert eq(Mtp(41,13,95,0,90,0) * v(41, 13, 95+1), v(-1,0,0))
    assert eq(Mtp(41,13,95,0,0,90) * v(41+1, 13, 95), v(0,-1,0))
    assert eq(Mtp(41,13,95,0,0,90) * v(41, 13+1, 95), v(1,0,0))

def test_Mfg():
    assert (Mfg(0) == np.eye(4)).all()
    assert eq(Mfg(90) * v(1,0,0), v(0,0,1))
    assert eq(Mfg(90) * v(0,1,0), v(0,1,0))
    assert eq(Mfg(90) * v(0,0,1), v(-1,0,0))
    assert all_rots(lambda i: Mfg(i))

def test_Mgb():
    assert (Mgb(0,0) == np.eye(4)).all()
    assert eq(Mgb(100,0) * v(0,0,100), v(0,0,0))
    assert eq(Mgb(100,90) * v(1,0,100), v(0,-1,0))
    assert eq(Mgb(100,90) * v(0,1,100), v(1,0,0))
    assert all_rots(lambda i: Mgb(0, i))
    
def test_Mbw():
    assert (Mbw(0,0) == np.eye(4)).all()
    assert eq(Mbw(17.2, 0) * v(0,0,17.2), v(0,0,0))
    assert eq(Mbw(17.2, 90) * v(0,1,17.2), v(1,0,0))    
    assert eq(Mbw(17.2, 90) * v(1,0,17.2), v(0,-1,0))    
    assert eq(Mbw(17.2, 90) * v(0,0,17.2+1), v(0,0,1))

def test_Mgr():
    assert (Mgr(0, 0, 0, 0) == np.eye(4)).all()
    assert eq(Mgr(7, 19, 27, 0) * v(7,19,27), v(0,0,0))
    assert eq(Mgr(0,0,0,90) * v(1,0,0), v(0,-1,0))
    assert eq(Mgr(0,0,0,90) * v(0,1,0), v(1,0,0))
    assert eq(Mgr(0,0,0,90) * v(0,0,1), v(0,0,1))

    

