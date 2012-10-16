import numpy as np
debug = False

if not debug:
    from math import sin, cos, pi
    def n(x):
        return float(x)
else:    
    from sympy import sin, cos, pi, Symbol
    tg, tb, tw, ts, te, Ey, Ty, Bz, Wz = [Symbol(s) for s in ['tg', 'tb', 'tw', 'ts', 'te', 'Ey', 'Ty', 'Bz', 'Wz']]
    def n(x):
        return x

def rotX(psi):
    return np.matrix([[1, 0, 0, 0],
                      [0, cos(n(psi)*pi/180), sin(n(psi)*pi/180), 0],
                      [0, -sin(n(psi)*pi/180), cos(n(psi)*pi/180), 0],
                      [0, 0, 0, 1]])
def rotY(phi):
    return np.matrix([[cos(n(phi)*pi/180), 0, sin(n(phi)*pi/180), 0],
                      [0, 1, 0, 0],
                      [-sin(n(phi)*pi/180), 0, cos(n(phi)*pi/180), 0],
                      [0, 0, 0, 1]])
def rotZ(theta):
    return np.matrix([[cos(n(theta)*pi/180), sin(n(theta)*pi/180), 0, 0],
                      [-sin(n(theta)*pi/180), cos(n(theta)*pi/180), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

def translate(x,y,z):
    return np.matrix([[1,0,0,n(x)],
                      [0,1,0,n(y)],
                      [0,0,1,n(z)],
                      [0,0,0,1]])

def invert4x4fast(m):
    """Actually slower and less exact than numpy.linalg.inv()."""
    m = m.copy()
    m[1,0], m[0,1] = m[0,1], m[1,0]
    m[2,0], m[0,2] = m[0,2], m[2,0]
    m[2,1], m[1,2] = m[1,2], m[2,1] 
    m03 = -(m[0,0] * m[0,3] + m[0,1] * m[1,3] + m[0,2] * m[2,3])
    m13 = -(m[1,0] * m[0,3] + m[1,1] * m[1,3] + m[1,2] * m[2,3])
    m[2,3] = -(m[2,0] * m[0,3] + m[2,1] * m[1,3] + m[2,2] * m[2,3])
    m[1,3] = m13
    m[0,3] = m03
    return m

def Mfs(theta_s):
    """Transform from fixed to patient support coordinate system."""
    return rotZ(theta_s)
def Mse(Ls, theta_e):
    """Transform from patient support to table top eccentric coordinate system."""
    return rotZ(theta_e) * translate(0, -Ls, 0)
def Met(Tx, Ty, Tz, psi_t):
    """Transform from table top eccentric to table top coordinate system."""
    return rotX(psi_t) * translate(-Tx, -Ty, -Tz)
def Mtp(Px, Py, Pz, psi_p, phi_p, theta_p):
    """Transform from table top to patient coordinate system."""
    return rotZ(theta_p) * rotY(-phi_p) * rotX(psi_p) * translate(-Px, -Py, -Pz)
def Mfg(phi_g):
    """Transform from fixed to gantry coordinate system."""
    return rotY(-phi_g)
def Mgb(SAD, theta_b):
    """Transform from gantry to beam limiting device or delineator coordinate system."""
    return rotZ(theta_b) * translate(0, 0, -SAD)
def Mbw(Wz, theta_w):
    """Transform from beam limiting device or delineator to wedge filter coordinate system."""
    return rotZ(theta_w) * translate(0, 0, -Wz)
def Mgr(Rx, Ry, Rz, theta_r):
    """Transform from gantry to X-ray image receptor coordinate system."""
    return rotZ(theta_r) * translate(-Rx, -Ry, -Rz)
def Mpd():
    """Transform from patient to DICOM patient coordinate system."""
    return np.matrix([[1, 0, 0, 0],
                      [0, 0, -1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])

def transform3d(v3, m):
    assert v3.shape[0] == 3
    return (m * np.vstack((v3, np.ones((1,v3.shape[1])))))[:3,:]
coordinate_systems = [
    ("Fixed", "f", None),
    ("Gantry", "g", "f", Mfg),
    ("Beam limiting device or delineator", "b", "g", Mgb),
    ("Wedge filter", "w", "b", Mbw),
    ("X-ray image receptor", "r", "g", Mgr),
    ("Patient support", "s", "f", Mfs),
    ("Table top eccentric rotation", "e", "s", Mse),
    ("Table top", "t", "e", Met),
    ("Patient", "p", "t", Mtp),
    ("DICOM Patient", "d", "p", Mpd) # Non-standard
    ]
    
