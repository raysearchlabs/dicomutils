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
    return np.matrix([[cos(n(phi)*pi/180), 0, -sin(n(phi)*pi/180), 0],
                      [0, 1, 0, 0],
                      [sin(n(phi)*pi/180), 0, cos(n(phi)*pi/180), 0],
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
def Met(Tx, Ty, Tz, psi_t, phi_t):
    """Transform from table top eccentric to table top coordinate system."""
    # The order of rotations must be the same as the rotations are described in IEC61217
    return rotY(phi_t) * rotX(psi_t) * translate(-Tx, -Ty, -Tz)
def Mtp(Px, Py, Pz, psi_p, phi_p, theta_p):
    """Transform from table top to patient coordinate system."""
    # The order of rotations must be the same as the rotations are described in IEC61217
    return rotZ(theta_p) * rotY(phi_p) * rotX(psi_p) * translate(-Px, -Py, -Pz)
def Mfg(psi_g, phi_g):
    """Transform from fixed to gantry coordinate system, plus the non-standard DICOM gantry pitch rotation psi_g."""
    return rotX(psi_g) * rotY(phi_g)
def Mgb(Bz, theta_b):
    """Transform from gantry to beam limiting device or delineator coordinate system."""
    return rotZ(theta_b) * translate(0, 0, -Bz)
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

class Coordinates(object):
    def __init__(self, theta_s = 0, 
                 Ls = 0, theta_e = 0, 
                 Tx = 0, Ty = 0, Tz = 0, psi_t = 0, phi_t = 0, 
                 Px = 0, Py = 0, Pz = 0, psi_p = 0, phi_p = 0, theta_p = 0,
                 psi_g = 0, phi_g = 0,
                 Bz = 0, theta_b = 0,
                 Wz = 0, theta_w = 0,
                 Rx = 0, Ry = 0, Rz = 0, theta_r = 0):
        self.theta_s = theta_s
        self.Ls = Ls
        self.theta_e = theta_e
        self.Tx = Tx
        self.Ty = Ty
        self.Tz = Tz
        self.psi_t = psi_t
        self.phi_t = phi_t
        self.Px = Px
        self.Py = Py
        self.Pz = Pz
        self.psi_p = psi_p
        self.phi_p = phi_p
        self.theta_p = theta_p
        self.psi_g = psi_g
        self.phi_g = phi_g
        self.Bz = Bz
        self.theta_b = theta_b
        self.Wz = Wz
        self.theta_w = theta_w
        self.Rx = Rx
        self.Ry = Ry
        self.Rz = Rz
        self.theta_r = theta_r

    def get_parents(self, from_system):
        for c in coordinate_systems:
            if c[1] == from_system and c[2] != None:
                return [c[2]] + self.get_parents(c[2])
        return []

    def fill_out(self, transformation):
        return transformation(**{k:v for k,v in self.__dict__.iteritems() if k in transformation.func_code.co_varnames})

    def get_transformation(self, from_system, to_system):
        M = np.eye(4)
        parents_of_from = self.get_parents(from_system)
        if to_system in parents_of_from:
            for c in [from_system] + parents_of_from:
                transform = [ct[3] for ct in coordinate_systems if ct[1] == c][0]
                M = np.linalg.inv(self.fill_out(c[3])) * M
            return M
        parents_of_to = self.get_parents(to_system)
            
        
                 

def transform3d(v3, m):
    assert v3.shape[0] == 3
    return (m * np.vstack((v3, np.ones((1,v3.shape[1])))))[:3,:]

coordinate_systems = [
    ("Fixed", "f", None, None),
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
    
