from mayavi import mlab
from traits.api import HasTraits, on_trait_change, Array, CFloat
from traitsui.api import View
import numpy as np
from .. import coordinates
from ..coordinates import transform3d

class MLC(HasTraits):
    """ A class that shows an MLC (Multi-Leaf Collimator)
    """

    source_distance = CFloat(500.0, 
                             desc='distance from the source to the furthest point on the MLC along the beam axis (bottom of the mlc)',
                             enter_set=True, auto_set=False)
    leaf_positions = Array(float, value=np.zeros((2,1)), shape=(2,(1,None)), desc='the dynamic positions of the leaves',
                           enter_set=True, auto_set=False)
    leaf_boundaries = Array(float, value=((0,1),), shape=(1,(2,None)), desc='the boundaires of the leaves, in the direction orthogonal to the movement',
                           enter_set=True, auto_set=False)
    beam_limiting_device_angle = CFloat(0.0, desc="Beam Limiting Device Angle", 
                           enter_set=True, auto_set=False)
    gantry_angle = CFloat(0.0, desc="Gantry Angle", 
                           enter_set=True, auto_set=False)
    gantry_pitch_angle = CFloat(0.0, desc="Gantry Pitch Angle", 
                           enter_set=True, auto_set=False)
    sad = CFloat(1000.0, desc="Source-to-axis distance, in mm", 
                           enter_set=True, auto_set=False)
    thickness = CFloat(70.0, desc="MLC Thickness",
                       enter_set=True, auto_set=False)

    _trimesh = None

    view = View('source_distance', 'leaf_positions', 'leaf_boundaries',
                'beam_limiting_device_angle', 'gantry_angle', 'gantry_pitch_angle', 
                'sad', 'thickness', '_')

    @on_trait_change('source_distance,leaf_positions,leaf_boundaries,beam_limiting_device_angle,gantry_angle,gantry_pitch_angle,sad,thickness')
    def redraw(self):
        if hasattr(self, 'app') and self.app.scene._renderer is not None:
            self.display()
            #self.app.visualize_field()

    def display(self):
        """
        Display the MLC in the 3D view.
        """
        
        def leaf(pts, polys, xspan, yspan, zspan, sad):
            """ spans are tuples (min, max) """
            ptsi = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]
            i0 = len(pts)
            for pt in ptsi:
                zp = zspan[pt[2]]
                xp = xspan[pt[0]] * zp / -sad
                yp = yspan[pt[1]] * zp / -sad
                
                pts.append([xp, yp, zp])
                            
            polys += [[xi+i0, yi+i0, zi+i0] for xi,yi,zi in 
                      [[2,1,0],[1,2,3],[4,5,6],[7,6,5],
                       [4,1,0],[1,4,5],[2,3,6],[7,6,3],
                       [4,2,0],[2,4,6],[1,3,5],[7,5,3]]]
            return pts, polys

        pts = []
        polys = []
        assert self.leaf_positions.shape[1]+1 == self.leaf_boundaries.shape[1]
        nleaves = self.leaf_boundaries.shape[1] - 1
        xmin = np.amin(self.leaf_positions) - 5
        xmax = np.amax(self.leaf_positions) + 5
        zspan = (-self.source_distance, -self.source_distance + self.thickness)
        for i in range(nleaves):
            leaf(pts, polys, (xmin, self.leaf_positions[0,i]), self.leaf_boundaries[0,i:i+2], zspan, self.sad)
            leaf(pts, polys, (self.leaf_positions[1,i], xmax), self.leaf_boundaries[0,i:i+2], zspan, self.sad)

        polys = np.array(polys)
        pts = np.array(pts)

        # pts are now in BLD coordinates. Transform to fixed coordinate system:
        M = (np.linalg.inv(coordinates.Mfg(self.gantry_pitch_angle, self.gantry_angle))
             * np.linalg.inv(coordinates.Mgb(self.sad, self.beam_limiting_device_angle)))

        pts = transform3d(pts.T, M).T

        if self._trimesh is None:
            self._trimesh = mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], polys, color=(.5,.5,1))        
        else:
            self._trimesh.mlab_source.set(x=pts[:,0], y=pts[:,1], z=pts[:,2], triangles=polys)

    def move_camera_to_bev(self):
        # Okay, so there is a better solution to inclination and azimuth, but in
        # the interest of getting things done, I leave this as an 
        # exercise to the reader to solve exactly.
        M = (np.linalg.inv(coordinates.Mfg(self.gantry_pitch_angle, self.gantry_angle))
             * np.linalg.inv(coordinates.Mgb(self.sad, self.beam_limiting_device_angle)))
        p = transform3d([0,0,0], M)
        inclination = np.arccos(p[2]/np.linalg.norm(p)) * 180 / np.pi
        azimuth = np.arctan2(p[1], p[0]) * 180 / np.pi
        mlab.view(azimuth = azimuth[0], elevation = inclination[0], focalpoint = [0,0,0],
                  distance = self.sad, roll = -self.beam_limiting_device_angle)        
        
def _test(mlc=None):
    import dicom
    import os
    rtplan = dicom.read_file(os.path.join(os.path.dirname(__file__), "..", "RTPLAN.dcm"))
    beam = rtplan.Beams[0]
    cp = beam.CPs[0]
    lp = cp.BLDPositions[2].LeafJawPositions
    lpb = np.atleast_2d(beam.BLDs[2].LeafPositionBoundaries)
    nleaves = lpb.shape[1] - 1
    if mlc == None:
        mlc=MLC()
    mlc.set(leaf_boundaries = lpb,
            leaf_positions = np.array([x for x in lp]).reshape((2,nleaves)),
            beam_limiting_device_angle = cp.BeamLimitingDeviceAngle,
            gantry_angle = cp.GantryAngle,
            gantry_pitch_angle = getattr(cp, 'GantryPitchAngle', 0),
            sad = beam.SourceAxisDistance,
            source_distance = beam.SourceAxisDistance / 2.0,
            thickness = beam.SourceAxisDistance * 0.05,
            )

    mlc.display()


if __name__ == '__main__':
    _test()
