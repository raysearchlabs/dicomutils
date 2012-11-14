import numpy as np
import modules
from collections import defaultdict
import dicom, os

class TableTop(object):
    def __init__(self, psi_t=0, phi_t=0, Tx=0, Ty=0, Tz=0):
        self.psi_t = psi_t
        self.phi_t = phi_t
        self.Tx = Tx
        self.Ty = Ty
        self.Tz = Tz

class TableTopEcc(object):
    def __init__(self, Ls=0, theta_e=0):
        self.Ls = Ls
        self.theta_e = theta_e

class StudyBuilder(object):
    def __init__(self, patient_position, patient_id, patients_name = "", patients_birthdate = ""):
        self.modalityorder = ["CT", "RTSTRUCT", "RTPLAN", "RTDOSE"]
        self.current_study = {}
        self.current_study['PatientID'] = patient_id
        self.current_study['PatientsName'] = patients_name
        self.current_study['PatientsBirthDate'] = patients_birthdate
        self.current_study['PatientPosition'] = patient_position
        self.seriesbuilders = defaultdict(lambda: [])
        self.built = False

    def build_ct(self, **kwargs):
        b = CTBuilder(self.current_study, **kwargs)
        self.seriesbuilders['CT'].append(b)
        return b

    def build_static_plan(self, **kwargs):
        b = StaticPlanBuilder(self.current_study, **kwargs)
        self.seriesbuilders['RTPLAN'].append(b)
        return b

    def build_structure_set(self, images, **kwargs):
        b = StructureSetBuilder(self.current_study, images=images, **kwargs)
        self.seriesbuilders['RTSTRUCT'].append(b)
        return b

    def build(self):
        if self.built:
            return self.datasets
        datasets = []
        for modality, sbs in self.seriesbuilders.iteritems():
            for sb in sbs:
                datasets += sb.Build()
        self.built = True
        self.datasets = datasets
        return self.datasets

    def write(self, outdir='.', print_filenames=True):
        for modality in self.modalityorder:
            for sb in self.seriesbuilders[modality]:
                for ds in sb.build():
                    dicom.write_file(os.path.join(outdir, ds.filename), ds)
                    if print_filenames:
                        print ds.filename

class CTBuilder(object):
    def __init__(self, current_study, voxels, voxel_size, corner=None, rescale_slope=1, rescale_intercept=-1024, column_direction=None, row_direction=None, slice_direction=None):
        self.voxels = voxels
        self.voxel_size = voxel_size
        self.rescale_slope = rescale_slope
        if corner == None:
            corner = -self.gridsize / 2.0
        self.corner = np.array(corner)
        self.rescale_intercept = rescale_intercept
        self.ct_data = np.zeros(self.voxels, dtype=np.int16)
        if column_direction == None or row_direction == None:
            assert column_direction == None and row_direction == None
            column_direction = [0,1,0]
            row_direction = [1,0,0]
        if slice_direction == None:
            slice_direction = np.cross(column_direction, row_direction)
        slice_direction = slice_direction / np.linalg.norm(slice_direction)
        self.ImageOrientationPatient = column_direction + row_direction
        self.slice_direction = slice_direction
        self.current_study = current_study
        self.built = False

    @property
    def gridsize(self):
        return np.array([self.voxels[0] * self.voxel_size[0],
                         self.voxels[1] * self.voxel_size[2],
                         self.voxels[2] * self.voxel_size[2]])

    def real_valueToStoredValue(self, real_value):
        return (real_value - self.rescale_intercept) / self.rescale_slope

    def clear(self, real_value = None, stored_value = None):
        if real_value != None:
            assert stored_value == None
            stored_value = (real_value - self.rescale_intercept) / self.rescale_slope
        self.ct_data[:] = stored_value

    def mgrid(self):
        x,y,z=np.mgrid[:self.voxels[0],:self.voxels[1],:self.voxels[2]]
        coldir = self.ImageOrientationPatient[:3]
        rowdir = self.ImageOrientationPatient[3:]
        slicedir = self.slice_direction
        x = self.corner[0] + (x + 0.5) * rowdir[0] + (y + 0.5) * coldir[0] + (z + 0.5) * slicedir[0]
        y = self.corner[1] + (x + 0.5) * rowdir[1] + (y + 0.5) * coldir[1] + (z + 0.5) * slicedir[1]
        z = self.corner[2] + (x + 0.5) * rowdir[2] + (y + 0.5) * coldir[2] + (z + 0.5) * slicedir[2]
        return x,y,z

    def add_sphere(self, radius, center, stored_value = None, real_value = None, mode = 'set'):
        if real_value != None:
            assert stored_value == None
            stored_value = (real_value - self.rescale_intercept) / self.rescale_slope
        x,y,z = self.mgrid()
        voxels = self.ct_data[(x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 <= radius**2]
        print "x: {0}-{1}".format(x.min(), x.max())
        print "y: {0}-{1}".format(y.min(), y.max())
        print "z: {0}-{1}".format(z.min(), z.max())
        print "sum voxels: {0}".format((voxels == voxels).sum())
        if mode == 'set':
            voxels = stored_value
        elif mode == 'add':
            voxels += stored_value
        elif mode == 'subtract':
            voxels -= stored_value
        else:
            assert 'unknown mode'

    def add_box(self, size, center, stored_value = None, real_value = None, mode = 'set'):
        if real_value != None:
            assert stored_value == None
            stored_value = (real_value - self.rescale_intercept) / self.rescale_slope
        x,y,z = self.mgrid()
        voxels = self.ct_data[(abs(x-center[0]) <= size[0]/2.0) * (abs(y-center[1]) <= size[1]/2.0) * (abs(z-center[2]) <= size[2]/2.0)]
        if mode == 'set':
            voxels = stored_value
        elif mode == 'add':
            voxels += stored_value
        elif mode == 'subtract':
            voxels -= stored_value
        else:
            assert 'unknown mode'

    def build(self):
        if self.built:
            return self.datasets
        cts = modules.build_ct(self.ct_data, self.voxel_size, current_study = self.current_study)
        x,y,z = self.mgrid()
        for slicei in range(len(cts)):
            cts[slicei].ImagePositionPatient = [x[0,0,slicei],y[0,0,slicei],z[0,0,slicei]]
            cts[slicei].ImageOrientationPatient = self.ImageOrientationPatient
        self.built = True
        self.datasets = cts
        return self.datasets

class StaticBeamBuilder(object):
    def __init__(self, current_study, gantry_angle, collimator_angle=0, patient_support_angle=0, table_top=None, table_top_eccentric=None):
        if table_top == None:
            table_top = TableTop()
        if table_top_eccentric == None:
            table_top_eccentric = TableTopEcc()
        self.gantry_angle = gantry_angle
        self.collimator_angle = collimator_angle=0
        self.patient_support_angle = patient_support_angle=0
        self.table_top = table_top
        self.table_top_eccentric = table_top_eccentric
        self.current_study = current_study
        self.built = False

    def build(self, rtplan, planbuilder):
        if self.built:
            return self.beambuilder
        self.built = True
        self.beambuilder = modules.add_static_rt_beam(ds = rtplan, nleaves = planbuilder.num_leaves, leafwidths = planbuilder.leaf_widths, gantry_angle = self.gantry_angle, collimator_angle = self.collimator_angle, patient_support_angle = self.patient_support_angle, table_top = self.table_top, table_top_eccentric = self.table_top_eccentric, isocenter = planbuilder.isocenter, current_study = self.current_study)
        return self.beambuilder

class StaticPlanBuilder(object):
    def __init__(self, current_study, isocenter=None, num_leaves=None, leaf_widths=None):
        self.isocenter = isocenter or [0,0,0]
        self.num_leaves = num_leaves or [10,40,10]
        self.leaf_widths = leaf_widths or [1, 0.5, 1]
        self.beam_builders = []
        self.current_study = current_study
        self.built = False

    def build_beam(self, gantry_angle, collimator_angle=0, patient_support_angle=0, table_top=None, table_top_eccentric=None):
        sbb = StaticBeamBuilder(current_study = self.current_study, gantry_angle = gantry_angle, collimator_angle = collimator_angle, patient_support_angle = patient_support_angle, table_top = table_top, table_top_eccentric = table_top_eccentric)
        self.beam_builders.append(sbb)
        return sbb

    def build(self):
        if self.built:
            return self.datasets
        rtplan = modules.build_rt_plan(self.current_study, self.isocenter)
        for bb in self.beam_builders:
            bb.build(rtplan, self)
        self.built = True
        self.datasets = [rtplan]
        return self.datasets

class ROIBuilder(object):
    def __init__(self, structure_set_builder, name, interpreted_type, roi_number, contours=None):
        self.structure_set_builder = structure_set_builder
        if contours == None:
            self.contours = []
        else:
            self.contours = contours
        self.name = name
        self.interpreted_type = interpreted_type
        self.roi_number = roi_number
        self.built = False

    def build(self, structure_set):
        if self.built:
            return self.roi
        roi = modules.add_roi_to_structure_set(structure_set, self.name, self.structure_set_builder.current_study)
        modules.add_roi_to_roi_contour(structure_set, roi, self.contours, self.structure_set_builder.current_study)
        modules.add_roi_to_rt_roi_observation(structure_set, roi, self.name, self.interpreted_type)
        self.built = True
        self.roi = roi
        return self.roi

class StructureSetBuilder(object):
    def __init__(self, current_study, images):
        self.current_study = current_study
        self.images = images
        self.roi_builders = []
        self.built = False

    def add_external_box(self, name="External", roi_number=None):
        corner = self.images.corner
        self.add_box(size = self.images.gridsize,
                     center = self.images.corner + self.images.gridsize / 2,
                     name = name,
                     interpreted_type = "EXTERNAL",
                     roi_number = roi_number)

    def add_box(self, size, center, name, interpreted_type, roi_number = None):
        x,y,z = self.images.mgrid()
        contours = np.array([[[X*size[0]/2 + center[0],
                               Y*X*size[1]/2 + center[1],
                               Z]
                               for X in [-1,1]
                               for Y in [-1,1]]
                               for Z in z[0,0,np.abs(z[0,0,:] - center[2]) < size[2]/2]])
        if roi_number == None:
            roi_number = 1
            for rb in self.roi_builders:
                roi_number = max(roi_number, rb.roi_number + 1)

        rb = ROIBuilder(name = name, structure_set_builder = self, interpreted_type = interpreted_type,
                        roi_number = roi_number, contours = contours)
        self.roi_builders.append(rb)
        return rb

    def add_sphere(self, radius, center, name, interpreted_type):
        pass

    def build(self):
        if self.built:
            return self.datasets
        rs = modules.build_rt_structure_set(self.images.build(), self.current_study)
        print [x.SOPInstanceUID for x in self.images.build()]
        for rb in self.roi_builders:
            rb.build(rs)
        self.built = True
        self.datasets = [rs]
        return self.datasets
