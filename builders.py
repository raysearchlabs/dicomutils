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

class ImageBuilder(object):
    @property
    def gridsize(self):
        return np.array([self.num_voxels[0] * self.voxel_size[0],
                         self.num_voxels[1] * self.voxel_size[1],
                         self.num_voxels[2] * self.voxel_size[2]])

    @property
    def column_direction(self):
        return self.ImageOrientationPatient[:3]

    @property
    def row_direction(self):
        return self.ImageOrientationPatient[3:]

    def mgrid(self):
        coldir = self.column_direction
        rowdir = self.row_direction
        slicedir = self.slice_direction
        if hasattr(self, '_last_mgrid_params') and (coldir, rowdir, slicedir, self.num_voxels, self.center, self.voxel_size) == self._last_mgrid_params:
            return self._last_mgrid
        self._last_mgrid_params = (coldir, rowdir, slicedir, self.num_voxels, self.center, self.voxel_size)
        nv = np.array(self.num_voxels)/2.0
        col,row,slice=np.mgrid[-nv[0]:nv[0], -nv[1]:nv[1], -nv[2]:nv[2]]
        x = (self.center[0] + (row + 0.5) * rowdir[0] * self.voxel_size[1] +
             (col + 0.5) * coldir[0] * self.voxel_size[0] +
             (slice + 0.5) * slicedir[0] * self.voxel_size[2])
        y = (self.center[1] + (row + 0.5) * rowdir[1] * self.voxel_size[1] +
             (col + 0.5) * coldir[1] * self.voxel_size[0] +
             (slice + 0.5) * slicedir[1] * self.voxel_size[2])
        z = (self.center[2] + (row + 0.5) * rowdir[2] * self.voxel_size[1] +
             (col + 0.5) * coldir[2] * self.voxel_size[0] +
             (slice + 0.5) * slicedir[2] * self.voxel_size[2])
        self._last_mgrid = (x,y,z)
        return x,y,z

    def clear(self, real_value = None, stored_value = None):
        if real_value != None:
            assert stored_value == None
            stored_value = self.real_value_to_stored_value(real_value)
        self.pixel_array[:] = stored_value


    def add_sphere(self, radius, center, stored_value = None, real_value = None, mode = 'set'):
        if real_value != None:
            assert stored_value == None
            stored_value = self.real_value_to_stored_value(real_value)
        x,y,z = self.mgrid()
        voxels = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 <= radius**2
        if mode == 'set':
            self.pixel_array[voxels] = stored_value
        elif mode == 'add':
            self.pixel_array[voxels] += stored_value
        elif mode == 'subtract':
            self.pixel_array[voxels] -= stored_value
        else:
            assert 'unknown mode'

    def add_box(self, size, center, stored_value = None, real_value = None, mode = 'set'):
        if real_value != None:
            assert stored_value == None
            stored_value = (real_value - self.rescale_intercept) / self.rescale_slope
        x,y,z = self.mgrid()
        voxels = (abs(x-center[0]) <= size[0]/2.0) * (abs(y-center[1]) <= size[1]/2.0) * (abs(z-center[2]) <= size[2]/2.0)
        if mode == 'set':
            self.pixel_array[voxels] = stored_value
        elif mode == 'add':
            self.pixel_array[voxels] += stored_value
        elif mode == 'subtract':
            self.pixel_array[voxels] -= stored_value
        else:
            assert 'unknown mode'

class StudyBuilder(object):
    def __init__(self, patient_position="HFS", patient_id="", patients_name = "", patients_birthdate = ""):
        self.modalityorder = ["CT", "RTSTRUCT", "RTPLAN", "RTDOSE"]
        self.current_study = {}
        self.current_study['PatientID'] = patient_id
        self.current_study['PatientsName'] = patients_name
        self.current_study['PatientsBirthDate'] = patients_birthdate
        self.current_study['PatientPosition'] = patient_position
        self.seriesbuilders = defaultdict(lambda: [])
        self.built = False

    def build_ct(self, num_voxels, voxel_size, center=None, rescale_slope=1, rescale_intercept=-1024, column_direction=None, row_direction=None, slice_direction=None):
        b = CTBuilder(self.current_study, num_voxels, voxel_size, center=center, rescale_slope=rescale_slope, rescale_intercept=rescale_intercept, column_direction=column_direction, row_direction=row_direction, slice_direction=slice_direction)
        self.seriesbuilders['CT'].append(b)
        return b

    def build_static_plan(self, nominal_beam_energy=6, isocenter=None, num_leaves=None, leaf_widths=None, structure_set=None, sad=None):
        if structure_set == None and len(self.seriesbuilders['RTSTRUCT']) == 1:
            structure_set = self.seriesbuilders['RTSTRUCT'][0]
        b = StaticPlanBuilder(current_study=self.current_study,
                              nominal_beam_energy=nominal_beam_energy, isocenter=isocenter,
                              num_leaves=num_leaves, leaf_widths=leaf_widths, structure_set=structure_set, sad=sad)
        self.seriesbuilders['RTPLAN'].append(b)
        return b

    def build_structure_set(self, images=None):
        if images == None and len(self.seriesbuilders['CT']) == 1:
            images = self.seriesbuilders['CT'][0]
        b = StructureSetBuilder(self.current_study, images=images)
        self.seriesbuilders['RTSTRUCT'].append(b)
        return b

    def build_dose(self, planbuilder=None, num_voxels=None, voxel_size=None, center=None, dose_grid_scaling=1.0, column_direction=None, row_direction=None, slice_direction=None):
        if planbuilder == None and len(self.seriesbuilders['RTPLAN']) == 1:
            planbuilder = self.seriesbuilders['RTPLAN'][0]
        if (planbuilder != None
            and planbuilder.structure_set != None
            and planbuilder.structure_set.images != None):
            images = planbuilder.structure_set.images
            if num_voxels == None and voxel_size == None and center == None:
                num_voxels = images.num_voxels
                voxel_size = images.voxel_size
                center = images.center
            if column_direction == None and row_direction == None:
                column_direction, row_direction = images.column_direction, images.row_direction
            if slice_direction == None:
                slice_direction = images.slice_direction

        b = DoseBuilder(current_study=self.current_study, planbuilder=planbuilder, num_voxels=num_voxels, voxel_size=voxel_size, center=center, dose_grid_scaling=dose_grid_scaling, column_direction=column_direction, row_direction=row_direction, slice_direction=slice_direction)
        self.seriesbuilders['RTDOSE'].append(b)
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

    def write(self, outdir='.', print_filenames=False):
        for modality in self.modalityorder:
            for sb in self.seriesbuilders[modality]:
                print modality, sb
                for ds in sb.build():
                    dicom.write_file(os.path.join(outdir, ds.filename), ds)
                    if print_filenames:
                        print ds.filename

class CTBuilder(ImageBuilder):
    def __init__(self, current_study, num_voxels, voxel_size, center=None, rescale_slope=1, rescale_intercept=-1024, column_direction=None, row_direction=None, slice_direction=None):
        self.num_voxels = num_voxels
        self.voxel_size = voxel_size
        self.rescale_slope = rescale_slope
        if center == None:
            center = [0,0,0]
        self.center = np.array(center)
        self.rescale_intercept = rescale_intercept
        self.pixel_array = np.zeros(self.num_voxels, dtype=np.int16)
        if column_direction == None or row_direction == None:
            assert column_direction == None and row_direction == None
            column_direction = [1,0,0]
            row_direction = [0,1,0]
        if slice_direction == None:
            slice_direction = np.cross(column_direction, row_direction)
        slice_direction = slice_direction / np.linalg.norm(slice_direction)
        self.ImageOrientationPatient = column_direction + row_direction
        self.slice_direction = slice_direction
        self.current_study = current_study
        self.built = False

    def real_value_to_stored_value(self, real_value):
        return (real_value - self.rescale_intercept) / self.rescale_slope

    def build(self):
        if self.built:
            return self.datasets
        cts = modules.build_ct(ct_data = self.pixel_array, voxel_size = self.voxel_size, center = self.center, current_study = self.current_study)
        x,y,z = self.mgrid()
        for slicei in range(len(cts)):
            cts[slicei].ImagePositionPatient = [x[0,0,slicei],y[0,0,slicei],z[0,0,slicei]]
            cts[slicei].ImageOrientationPatient = self.ImageOrientationPatient
        self.built = True
        self.datasets = cts
        return self.datasets

class StaticBeamBuilder(object):
    def __init__(self, current_study, gantry_angle, meterset, nominal_beam_energy, collimator_angle=0, patient_support_angle=0, table_top=None, table_top_eccentric=None, sad=None):
        if table_top == None:
            table_top = TableTop()
        if table_top_eccentric == None:
            table_top_eccentric = TableTopEcc()
        self.gantry_angle = gantry_angle
        self.sad = sad
        self.collimator_angle = collimator_angle
        self.patient_support_angle = patient_support_angle
        self.table_top = table_top
        self.table_top_eccentric = table_top_eccentric
        self.meterset = meterset
        self.nominal_beam_energy = nominal_beam_energy
        self.current_study = current_study
        self.conform_calls = []
        self.jaws = None
        self.built = False

    def conform_to_circle(self, radius, center):
        self.conform_calls.append(lambda beam: modules.conform_mlc_to_circle(beam, radius, center))

    def conform_to_rectangle(self, x, y, center):
        self.conform_calls.append(lambda beam: modules.conform_mlc_to_rectangle(beam, x, y, center))

    def conform_jaws_to_rectangle(self, x, y, center):
        self.conform_calls.append(lambda beam: modules.conform_jaws_to_rectangle(beam, x, y, center))

    def conform_jaws_to_mlc(self):
        self.conform_calls.append(lambda beam: modules.conform_jaws_to_mlc(beam))

    def build(self, rtplan, planbuilder):
        if self.built:
            return self.rtbeam
        self.built = True
        self.rtbeam = modules.add_static_rt_beam(ds = rtplan, nleaves = planbuilder.num_leaves, leafwidths = planbuilder.leaf_widths, gantry_angle = self.gantry_angle, collimator_angle = self.collimator_angle, patient_support_angle = self.patient_support_angle, table_top = self.table_top, table_top_eccentric = self.table_top_eccentric, isocenter = planbuilder.isocenter, nominal_beam_energy = self.nominal_beam_energy, current_study = self.current_study, sad=self.sad)
        for call in self.conform_calls:
            call(self.rtbeam)
        if self.jaws == None:
            modules.conform_jaws_to_mlc(self.rtbeam)
        return self.rtbeam

class StaticPlanBuilder(object):
    def __init__(self, current_study, nominal_beam_energy=6, isocenter=None, num_leaves=None, leaf_widths=None, structure_set=None, sad=None):
        self.isocenter = isocenter or [0,0,0]
        self.num_leaves = num_leaves or [10,40,10]
        self.leaf_widths = leaf_widths or [10, 5, 10]
        self.beam_builders = []
        self.current_study = current_study
        self.structure_set = structure_set
        self.nominal_beam_energy = nominal_beam_energy
        self.sad = sad
        self.built = False

    def build_beam(self, gantry_angle, meterset, collimator_angle=0, patient_support_angle=0, table_top=None, table_top_eccentric=None, sad=None):
        if sad == None:
            sad = self.sad
        sbb = StaticBeamBuilder(current_study = self.current_study, meterset = meterset, nominal_beam_energy = self.nominal_beam_energy, gantry_angle = gantry_angle, collimator_angle = collimator_angle, patient_support_angle = patient_support_angle, table_top = table_top, table_top_eccentric = table_top_eccentric, sad = sad)
        self.beam_builders.append(sbb)
        return sbb

    def build(self):
        if self.built:
            return self.datasets
        rtplan = modules.build_rt_plan(self.current_study, self.isocenter, self.structure_set.build()[0])
        assert len(rtplan.FractionGroupSequence) == 1
        fraction_group = rtplan.FractionGroupSequence[0]
        for bb in self.beam_builders:
            rtbeam = bb.build(rtplan, self)
            modules.add_beam_to_rt_fraction_group(fraction_group, rtbeam, bb.meterset)
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
        modules.add_roi_to_roi_contour(structure_set, roi, self.contours, self.structure_set_builder.images.build())
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
        self.add_box(size = self.images.gridsize,
                     center = self.images.center,
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
                               for Z in z[0,0,:] if ((Z - center[2]) >= -size[2]/2 and
                                                     (Z - center[2]) < size[2]/2)])
        return self.add_contours(contours, name, interpreted_type, roi_number)

    def add_sphere(self, radius, center, name, interpreted_type, roi_number = None, ntheta = 12):
        x,y,z = self.images.mgrid()
        contours = np.array([[[np.sqrt(radius**2 - (Z-center[2])**2) * np.cos(theta) + center[0],
                               np.sqrt(radius**2 - (Z-center[2])**2) * np.sin(theta) + center[1],
                               Z]
                              for theta in np.linspace(0, 2*np.pi, ntheta, endpoint=False)]
                             for Z in z[0,0,np.abs(z[0,0,:] - center[2]) < radius]])
        return self.add_contours(contours, name, interpreted_type, roi_number)


    def add_contours(self, contours, name, interpreted_type, roi_number = None):
        if roi_number == None:
            roi_number = 1
            for rb in self.roi_builders:
                roi_number = max(roi_number, rb.roi_number + 1)

        rb = ROIBuilder(name = name, structure_set_builder = self, interpreted_type = interpreted_type,
                        roi_number = roi_number, contours = contours)
        self.roi_builders.append(rb)
        return rb


    def build(self):
        if self.built:
            return self.datasets
        rs = modules.build_rt_structure_set(self.images.build(), self.current_study)
        for rb in self.roi_builders:
            rb.build(rs)
        self.built = True
        self.datasets = [rs]
        return self.datasets

from decimal import Decimal

class DoseBuilder(ImageBuilder):
    def __init__(self, current_study, planbuilder, num_voxels, voxel_size, center=None, dose_grid_scaling=1.0, column_direction=None, row_direction=None, slice_direction=None):
        self.current_study = current_study
        self.planbuilder = planbuilder
        self.num_voxels = num_voxels
        self.voxel_size = voxel_size
        self.pixel_array = np.zeros(self.num_voxels, dtype=np.int16)
        if center == None:
            center = [0,0,0]
        self.center = np.array(center)
        if column_direction == None or row_direction == None:
            assert column_direction == None and row_direction == None
            column_direction = [1,0,0]
            row_direction = [0,1,0]
        if slice_direction == None:
            slice_direction = np.cross(column_direction, row_direction)
        slice_direction = slice_direction / np.linalg.norm(slice_direction)
        self.ImageOrientationPatient = column_direction + row_direction
        self.slice_direction = slice_direction
        self.dose_grid_scaling = dose_grid_scaling
        self.built = False

    def real_value_to_stored_value(self, real_value):
        return real_value / self.dose_grid_scaling

    def add_lightfield(self, beam, weight):
        x,y,z = self.mgrid()
        bld = modules.getblds(beam.BeamLimitingDeviceSequence)
        gantry_angle = None
        gantry_pitch_angle = 0
        isocenter = [0,0,0]
        beam_limiting_device_angle = Decimal(0)
        table_top = TableTop()
        table_top_ecc = TableTopEcc()
        patient_position = self.current_study['PatientPosition']
        patient_support_angle = 0
        if hasattr(beam, 'SourceAxisDistance'):
            sad = beam.SourceAxisDistance
        else:
            sad = 1000
        for cp in beam.ControlPointSequence:
            if hasattr(cp, 'BeamLimitingDevicePositionSequence') and cp.BeamLimitingDevicePositionSequence != None:
                bldp = modules.getblds(cp.BeamLimitingDevicePositionSequence)
            gantry_angle = getattr(cp, 'GantryAngle', gantry_angle)
            gantry_pitch_angle = getattr(cp, 'GantryPitchAngle', gantry_pitch_angle)
            beam_limiting_device_angle = getattr(cp, 'BeamLimitingDeviceAngle', beam_limiting_device_angle)
            patient_support_angle = getattr(cp, 'PatientSupportAngle', patient_support_angle)
            isocenter = getattr(cp, 'IsocenterPosition', isocenter)
            table_top_ecc.Ls = getattr(cp, 'TableTopEccentricAxisDistance', table_top_ecc.Ls)
            table_top_ecc.theta_e = getattr(cp, 'TableTopEccentricAngle', table_top_ecc.theta_e)
            table_top.psi_t = getattr(cp, 'TableTopPitchAngle', table_top.psi_t)
            table_top.phi_t = getattr(cp, 'TableTopRollAngle', table_top.phi_t)
            table_top.Tx = getattr(cp, 'TableTopLateralPosition', table_top.Tx)
            table_top.Ty = getattr(cp, 'TableTopLongitudinalPosition', table_top.Ty)
            table_top.Tz = getattr(cp, 'TableTopVerticalPosition', table_top.Tz)
            patient_position = getattr(cp, 'PatientPosition', patient_position)

            table_top = TableTop()
            Mdb = modules.get_dicom_to_bld_coordinate_transform(gantry_angle, gantry_pitch_angle, beam_limiting_device_angle,
                                                                patient_support_angle, patient_position,
                                                                table_top, table_top_ecc, sad, isocenter)
            coords = np.array([x.ravel(),
                               y.ravel(),
                               z.ravel(),
                               np.ones(x.shape).ravel()]).reshape((4,1,1,np.prod(x.shape)))
            c = Mdb * coords
            # Negation here since everything is at z < 0 in the b system, and that rotates by 180 degrees
            c2 = -np.array([float(beam.SourceAxisDistance)*c[0,:]/c[2,:], float(beam.SourceAxisDistance)*c[1,:]/c[2,:]]).squeeze()
            nleaves = len(bld['MLCX'].LeafPositionBoundaries)-1
            for i in range(nleaves):
                self.pixel_array.ravel()[(c2[0,:] >= max(float(bldp['ASYMX'].LeafJawPositions[0]),
                                              float(bldp['MLCX'].LeafJawPositions[i])))
                               * (c2[0,:] <= min(float(bldp['ASYMX'].LeafJawPositions[1]),
                                                 float(bldp['MLCX'].LeafJawPositions[i + nleaves])))
                               * (c2[1,:] > max(float(bldp['ASYMY'].LeafJawPositions[0]),
                                                 float(bld['MLCX'].LeafPositionBoundaries[i])))
                               * (c2[1,:] <= min(float(bldp['ASYMY'].LeafJawPositions[1]),
                                                 float(bld['MLCX'].LeafPositionBoundaries[i+1])))] += 1

    def build(self):
        if self.built:
            return self.datasets
        rd = modules.build_rt_dose(self.pixel_array, self.voxel_size, self.center, self.current_study,
                                   self.planbuilder.build()[0], self.dose_grid_scaling)
        x,y,z = self.mgrid()
        rd.ImagePositionPatient = [x[0,0,0],y[0,0,0],z[0,0,0]]
        rd.ImageOrientationPatient = self.ImageOrientationPatient
                                   
        self.built = True
        self.datasets = [rd]
        return self.datasets
