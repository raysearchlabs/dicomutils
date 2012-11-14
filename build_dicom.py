#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import dicom, time, uuid, sys, datetime, os
import coordinates
# Be careful to pass good fp numbers...
if hasattr(dicom, 'config'):
    dicom.config.allow_DS_float = True


def nmin(it):
    n = None
    for i in it:
        if n == None or i < n:
            n = i
    return n

def nmax(it):
    n = None
    for i in it:
        if n == None or i > n:
            n = i
    return n

from collections import defaultdict
def getblds(blds):
    d = defaultdict(lambda: None)
    for bld in blds:
        if hasattr(bld, 'RTBeamLimitingDeviceType'):
            d[bld.RTBeamLimitingDeviceType] = bld
    return d

from decimal import Decimal

def conform_jaws_to_mlc(beam):
    bld = getblds(beam.BeamLimitingDeviceSequence)
    nleaves = len(bld['MLCX'].LeafPositionBoundaries)-1
    for cp in beam.ControlPointSequence:
        opentolerance = Decimal("0.5") # mm
        if hasattr(cp, 'BeamLimitingDevicePositionSequence') and cp.BeamLimitingDevicePositionSequence != None:
            bldp = getblds(cp.BeamLimitingDevicePositionSequence)

            if bldp['MLCX'] != None and bldp['ASYMY'] != None:
                min_open_leafi = nmin(i for i in range(nleaves)
                                      if bldp['MLCX'].LeafJawPositions[i] <= bldp['MLCX'].LeafJawPositions[i+nleaves] - opentolerance)
                max_open_leafi = nmax(i for i in range(nleaves)
                                      if bldp['MLCX'].LeafJawPositions[i] <= bldp['MLCX'].LeafJawPositions[i+nleaves] - opentolerance)
                if min_open_leafi != None and max_open_leafi != None:
                    bldp['ASYMY'].LeafJawPositions = [bld['MLCX'].LeafPositionBoundaries[min_open_leafi],
                                                      bld['MLCX'].LeafPositionBoundaries[max_open_leafi + 1]]
            if bldp['MLCX'] != None and bldp['ASYMX'] != None:
                min_open_leaf = min(bldp['MLCX'].LeafJawPositions[i] for i in range(nleaves)
                                    if bldp['MLCX'].LeafJawPositions[i] <= bldp['MLCX'].LeafJawPositions[i+nleaves] - opentolerance)
                max_open_leaf = max(bldp['MLCX'].LeafJawPositions[i+nleaves] for i in range(nleaves)
                                    if bldp['MLCX'].LeafJawPositions[i] <= bldp['MLCX'].LeafJawPositions[i+nleaves] - opentolerance)
                bldp['ASYMX'].LeafJawPositions = [min_open_leaf, max_open_leaf]

def conform_mlc_to_circle(beam, radius, center):
    bld = getblds(beam.BeamLimitingDeviceSequence)
    nleaves = len(bld['MLCX'].LeafPositionBoundaries)-1
    for cp in beam.ControlPointSequence:
        if hasattr(cp, 'BeamLimitingDevicePositionSequence') and cp.BeamLimitingDevicePositionSequence != None:
            bldp = getblds(cp.BeamLimitingDevicePositionSequence)
            for i in range(nleaves):
                y = float((bld['MLCX'].LeafPositionBoundaries[i] + bld['MLCX'].LeafPositionBoundaries[i+1]) / 2)
                if abs(y) < radius:
                    bldp['MLCX'].LeafJawPositions[i] = Decimal(-np.sqrt(radius**2 - (y-center[1])**2) + center[0]).quantize(Decimal("0.01"))
                    bldp['MLCX'].LeafJawPositions[i + nleaves] = Decimal(np.sqrt(radius**2 - (y-center[1])**2) + center[0]).quantize(Decimal("0.01"))

def conform_mlc_to_rectangular_field(beam, x, y, center):
    """Sets MLC to open at least x * y cm"""
    bld = getblds(beam.BeamLimitingDeviceSequence)
    nleaves = len(bld['MLCX'].LeafPositionBoundaries)-1
    for cp in beam.ControlPointSequence:
        if hasattr(cp, 'BeamLimitingDevicePositionSequence') and cp.BeamLimitingDevicePositionSequence != None:
            bldp = getblds(cp.BeamLimitingDevicePositionSequence)
            for i in range(nleaves):
                if bld['MLCX'].LeafPositionBoundaries[i+1] > (center[1]-y/2.0) and bld['MLCX'].LeafPositionBoundaries[i] < (center[1]+y/2.0):
                    bldp['MLCX'].LeafJawPositions[i] = Decimal(center[0] - x/2.0)
                    bldp['MLCX'].LeafJawPositions[i + nleaves] = Decimal(center[0] + x/2.0)

def conform_jaws_to_rectangular_field(beam, x, y, center):
    """Sets jaws opening to x * y cm, centered at `center`"""
    bld = getblds(beam.BeamLimitingDeviceSequence)
    nleaves = len(bld['MLCX'].LeafPositionBoundaries)-1
    for cp in beam.ControlPointSequence:
        if hasattr(cp, 'BeamLimitingDevicePositionSequence') and cp.BeamLimitingDevicePositionSequence != None:
            bldp = getblds(cp.BeamLimitingDevicePositionSequence)
            bldp['ASYMX'].LeafJawPositions = [Decimal(center[0] - x/2.0),
                                              Decimal(center[0] + x/2.0)]
            bldp['ASYMY'].LeafJawPositions = [Decimal(center[1] - y/2.0),
                                              Decimal(center[1] + y/2.0)]

#def conform_mlc_to_roi(beam, roi, current_study):
#    for contour in roi.ContourSequence:
#        for i in range(0, len(contour.ContourData) - 3, 3):
#            pass

def open_mlc_for_line_segment(lpb, lp, v1, v2):
    if v1[1] > v2[1]:
        v1,v2 = v2,v1
    # line segment outside in y?
    if v2[1] < lpb[0] or v1[1] > lpb[-1]:
        return
    nleaves = len(lpb)-1
    for i in range(0,nleaves):
        if lpb[i+1] < v1[1]:
            continue
        if lpb[i] > v2[1]:
            break
        if v1[1] < lpb[i]:
            xstart = v1[0] + (v2[0]-v1[0]) * (lpb[i]-v1[1])/(v2[1]-v1[1])
        else:
            xstart = v1[0]
        if v2[1] > lpb[i+1]:
            xend = v2[0] - (v1[0]-v2[0]) * (lpb[i+1]-v2[1])/(v1[1]-v2[1])
        else:
            xend = v2[0]
        lp[i] = min(lp[i], xstart, xend)
        lp[i+nleaves] = max(lp[i+nleaves], xstart, xend)

def add_roi_to_structure_set(ds, ROIName, current_study):
    newroi = dicom.dataset.Dataset()
    roinumber = max([0] + [roi.ROINumber for roi in ds.StructureSetROISequence]) + 1
    newroi.ROIName = ROIName
    newroi.ReferencedFrameofReferenceUID = get_current_study_uid('FrameofReferenceUID', current_study)
    newroi.ROINumber = roinumber
    newroi.ROIGenerationAlgorithm = "SEMIAUTOMATIC"
    ds.StructureSetROISequence.append(newroi)
    return newroi

def get_roi_contour_module(ds):
    ds.ROIContourSequence = []
    return ds

roicolors = [[255,0,0],
             [0,255,0],
             [0,0,255],
             [255,255,0],
             [0,255,255],
             [255,0,255],
             [255,127,0],
             [127,255,0],
             [0,255,127],
             [0,127,255],
             [127,0,255],
             [255,0,127],
             [255,127,127],
             [127,255,127],
             [127,127,255],
             [255,255,127],
             [255,127,255],
             [127,255,255]]

def add_roi_to_roi_contour(ds, roi, contours, current_study):
    newroi = dicom.dataset.Dataset()
    ds.ROIContourSequence.append(newroi)
    newroi.ReferencedROINumber = roi.ROINumber
    newroi.ROIDisplayColor = roicolors[(roi.ROINumber-1) % len(roicolors)]
    newroi.ContourSequence = []
    for i, contour in enumerate(contours, 1):
        c = dicom.dataset.Dataset()
        newroi.ContourSequence.append(c)
        c.ContourNumber = i
        c.ContourGeometricType = 'CLOSED_PLANAR'
        # c.AttachedContours = [] # T3
        if 'CT' in current_study:
            c.ContourImageSequence = [] # T3
            for image in current_study['CT']:
                if image.ImagePositionPatient[2] == contour[0,2]:
                    imgref = dicom.dataset.Dataset()
                    imgref.ReferencedSOPInstanceUID = image.SOPInstanceUID
                    imgref.ReferencedSOPClassUID = image.SOPClassUID
                    # imgref.ReferencedFrameNumber = "" # T1C on multiframe
                    # imgref.ReferencedSegmentNumber = "" # T1C on segmentation
                    c.ContourImageSequence.append(imgref)
        # c.ContourSlabThickness = "" # T3
        # c.ContourOffsetVector = [0,0,0] # T3
        c.NumberofContourPoints = len(contour)
        c.ContourData = "\\".join(["%g" % x for x in contour.ravel().tolist()])
    return newroi

def get_rt_roi_observations_module(ds):
    ds.RTROIObservationsSequence = []
    return ds

def add_roi_to_rt_roi_observation(ds, roi, label, interpreted_type):
    roiobs = dicom.dataset.Dataset()
    ds.RTROIObservationsSequence.append(roiobs)
    roiobs.ObservationNumber = roi.ROINumber
    roiobs.ReferencedROINumber = roi.ROINumber
    roiobs.ROIObservationLabel = label # T3
    # roiobs.ROIObservationDescription = "" # T3
    # roiobs.RTRelatedROISequence = [] # T3
    # roiobs.RelatedRTROIObservationsSequence = [] # T3
    roiobs.RTROIInterpretedType = interpreted_type # T3
    roiobs.ROIInterpreter = "" # T2
    # roiobs.MaterialID = "" # T3
    # roiobs.ROIPhysicalPropertiesSequence = [] # T3
    return roiobs

def get_centered_coordinates(voxelGrid, nVoxels):
    x,y,z=np.mgrid[:nVoxels[0],:nVoxels[1],:nVoxels[2]]
    x=(x-(nVoxels[0]-1)/2.0)*voxelGrid[0]
    y=(y-(nVoxels[1]-1)/2.0)*voxelGrid[1]
    z=(z-(nVoxels[2]-1)/2.0)*voxelGrid[2]
    return x,y,z

def get_dicom_to_bld_coordinate_transform(gantryAngle, gantryPitchAngle, beamLimitingDeviceAngle, patientSupportAngle, patientPosition, table_top, table_top_ecc, SAD, isocenter_d):
    if patientPosition == 'HFS':
        psi_p, phi_p, theta_p = 0,0,0
    elif patientPosition == 'HFP':
        psi_p, phi_p, theta_p = 0,180,0
    elif patientPosition == 'FFS':
        psi_p, phi_p, theta_p = 0,0,180
    elif patientPosition == 'FFP':
        psi_p, phi_p, theta_p = 180,0,0
    elif patientPosition == 'HFDL':
        psi_p, phi_p, theta_p = 0,90,0
    elif patientPosition == 'HFDR':
        psi_p, phi_p, theta_p = 0,270,0
    elif patientPosition == 'FFDL':
        psi_p, phi_p, theta_p = 180,270,0
    elif patientPosition == 'FFDR':
        psi_p, phi_p, theta_p = 180,90,0
    else:
        assert False, "Unknown patient position %s!" % (patientPosition,)

    # Find the isocenter in patient coordinate system, had the patient system not been translated
    isocenter_p0 = (coordinates.Mfs(patientSupportAngle)
                    * coordinates.Mse(table_top_ecc.Ls, table_top_ecc.theta_e)
                    * coordinates.Met(table_top.Tx, table_top.Ty, table_top.Tz, table_top.psi_t, table_top.phi_t)
                    * coordinates.Mtp(0, 0, 0, psi_p, phi_p, theta_p)) * [[0],[0],[0],[1]]
    # Find the coordinates in the patient system of the desired isocenter
    isocenter_p1 = np.linalg.inv(coordinates.Mpd()) * np.array([float(isocenter_d[0]), float(isocenter_d[1]), float(isocenter_d[2]), 1.0]).reshape((4,1))
    # Compute the patient coordinate system translation
    Px,Py,Pz,_ = isocenter_p0 - isocenter_p1

    M = (coordinates.Mgb(SAD, beamLimitingDeviceAngle)
         * coordinates.Mfg(gantryPitchAngle, gantryAngle)
         * np.linalg.inv(coordinates.Mfs(patientSupportAngle))
         * np.linalg.inv(coordinates.Mse(table_top_ecc.Ls, table_top_ecc.theta_e))
         * np.linalg.inv(coordinates.Met(table_top.Tx, table_top.Ty, table_top.Tz, table_top.psi_t, table_top.phi_t))
         * np.linalg.inv(coordinates.Mtp(Px, Py, Pz, psi_p, phi_p, theta_p))
         * np.linalg.inv(coordinates.Mpd()))
    return M

def add_lightfield(ctData, beam, x, y, z):
    bld = getblds(beam.BeamLimitingDeviceSequence)
    gantry_angle = None
    gantry_pitch_angle = 0
    isocenter = [0,0,0]
    beam_limiting_device_angle = Decimal(0)
    table_top = TableTop()
    table_top_ecc = TableTopEcc()
    patient_position = 'HFS'
    patient_support_angle = 0
    if hasattr(beam, 'SourceAxisDistance'):
        sad = beam.SourceAxisDistance
    else:
        sad = 1000
    for cp in beam.ControlPointSequence:
        if hasattr(cp, 'BeamLimitingDevicePositionSequence') and cp.BeamLimitingDevicePositionSequence != None:
            bldp = getblds(cp.BeamLimitingDevicePositionSequence)
        gantry_angle = getattr(cp, 'GantryAngle', gantry_angle)
        gantry_pitch_angle = getattr(cp, 'GantryPitchAngle', gantry_angle)
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
        Mdb = get_dicom_to_bld_coordinate_transform(gantry_angle, gantry_pitch_angle, beam_limiting_device_angle,
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
            ctData.ravel()[(c2[0,:] >= max(float(bldp['ASYMX'].LeafJawPositions[0]),
                                          float(bldp['MLCX'].LeafJawPositions[i])))
                           * (c2[0,:] <= min(float(bldp['ASYMX'].LeafJawPositions[1]),
                                             float(bldp['MLCX'].LeafJawPositions[i + nleaves])))
                           * (c2[1,:] > max(float(bldp['ASYMY'].LeafJawPositions[0]),
                                             float(bld['MLCX'].LeafPositionBoundaries[i])))
                           * (c2[1,:] <= min(float(bldp['ASYMY'].LeafJawPositions[1]),
                                             float(bld['MLCX'].LeafPositionBoundaries[i+1])))] += 1

if __name__ == '__main__':
    import argparse
    class ModalityGroupAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            ns = namespace.__dict__.copy()
            ns.pop('studies')
            ns['modality'] = values
            namespace.studies[-1].append(argparse.Namespace(**ns))
    class NewStudyAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.studies.append([])

    parser = argparse.ArgumentParser(description='Create DICOM data.')
    parser.add_argument('--patient-position', dest='patient_position', choices = ['HFS', 'HFP', 'FFS', 'FFP', 'HFDR', 'HFDL', 'FFDR', 'FFDP'],
                        help='The patient position written in the images. Required for CT and MR. (default: not specified)')
    parser.add_argument('--patient-id', dest='patient_id', default='Patient ID',
                        help='The patient ID.')
    parser.add_argument('--patients-name', dest='patients_name', default='LastName^GivenName^^^',
                        help="The patient's name, in DICOM caret notation.")
    parser.add_argument('--patients-birthdate', dest='patients_birthdate', default='',
                        help="The patient's birthdate, in DICOM DA notation (YYYYMMDD).")
    parser.add_argument('--voxelsize', dest='VoxelSize', default="1,2,4",
                        help='The size of a single voxel in mm. (default: 1,2,4)')
    parser.add_argument('--voxels', dest='Voxels', default="64,32,16",
                        help='The number of voxels in the dataset. (default: 64,32,16)')
    parser.add_argument('--modality', dest='modality', default=[], choices = ['CT', "RTDOSE", "RTPLAN", "RTSTRUCT"],
                        help='The modality to write. (default: CT)', action=ModalityGroupAction)
    parser.add_argument('--nominal-energy', dest='nominal_energy', default=None,
                        help='The nominal energy of beams in an RT Plan.')
    parser.add_argument('--values', dest='values', default=[], action='append', metavar='VALUE | SHAPE{,PARAMETERS}',
                        help="""Set the Hounsfield or dose values in a volume to the given value.
    \n\n\n
                        For syntax, see the forthcoming documentation or the source code...""")
    parser.add_argument('--sad', dest='sad', default=1000, help="The Source to Axis distance.")
    parser.add_argument('--structure', dest='structures', default=[], action='append', metavar='SHAPE{,PARAMETERS}',
                        help="""Add a structure to the current list of structure sets.
                        For syntax, see the forthcoming documentation or the source code...""")
    parser.add_argument('--beams', dest='beams', default='3',
                        help="""Set the number of equidistant beams to write in an RTPLAN.""")
    parser.add_argument('--collimator-angles', dest='collimator_angles', default='0',
                        help="""Set the collimator angle (Beam Limiting Device Angle) of the beams.
                        In IEC61217 terminology, that corresponds to the theta_b angle.""")
    parser.add_argument('--patient-support-angles', dest='patient_support_angles', default='0',
                        help="""Set the Patient Support Angle ("couch angle") of the beams.
                        In IEC61217 terminology, that corresponds to the theta_s angle.""")
    parser.add_argument('--table-top', dest='table_top', default='0,0,0,0,0',
                        help="""Set the table top pitch, roll and lateral, longitudinal and vertical positions.
                        In IEC61217 terminology, that corresponds to the
                        psi_t, phi_t, Tx, Ty, Tz coordinates, respectively.""")
    parser.add_argument('--table-top-eccentric', dest='table_top_eccentric', default='0,0',
                        help="""Set the table top eccentric axis distance and angle.
                        In IEC61217 terminology, that corresponds to the Ls and theta_e coordinates, respectively.""")
    parser.add_argument('--isocenter', dest='isocenter', default='[0;0;0]',
                        help="""Set the isocenter of the beams.""")
    parser.add_argument('--mlc-shape', dest='mlcshapes', default=[], action='append',
                        help="""Add an opening to the current list of mlc openings.
                        For syntax, see the forthcoming documentation or the source code...""")
    parser.add_argument('--jaw-shape', dest='jawshapes', default=[], action='append',
                        help="""Sets the jaw shape to x * y, centered at (xc, yc). Given as [x;y;xc;yc]. Defaults to conforming to the MLC.""")
    parser.add_argument('--outdir', dest='outdir', default='.',
                        help="""Generate data to this directory. (default: working directory)""")

    args = parser.parse_args(namespace = argparse.Namespace(studies=[[]]))

    voxelGrid = [float(x) for x in args.VoxelSize.split(",")]
    nVoxels = [int(x) for x in args.Voxels.split(",")]
    x,y,z = get_centered_coordinates(voxelGrid, nVoxels)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    def build_sphere_contours(z, name, radius, center, interpreted_type, ntheta = 32):
        #print "build_sphere_contours", z, name, radius, center, interpreted_type, ntheta
        contours = np.array([[[np.sqrt(radius**2 - (Z-center[2])**2) * np.cos(theta) + center[0],
                               np.sqrt(radius**2 - (Z-center[2])**2) * np.sin(theta) + center[1],
                               Z]
                              for theta in np.linspace(0, 2*np.pi, ntheta, endpoint=False)]
                             for Z in z[np.abs(z - center[2]) < radius]])
        return {'Name': name,
                'InterpretedType': interpreted_type,
                'Contours': contours}

    for study in args.studies:
        current_study = {}
        for series in study:
            for value in series.values:
                value = value.split(",")
                if len(value) == 1 and (value[0][0].isdigit() or value[0][0] == '-'):
                    ctData[:] = float(value[0])
                else:
                    shape = value[0]
                    if shape == "sphere":
                        val = float(value[1])
                        radius = float(value[2])
                        if len(value) > 3:
                            center = [float(c) for c in value[3].lstrip('[').rstrip(']').split(";")]
                        else:
                            center = [0,0,0]
                        ctData[(x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 <= radius**2] = val
                    elif shape == "box":
                        val = float(value[1])
                        size = value[2]
                        if size.startswith("[") and size.endswith("]"):
                            size = [float(c) for c in size.lstrip('[').rstrip(']').split(";")]
                        else:
                            size = [float(size),float(size),float(size)]
                        if len(value) > 3:
                            center = [float(c) for c in value[3].lstrip('[').rstrip(']').split(";")]
                        else:
                            center = [0,0,0]
                        ctData[(abs(x-center[0]) <= size[0]/2.0) * (abs(y-center[1]) <= size[1]/2.0) * (abs(z-center[2]) <= size[2]/2.0)] = val
                    elif shape == "lightfield":
                        for beam in current_study['RTPLAN'].BeamSequence:
                            add_lightfield(ctData, beam, x, y, z)

            if series.patient_position != None:
                current_study['PatientPosition'] = series.patient_position
            if series.patient_id != None:
                current_study['PatientID'] = series.patient_id
            if series.patients_name != None:
                current_study['PatientsName'] = series.patients_name
            if series.patients_birthdate != None:
                current_study['PatientsBirthDate'] = series.patients_birthdate
            if series.nominal_energy != None:
                current_study['NominalEnergy'] = series.nominal_energy
            if series.modality == "CT":
                if 'PatientPosition' not in current_study:
                    parser.error("Patient position must be specified when writing CT images!")
                datasets = build_ct(ctData, voxelGrid, current_study = current_study)
                current_study['CT'] = datasets
                for ds in datasets:
                    dicom.write_file(os.path.join(args.outdir, ds.filename), ds)
            elif series.modality == "RTDOSE":
                rd = build_rt_dose(ctData, voxelGrid, current_study = current_study)
                current_study['RTDOSE'] = rd
                dicom.write_file(os.path.join(args.outdir, rd.filename), rd)
            elif series.modality == "RTPLAN":
                if all(d.isdigit() for d in series.beams):
                    beams = int(series.beams)
                else:
                    beams = [int(b) for b in series.beams.lstrip('[').rstrip(']').split(";")]
                if all(d.isdigit() for d in series.collimator_angles):
                    collimator_angles = int(series.collimator_angles)
                else:
                    collimator_angles = [int(b) for b in series.collimator_angles.lstrip('[').rstrip(']').split(";")]
                if all(d.isdigit() for d in series.patient_support_angles):
                    patient_support_angles = int(series.patient_support_angles)
                else:
                    patient_support_angles = [int(b) for b in series.patient_support_angles.lstrip('[').rstrip(']').split(";")]
                table_top = TableTop(*[float(b) for b in series.table_top.split(",")])
                table_top_eccentric = TableTopEcc(*[float(b) for b in series.table_top_eccentric.split(",")])
                isocenter = [float(b) for b in series.isocenter.lstrip('[').rstrip(']').split(";")]
                rp = build_rt_plan(current_study = current_study, numbeams = beams, collimator_angles = collimator_angles,
                                   patient_support_angles = patient_support_angles,
                                   table_top = table_top, table_top_eccentric = table_top_eccentric,
                                   sad = series.sad, isocenter = isocenter)
                for mlcshape in series.mlcshapes:
                    mlcshape = mlcshape.split(",")
                    if all(d.isdigit() for d in mlcshape[0]):
                        beams = [rp.BeamSequence[int(mlcshape[0])-1]]
                        mlcshape=mlcshape[1:]
                    else:
                        beams = rp.BeamSequence

                    if mlcshape[0] == "circle":
                        radius = float(mlcshape[1])
                        if len(mlcshape) > 2:
                            center = [float(c) for c in mlcshape[2].lstrip('[').rstrip(']').split(";")]
                        else:
                            center = [0,0]
                        for beam in beams:
                            conform_mlc_to_circle(beam, radius, center)
                    elif mlcshape[0] == "rectangle":
                        X,Y = float(mlcshape[1]),float(mlcshape[2])
                        if len(mlcshape) > 3:
                            center = [float(c) for c in mlcshape[3].lstrip('[').rstrip(']').split(";")]
                        else:
                            center = [0,0]
                        for beam in beams:
                            conform_mlc_to_rectangular_field(beam, X, Y, center)
                for beam in beams:
                    conform_jaws_to_mlc(beam)

                for jawshape in series.jawshapes:
                    jawshape = jawshape.split(",")
                    if len(jawshape) == 2:
                        beams = [rp.BeamSequence[int(jawshape[0])-1]]
                        jawshape=jawshape[1:]
                    else:
                        beams = rp.BeamSequence
                    jawsize = [float(c) for c in jawshape[0].lstrip('[').rstrip(']').split(";")]
                    if len(jawsize) > 2:
                        center = [jawsize[2],jawsize[3]]
                    else:
                        center = [0,0]
                    for beam in beams:
                        conform_jaws_to_rectangular_field(beam, jawsize[0], jawsize[1], center)
                current_study['RTPLAN'] = rp
                dicom.write_file(os.path.join(args.outdir, rp.filename), rp)
            elif series.modality == "RTSTRUCT":
                structures = []
                for structure in series.structures:
                    structure = structure.split(",")
                    shape = structure[0]
                    if shape == 'sphere':
                        name = structure[1]
                        radius = float(structure[2])
                        interpreted_type = structure[3]
                        if len(structure) > 4:
                            center = [float(c) for c in structure[4].lstrip('[').rstrip(']').split(";")]
                        else:
                            center = [0,0,0]
                        structures.append(build_sphere_contours(z[0,0,:], name, radius, center, interpreted_type))
                    elif shape == 'box':
                        name = structure[1]
                        size = structure[2]
                        if size.startswith("[") and size.endswith("]"):
                            size = [float(c) for c in size.lstrip('[').rstrip(']').split(";")]
                        else:
                            size = [float(size),float(size),float(size)]
                        interpreted_type = structure[3]
                        if len(structure) > 4:
                            center = [float(c) for c in structure[4].lstrip('[').rstrip(']').split(";")]
                        else:
                            center = [0,0,0]
                        structures.append(build_box_contours(z[0,0,:], name, size, center, interpreted_type))
                    elif shape == "external":
                        structures.append(build_box_contours(z[0,0,:], 'External', [voxelGrid[0]*nVoxels[0],
                                                                                    voxelGrid[1]*nVoxels[1],
                                                                                    voxelGrid[2]*nVoxels[2]],
                                                             [0,0,0], 'EXTERNAL'))

                rs = build_rt_structure_set(structures, current_study = current_study)
                current_study['RTSTRUCT'] = rs
                dicom.write_file(os.path.join(args.outdir, rs.filename), rs)
