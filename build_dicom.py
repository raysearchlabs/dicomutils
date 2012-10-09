#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import dicom, time, uuid, sys, datetime

# Be careful to pass good fp numbers...
if hasattr(dicom, 'config'):
    dicom.config.allow_DS_float = True

def get_uid(name):
    return [k for k,v in dicom.UID.UID_dictionary.iteritems() if v[0] == name][0]

def generate_uid(_uuid = None):
    """Returns a new DICOM UID based on a UUID, as specified in CP1156 (Final)."""
    if _uuid == None:
        _uuid = uuid.uuid1()
    return "2.25.%i" % _uuid.int

ImplementationClassUID = '2.25.229451600072090404564544894284998027172'

def get_empty_dataset(filename, storagesopclass):
    file_meta = dicom.dataset.Dataset()
    file_meta.MediaStorageSOPClassUID = get_uid(storagesopclass)
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = ImplementationClassUID
    ds = dicom.dataset.FileDataset(filename, {}, file_meta=file_meta, preamble="\0"*128)
    return ds

def get_default_ct_dataset(filename):
    DT = "%04i%02i%02i" % datetime.datetime.now().timetuple()[:3]
    TM = "%02i%02i%02i" % datetime.datetime.now().timetuple()[3:6]
    ds = get_empty_dataset(filename, "CT Image Storage")
    get_sop_common_module(ds, DT, TM, "CT Image Storage")
    get_ct_image_module(ds)
    get_image_pixel_macro(ds)
    get_patient_module(ds)
    get_general_study_module(ds, DT, TM)
    get_general_series_module(ds, DT, TM, "CT")
    get_frame_of_reference_module(ds)
    get_general_equipment_module(ds)
    get_general_image_module(ds, DT, TM)
    get_image_plane_module(ds)
    return ds

def get_default_rt_dose_dataset(filename, current_study):
    DT = "%04i%02i%02i" % datetime.datetime.now().timetuple()[:3]
    TM = "%02i%02i%02i" % datetime.datetime.now().timetuple()[3:6]
    ds = get_empty_dataset(filename, "RT Dose Storage")
    get_sop_common_module(ds, DT, TM, "RT Dose Storage")
    get_patient_module(ds)
    get_image_pixel_macro(ds)
    get_general_study_module(ds, DT, TM)
    get_rt_series_module(ds, DT, TM, "RTDOSE")
    get_frame_of_reference_module(ds)
    get_general_equipment_module(ds)
    get_general_image_module(ds, DT, TM)
    get_image_plane_module(ds)
    get_multi_frame_module(ds)
    get_rt_dose_module(ds, current_study)
    return ds

def get_default_rt_structure_set_dataset(filename, current_study):
    DT = "%04i%02i%02i" % datetime.datetime.now().timetuple()[:3]
    TM = "%02i%02i%02i" % datetime.datetime.now().timetuple()[3:6]
    ds = get_empty_dataset(filename, "RT Structure Set Storage")
    get_sop_common_module(ds, DT, TM, "RT Structure Set Storage")
    get_patient_module(ds)
    get_general_study_module(ds, DT, TM)
    get_rt_series_module(ds, DT, TM, "RTSTRUCT")
    get_general_equipment_module(ds)
    get_structure_set_module(ds, DT, TM, current_study)
    get_roi_contour_module(ds)
    get_rt_roi_observations_module(ds)
    return ds

def get_default_rt_plan_dataset(filename, current_study):
    DT = "%04i%02i%02i" % datetime.datetime.now().timetuple()[:3]
    TM = "%02i%02i%02i" % datetime.datetime.now().timetuple()[3:6]
    ds = get_empty_dataset(filename, "RT Plan Storage")
    get_sop_common_module(ds, DT, TM, "RT Plan Storage")
    get_patient_module(ds)
    get_general_study_module(ds, DT, TM)
    get_rt_series_module(ds, DT, TM, "RTPLAN")
    get_frame_of_reference_module(ds)
    get_general_equipment_module(ds)
    get_rt_general_plan_module(ds, DT, TM, current_study)
    #get_rt_prescription_module(ds)
    #get_rt_tolerance_tables(ds)
    if 'PatientPosition' in current_study:
        get_rt_patient_setup_module(ds, current_study)
    get_rt_beams_module(ds, 3, [10,40,10], [10,5,10], current_study)
    get_rt_fraction_scheme_module(ds, 30)
    #get_approval_module(ds)
    return ds

def get_sop_common_module(ds, DT, TM, modality):
    # Type 1
    ds.SOPClassUID = get_uid(modality)
    ds.SOPInstanceUID = ""
    # Type 3
    ds.InstanceCreationDate = DT
    ds.InstanceCreationTime = TM

def get_ct_image_module(ds):
    # Type 1
    ds.ImageType = "ORIGINAL\SECONDARY\AXIAL"
    ds.SamplesperPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.RescaleIntercept = -1024.0
    ds.RescaleSlope = 1.0
    # Type 2
    ds.KVP = ""
    ds.AcquisitionNumber = ""

def get_image_pixel_macro(ds):
    # Type 1
    ds.Rows = 256
    ds.Columns = 256
    ds.PixelRepresentation = 0

def get_patient_module(ds):
    # Type 2
    ds.PatientsName = ""
    ds.PatientID = "Patient's ID"
    ds.PatientsBirthDate = ""
    ds.PatientsSex = "O"

def get_general_study_module(ds, DT, TM):
    # Type 1
    ds.StudyInstanceUID = ""
    # Type 2
    ds.StudyDate = DT
    ds.StudyTime = TM
    ds.ReferringPhysiciansName = ""
    ds.StudyID = ""
    ds.AccessionNumber = ""
    # Type 3
    #ds.StudyDescription = ""

def get_general_series_module(ds, DT, TM, modality):
    # Type 1
    ds.Modality = modality
    ds.SeriesInstanceUID = ""
    # Type 2
    ds.SeriesNumber = ""
    # Type 2C on Modality in ['CT', 'MR', 'Enhanced CT', 'Enhanced MR Image', 'Enhanced Color MR Image', 'MR Spectroscopy']. May not be present if Patient Orientation Code Sequence is present.
    #ds.PatientPosition = "HFS"

    # Type 3
    ds.SeriesDate = DT
    ds.SeriesTime = TM
    #ds.SeriesDescription = ""

def get_rt_series_module(ds, DT, TM, modality):
    # Type 1
    ds.Modality = modality
    ds.SeriesInstanceUID = ""
    # Type 2
    ds.SeriesNumber = ""
    ds.OperatorsName = ""

    # Type 3
    ds.SeriesDate = DT
    ds.SeriesTime = TM
    # ds.SeriesDescriptionCodeSequence = None
    # ds.ReferencedPerformedProcedureStepSequence = None
    # ds.RequestAttributesSequence = None
    # Performed Procedure Step Summary Macro...
    # ds.SeriesDescription = ""

def get_frame_of_reference_module(ds):
    # Type 1
    ds.FrameofReferenceUID = ""
    # Type 2
    ds.PositionReferenceIndicator = ""

def get_general_equipment_module(ds):
    # Type 1
    ds.Manufacturer = "pydicom"
    # Type 3
    ds.ManufacturersModelName = "https://github.com/raysearchlabs/dicomutils"
    ds.SoftwareVersions = "PyDICOM %s" % (dicom.__version__,)

def get_general_image_module(ds, DT, TM):
    # Type 2
    ds.InstanceNumber = ""
    # Type 3
    ds.AcquisitionDate = DT
    ds.AcquisitionTime = TM
    ds.ImagesinAcquisition = 1
    ds.DerivationDescription = "Generated from numpy"

def get_image_plane_module(ds):
    # Type 1
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0]
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    # Type 2
    ds.SliceThickness = 1.0
    # Type 3
    # ds.SliceLocation = 0

def get_multi_frame_module(ds):
    # Type 1
    ds.NumberofFrames = 1
    ds.FrameIncrementPointer = dicom.datadict.Tag(dicom.datadict.tag_for_name("GridFrameOffsetVector"))

def get_rt_dose_module(ds, current_study):
    # Type 1C on PixelData
    ds.SamplesperPixel = 1
    ds.DoseGridScaling = 1.0
    ds.SamplesperPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    # Type 1
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"

    # Type 1C if Dose Summation Type is any of the enumerated values. 
    ds.ReferencedRTPlanSequence = []
    if 'RTPLAN' in current_study:
        refplan = dicom.dataset.Dataset()
        refplan.ReferencedSOPClassUID = get_uid("RT Plan Storage")
        refplan.ReferencedSOPInstanceUID = current_study['RTPLAN'].SOPInstanceUID
        ds.ReferencedRTPlanSequence.append(refplan)

    # Type 1C on multi-frame
    ds.GridFrameOffsetVector = [0,1,2,3,4]

    # Type 1C
    if (ds.DoseSummationType == "FRACTION" or
        ds.DoseSummationType == "BEAM" or
        ds.DoseSummationType == "BRACHY" or
        ds.DoseSummationType == "CONTROL_POINT"):
        ds.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence = [dicom.dataset.Dataset()]
        # Type 1
        ds.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedFractionGroupNumber = 0
        # Type 1C
        if (ds.DoseSummationType == "BEAM" or
            ds.DoseSummationType == "CONTROL_POINT"):
            ds.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence = [dicom.dataset.Dataset()]
            # ... and on it goes...
            raise NotImplementedError
        elif ds.DoseSummationType == "BRACHY":
            raise NotImplementedError
    
    # Type 3
    # ds.InstanceNumber = 0
    # ds.DoseComment = "blabla"
    # ds.NormalizationPoint = [0,0,0]
    # ds.TissueHeterogeneityCorrection = "IMAGE" # or "ROI_OVERRIDE" or "WATER"

def get_rt_general_plan_module(ds, DT, TM, current_study):
    # Type 1
    ds.RTPlanLabel = "Plan"
    if 'RTSTRUCT' not in current_study:
        ds.RTPlanGeometry = "TREATMENT_DEVICE"
    else:
        ds.RTPlanGeometry = "PATIENT"
        ds.ReferencedStructureSetSequence = [dicom.dataset.Dataset()]
        ds.ReferencedStructureSetSequence[0].ReferencedSOPClassUID = get_uid("RT Structure Set Storage")
        ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID = current_study['RTSTRUCT'].SOPInstanceUID
    
    # Type 2
    ds.RTPlanDate = DT
    ds.RTPlanTime = TM

    # Type 3
    ds.RTPlanName = "PlanName"
    # ds.RTPlanDescription = ""
    # ds.InstanceNumber = 1
    # ds.TreatmentProtocols = ""
    ds.PlanIntent = "RESEARCH"
    # ds.TreatmentSties = ""
    if 'RTDOSE' in current_study:
        ds.ReferencedDoseSequence = [dicom.dataset.Dataset()]
        ds.ReferencedDoseSequence[0].ReferencedSOPClassUID = get_uid("RT Dose Storage")
        ds.ReferencedDoseSequence[0].ReferencedSOPInstanceUID = current_study['RTDOSE'].SOPInstanceUID
    # ds.ReferencedRTPlanSequence = []

    
def get_rt_fraction_scheme_module(ds, nfractions):
    ds.FractionGroupSequence = [dicom.dataset.Dataset()] # T1
    fg = ds.FractionGroupSequence[0]
    fg.FractionGroupNumber = 1 # T1
    fg.FractionGroupDescription = "Primary fraction group" # T3
    # fg.ReferencedDoseSequence = [] # T3
    # fg.ReferencedDoseReferenceSequence = [] # T3
    fg.NumberofFractionsPlanned = nfractions # T2
    # fg.NumberofFractionPatternDigitsPerDay # T3
    # fg.RepeatFractionCycleLength # T3
    # fg.FractionPattern # T3
    fg.NumberofBeams = len(ds.BeamSequence) # T1
    if fg.NumberofBeams != 0:
        fg.ReferencedBeamSequence = [dicom.dataset.Dataset() for i in range(fg.NumberofBeams)]
        for i in range(fg.NumberofBeams):
            refbeam = fg.ReferencedBeamSequence[i]
            beam = ds.BeamSequence[i]
            refbeam.ReferencedBeamNumber = beam.BeamNumber
            # refbeam.BeamDoseSpecificationPoint = [0,0,0]  # T3
            # refbeam.BeamDose = 10 # T3
            # refbeam.BeamDosePointDepth  # T3
            # refbeam.BeamDosePointEquivalentDepth # T3
            # refbeam.BeamDosePointSSD # T3
            refbeam.BeamMeterset = 100.0
    fg.NumberofBrachyApplicationSetups = 0

def cumsum(i):
    """Yields len(i)+1 values from 0 to sum(i)"""
    s = 0.0
    yield s
    for x in i:
        s += x
        yield s

def get_rt_patient_setup_module(ds, current_study):
    ps = dicom.dataset.Dataset()
    ps.PatientSetupNumber = 1
    ps.PatientPosition = current_study['PatientPosition']
    ds.PatientSetupSequence = [ps]
    return ps

def get_rt_beams_module(ds, nbeams, nleaves, leafwidths, current_study):
    """nleaves is a list [na, nb, nc, ...] and leafwidths is a list [wa, wb, wc, ...]
    so that there are na leaves with width wa followed by nb leaves with width wb etc."""
    ds.BeamSequence = [dicom.dataset.Dataset() for k in range(nbeams)]
    for i in range(nbeams):
        beam = ds.BeamSequence[i]
        beam.BeamNumber = i + 1
        beam.BeamName = "B{0}".format(i+1) # T3
        # beam.BeamDescription # T3
        beam.BeamType = "STATIC"
        beam.RadiationType = "PHOTON"
        # beam.PrimaryFluenceModeSequence = [] # T3
        # beam.HighDoseTechniqueType = "NORMAL" # T1C
        beam.TreatmentMachineName = "Linac" # T2
        # beam.Manufacturer = "" # T3
        # beam.InstitutionName # T3
        # beam.InstitutionAddress # T3
        # beam.InstitutionalDepartmentName # T3
        # beam.ManufacturersModelName # T3
        # beam.DeviceSerialNumber # T3
        beam.PrimaryDosimeterUnit = "MU" # T3
        # beam.ReferencedToleranceTableNumber # T3
        beam.SourceAxisDistance = 1000 # mm, T3
        beam.BeamLimitingDeviceSequence = [dicom.dataset.Dataset() for k in range(3)]
        beam.BeamLimitingDeviceSequence[0].RTBeamLimitingDeviceType = "ASYMX"
        #beam.BeamLimitingDeviceSequence[0].SourceToBeamLimitingDeviceDistance = 60 # T3
        beam.BeamLimitingDeviceSequence[0].NumberOfLeafJawPairs = 1
        beam.BeamLimitingDeviceSequence[1].RTBeamLimitingDeviceType = "ASYMY"
        #beam.BeamLimitingDeviceSequence[1].SourceToBeamLimitingDeviceDistance = 50 # T3
        beam.BeamLimitingDeviceSequence[1].NumberOfLeafJawPairs = 1
        beam.BeamLimitingDeviceSequence[2].RTBeamLimitingDeviceType = "MLCX"
        #beam.BeamLimitingDeviceSequence[2].SourceToBeamLimitingDeviceDistance = 40 # T3
        beam.BeamLimitingDeviceSequence[2].NumberOfLeafJawPairs = sum(nleaves)
        mlcsize = sum(n*w for n,w in zip(nleaves, leafwidths))
        beam.BeamLimitingDeviceSequence[2].LeafPositionBoundaries = list(x - mlcsize/2 for x in cumsum(w for n,w in zip(nleaves, leafwidths) for k in range(n)))
        if 'PatientPosition' in current_study:
            beam.ReferencedPatientSetupNumber = 1  # T3
        # beam.ReferencedReferenceImageSequence = []  # T3
        # beam.PlannedVerificationImageSequence = []  # T3
        beam.TreatmentDeliveryType = "TREATMENT"
        # beam.ReferencedDoseSequence = [] # T3
        beam.NumberofWedges = 0
        # beam.WedgeSequence = [] # T1C on NumberofWedges != 0
        beam.NumberofCompensators = 0
        beam.NumberofBoli = 0
        beam.NumberofBlocks = 0
        beam.FinalCumulativeMetersetWeight = 100
        beam.NumberofControlPoints = 2
        beam.ControlPointSequence = [dicom.dataset.Dataset() for k in range(2)]
        for j in range(2):
            cp = beam.ControlPointSequence[j]
            cp.ControlPointIndex = j
            cp.CumulativeMetersetWeight = j * beam.FinalCumulativeMetersetWeight / 1
            # cp.ReferencedDoseReferenceSequence = [] # T3
            # cp.ReferencedDoseSequence = [] # T1C on DoseSummationType == "CONTROL_POINT"
            # cp.NominalBeamEnergy = 6 # T3
            # cp.DoseRateSet = 100 # T3
            # cp.WedgePositionSequence = [] # T3
            if j == 0:
                cp.BeamLimitingDevicePositionSequence = [dicom.dataset.Dataset() for k in range(3)]
                cp.BeamLimitingDevicePositionSequence[0].RTBeamLimitingDeviceType = 'ASYMX'
                cp.BeamLimitingDevicePositionSequence[0].LeafJawPositions = [0,0]
                cp.BeamLimitingDevicePositionSequence[1].RTBeamLimitingDeviceType = 'ASYMY'
                cp.BeamLimitingDevicePositionSequence[1].LeafJawPositions = [0,0]
                cp.BeamLimitingDevicePositionSequence[2].RTBeamLimitingDeviceType = 'MLCX'
                cp.BeamLimitingDevicePositionSequence[2].LeafJawPositions = [0,0]*sum(nleaves)
                cp.GantryAngle = i * 360 / nbeams
                cp.GantryRotationDirection = 'NONE'
                if 'NominalEnergy' in current_study:
                    cp.NominalBeamEnergy = current_study['NominalEnergy']
                # cp.GantryPitchAngle = 0 # T3
                # cp.GantryPitchRotationDirection = "NONE" # T3
                cp.BeamLimitingDeviceAngle = 0
                cp.BeamLimitingDeviceRotationDirection = "NONE"
                cp.PatientSupportAngle = 0
                cp.PatientSupportRotationDirection = "NONE"
                # cp.TableTopEccentricAxisDistance = 0 # T3
                cp.TableTopEccentricAngle = 0
                cp.TableTopEccentricRotationDirection = "NONE"
                cp.TableTopPitchAngle = 0
                cp.TableTopPitchRotationDirection = "NONE"
                cp.TableTopRollAngle = 0
                cp.TableTopRollRotationDirection = "NONE"
                cp.TableTopVerticalPosition = ""
                cp.TableTopLongitudinalPosition = ""
                cp.TableTopLateralPosition = ""
                cp.IsocenterPosition = [0,0,0]
                # cp.SurfaceEntryPoint = [0,0,0] # T3
                # cp.SourceToSurfaceDistance = 70 # T3


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
                min_open_leafi = nmin(i for i in range(len(bldp['MLCX'].LeafJawPositions)/2) if bldp['MLCX'].LeafJawPositions[i] >= bldp['MLCX'].LeafJawPositions[i+nleaves] - opentolerance)
                max_open_leafi = nmax(i for i in range(len(bldp['MLCX'].LeafJawPositions)/2) if bldp['MLCX'].LeafJawPositions[i] >= bldp['MLCX'].LeafJawPositions[i+nleaves] - opentolerance)
                if min_open_leafi != None and max_open_leafi != None:
                    bldp['ASYMY'].LeafJawPositions = [bld['MLCX'].LeafPositionBoundaries[min_open_leafi],
                                                      bld['MLCX'].LeafPositionBoundaries[max_open_leafi + 1]]
            if bldp['MLCX'] != None and bldp['ASYMY'] != None:
                min_open_leaf = min(bldp['MLCX'].LeafJawPositions[i] for i in range(len(bldp['MLCX'].LeafJawPositions)/2) if bldp['MLCX'].LeafJawPositions[i] >= bldp['MLCX'].LeafJawPositions[i+nleaves] - opentolerance)
                max_open_leaf = max(bldp['MLCX'].LeafJawPositions[i] for i in range(len(bldp['MLCX'].LeafJawPositions)/2) if bldp['MLCX'].LeafJawPositions[i] >= bldp['MLCX'].LeafJawPositions[i+nleaves] - opentolerance)
                bldp['ASYMX'].LeafJawPositions = [min_open_leaf, max_open_leaf]

def conform_mlc_to_circle(beam, radius, center):
    bld = getblds(beam.BeamLimitingDeviceSequence)
    nleaves = len(bld['MLCX'].LeafPositionBoundaries)-1
    for cp in beam.ControlPointSequence:
        if hasattr(cp, 'BeamLimitingDevicePositionSequence') and cp.BeamLimitingDevicePositionSequence != None:
            bldp = getblds(cp.BeamLimitingDevicePositionSequence)
            for i in range(nleaves):
                y = (bld['MLCX'].LeafPositionBoundaries[i] + bld['MLCX'].LeafPositionBoundaries[i+1]) / 2
                if abs(y) < radius:
                    bldp['MLCX'].LeafJawPositions[i] = -np.sqrt(radius**2 - y**2)
                    bldp['MLCX'].LeafJawPositions[i + nleaves] = np.sqrt(radius**2 - y**2)

def get_structure_set_module(ds, DT, TM, current_study):
    ds.StructureSetLabel = "Structure Set" # T1
    # ds.StructureSetName = "" # T3
    # ds.StructureSetDescription = "" # T3
    # ds.InstanceNumber = "" # T3
    ds.StructureSetDate = DT # T2
    ds.StructureSetTime = TM # T2
    if 'CT' in current_study and len(current_study['CT']) > 0:
        reffor = dicom.dataset.Dataset()
        reffor.FrameofReferenceUID = get_current_study_uid('FrameofReferenceUID', current_study)
        reffor.RelatedFrameofReferenceUID = [] # T3
        refstudy = dicom.dataset.Dataset()
        refstudy.RTReferencedStudyUID = get_current_study_uid('StudyUID', current_study) # T3
        refseries = dicom.dataset.Dataset()
        refseries.SeriesInstanceUID = current_study['CT'][0].SeriesInstanceUID
        refseries.ContourImageSequence = [] # T3
        for image in current_study['CT']:
            imgref = dicom.dataset.Dataset()
            imgref.ReferencedSOPInstanceUID = image.SOPInstanceUID
            imgref.ReferencedSOPClassUID = image.SOPClassUID
            # imgref.ReferencedFrameNumber = "" # T1C on multiframe
            # imgref.ReferencedSegmentNumber = "" # T1C on segmentation
            refseries.ContourImageSequence.append(imgref)
        refstudy.RTReferencedSeriesSequence = [refseries]
        reffor.RTReferencedStudySequence = [refstudy]
        ds.ReferencedFrameOfReferenceSequence = [reffor] # T3
    ds.StructureSetROISequence = []

    return ds

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
    
def get_current_study_uid(prop, current_study):
    if prop not in current_study:
        current_study[prop] = generate_uid()
    return current_study[prop]


def build_rt_plan(current_study, **kwargs):
    FoRuid = get_current_study_uid('FrameofReferenceUID', current_study)
    studyuid = get_current_study_uid('StudyUID', current_study)
    seriesuid = generate_uid()
    sopinstanceuid = generate_uid()
    filename = "RTPLAN_%s.dcm" % (sopinstanceuid,)
    rp = get_default_rt_plan_dataset(filename, current_study)
    rp.SOPInstanceUID = sopinstanceuid
    rp.SeriesInstanceUID = seriesuid
    rp.StudyInstanceUID = studyuid
    rp.FrameofReferenceUID = FoRuid
    for k, v in kwargs.iteritems():
        if v != None:
            setattr(rp, k, v)
    return rp
        

def build_rt_dose(doseData, voxelGrid, current_study, **kwargs):
    nVoxels = doseData.shape
    rtdoseuid = generate_uid()
    FoRuid = get_current_study_uid('FrameofReferenceUID', current_study)
    studyuid = get_current_study_uid('StudyUID', current_study)
    seriesuid = generate_uid()
    sopinstanceuid = generate_uid()
    filename = "RTDOSE_%s.dcm" % (rtdoseuid,)
    rd = get_default_rt_dose_dataset(filename, current_study)
    rd.SOPInstanceUID = sopinstanceuid
    rd.SeriesInstanceUID = seriesuid
    rd.StudyInstanceUID = studyuid
    rd.FrameofReferenceUID = FoRuid
    rd.Rows = nVoxels[1]
    rd.Columns = nVoxels[0]
    rd.NumberofFrames = nVoxels[2]
    rd.PixelSpacing = [voxelGrid[1], voxelGrid[0]]
    rd.SliceThickness = voxelGrid[2]
    rd.GridFrameOffsetVector = [z*voxelGrid[2] for z in range(nVoxels[2])]
    rd.ImagePositionPatient = [-(nVoxels[0]-1)*voxelGrid[0]/2.0,
                               -(nVoxels[1]-1)*voxelGrid[1]/2.0,
                               -(nVoxels[2]-1)*voxelGrid[2]/2.0]
    if 'PatientPosition' in current_study:
        rd.PatientPosition = current_study['PatientPosition']
    
    rd.PixelData=doseData.tostring(order='F')
    for k, v in kwargs.iteritems():
        if v != None:
            setattr(rd, k, v)
    return rd

    
def build_rt_structure_set(rois, current_study, **kwargs):
    rtstructuid = generate_uid()
    FoRuid = get_current_study_uid('FrameofReferenceUID', current_study)
    studyuid = get_current_study_uid('StudyUID', current_study)
    seriesuid = generate_uid()
    sopinstanceuid = generate_uid()
    filename = "RTSTRUCT_%s.dcm" % (rtstructuid,)
    rs = get_default_rt_structure_set_dataset(filename, current_study)
    for roi in rois:
        structuresetroi = add_roi_to_structure_set(rs, roi['Name'], current_study)
        add_roi_to_roi_contour(rs, structuresetroi, roi['Contours'], current_study)
        add_roi_to_rt_roi_observation(rs, structuresetroi, roi['Name'], roi['InterpretedType'])
    rs.SOPInstanceUID = sopinstanceuid
    rs.SeriesInstanceUID = seriesuid
    rs.StudyInstanceUID = studyuid
    rs.FrameofReferenceUID = FoRuid
    rs.Rows = nVoxels[1]
    rs.Columns = nVoxels[0]
    rs.NumberofFrames = nVoxels[2]
    rs.PixelSpacing = [voxelGrid[1], voxelGrid[0]]
    rs.SliceThickness = voxelGrid[2]
    rs.GridFrameOffsetVector = [z*voxelGrid[2] for z in range(nVoxels[2])]
    rs.ImagePositionPatient = [-(nVoxels[0]-1)*voxelGrid[0]/2.0,
                               -(nVoxels[1]-1)*voxelGrid[1]/2.0,
                               -(nVoxels[2]-1)*voxelGrid[2]/2.0]
    if 'PatientPosition' in current_study:
        rs.PatientPosition = current_study['PatientPosition']
    for k, v in kwargs.iteritems():
        if v != None:
            setattr(rs, k, v)
    return rs

    
    
def build_ct(ctData, voxelGrid, current_study, **kwargs):
    nVoxels = ctData.shape
    ctbaseuid = generate_uid()
    FoRuid = get_current_study_uid('FrameofReferenceUID', current_study)
    studyuid = get_current_study_uid('StudyUID', current_study)
    seriesuid = generate_uid()
    cts=[]
    for z in range(nVoxels[2]):
        sopinstanceuid = "%s.%i" % (ctbaseuid, z)
        filename = "CT_%s.dcm" % (sopinstanceuid,)
        ct = get_default_ct_dataset(filename)
        ct.SOPInstanceUID = sopinstanceuid
        ct.SeriesInstanceUID = seriesuid
        ct.StudyInstanceUID = studyuid
        ct.FrameofReferenceUID = FoRuid
        ct.Rows = nVoxels[1]
        ct.Columns = nVoxels[0]
        ct.PixelSpacing = [voxelGrid[1], voxelGrid[0]]
        ct.SliceThickness = voxelGrid[2]
        ct.ImagePositionPatient = [-(nVoxels[0]-1)*voxelGrid[0]/2.0,
                                   -(nVoxels[1]-1)*voxelGrid[1]/2.0,
                                   -(nVoxels[2]-1)*voxelGrid[2]/2.0 + z*voxelGrid[2]]
        ct.PixelData=ctData[:,:,z].tostring(order='F')
        if 'PatientPosition' in current_study:
            ct.PatientPosition = current_study['PatientPosition']
        for k, v in kwargs.iteritems():
            if v != None:
                setattr(ct, k, v)
        cts.append(ct)
    return cts

def get_centered_coordinates(voxelGrid, nVoxels):
    x,y,z=np.mgrid[:nVoxels[0],:nVoxels[1],:nVoxels[2]]
    x=(x-(nVoxels[0]-1)/2.0)*voxelGrid[0]
    y=(y-(nVoxels[1]-1)/2.0)*voxelGrid[1]
    z=(z-(nVoxels[2]-1)/2.0)*voxelGrid[2]
    return x,y,z

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
    parser.add_argument('--voxelsize', dest='VoxelSize', default="1,2,4",
                        help='The size of a single voxel in mm. (default: 1,2,4)')
    parser.add_argument('--voxels', dest='Voxels', default="64,32,16",
                        help='The number of voxels in the dataset. (default: 64,32,16)')
    parser.add_argument('--modality', dest='modality', default=[], choices = ['CT', "RTDOSE", "RTPLAN", "RTSTRUCT"],
                        help='The modality to write. (default: CT)', action=ModalityGroupAction)
    parser.add_argument('--nominal-energy', dest='nominal_energy', default=None,
                        help='The nominal energy of beams in an RT Plan.')
    parser.add_argument('--values', dest='values', default=[], action='append',
                        help="""Set the Hounsfield or dose values in a volume to the given value.
                        For syntax, see the forthcoming documentation or the source code...""")
    parser.add_argument('--structure', dest='structures', default=[], action='append',
                        help="""Add a structure to the current list of structure sets.
                        For syntax, see the forthcoming documentation or the source code...""")
    
    args = parser.parse_args(namespace = argparse.Namespace(studies=[[]]))

    voxelGrid = [float(x) for x in args.VoxelSize.split(",")]
    nVoxels = [int(x) for x in args.Voxels.split(",")]
    x,y,z = get_centered_coordinates(voxelGrid, nVoxels)
    
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

    def build_box_contours(z, name, size, center, interpreted_type):
        #print "build_box_contours", z, name, size, center, interpreted_type
        if not hasattr(size, '__len__'):
            size = [size] * 3
        contours = np.array([[[X*size[0]/2 + center[0],
                               Y*X*size[1]/2 + center[1],
                               Z]
                              for X in [-1,1]
                              for Y in [-1,1]]
                             for Z in z[np.abs(z - center[2]) < size[2]/2]])
        return {'Name': name,
                'InterpretedType': interpreted_type,
                'Contours': contours}
    
    for study in args.studies:
        current_study = {}
        for series in study:
            ctData = np.zeros(nVoxels, dtype=np.int16)*1024
            for value in series.values:
                if value.find(",") == -1:
                    ctData[:] = float(value)
                else:
                    shape = value.split(",")[0]
                    if shape == "sphere":
                        val = value.split(",")[1]
                        radius = float(value.split(",")[2])
                        if len(value.split(",",3)) == 4:
                            center = value.split(",",3)[3]
                            center = [float(x) for x in center.lstrip('[').rstrip(']').split(",")]
                        else:
                            center = [0,0,0]
                        ctData[(x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 <= radius**2] = val

            if series.patient_position != None:
                current_study['PatientPosition'] = series.patient_position
            if series.nominal_energy != None:
                current_study['NominalEnergy'] = series.nominal_energy
            if series.modality == "CT":
                if 'PatientPosition' not in current_study:
                    parser.error("Patient position must be specified when writing CT images!")
                datasets = build_ct(ctData, voxelGrid, current_study = current_study)
                current_study['CT'] = datasets
                for ds in datasets:
                    dicom.write_file(ds.filename, ds)
            elif series.modality == "RTDOSE":
                rd = build_rt_dose(ctData, voxelGrid, current_study = current_study)
                current_study['RTDOSE'] = rd
                dicom.write_file(rd.filename, rd)
            elif series.modality == "RTPLAN":
                rp = build_rt_plan(current_study = current_study)
                for beam in rp.BeamSequence:
                    conform_mlc_to_circle(beam, 30, [0,0])
                    conform_jaws_to_mlc(beam)
                current_study['RTPLAN'] = rp
                dicom.write_file(rp.filename, rp)
            elif series.modality == "RTSTRUCT":
                structures = []
                for structure in series.structures:
                    shape = structure.split(",")[0]
                    if shape == 'sphere':
                        name = structure.split(",")[1]
                        radius = float(structure.split(",")[2])
                        interpreted_type = structure.split(",")[3]
                        if len(structure.split(",")) > 4:
                            center = structure.split(",",4)[4]
                            center = [float(x) for x in center.lstrip('[').rstrip(']').split(",")]
                        else:
                            center = [0,0,0]
                        structures.append(build_sphere_contours(z[0,0,:], name, radius, center, interpreted_type))
                    elif shape == 'box':
                        name = structure.split(",")[1]
                        size = structure.split(",",2)[2]
                        if size.startswith("["):
                            size = structure.split(",", 2)[2]
                            rest = size[size.find(']')+2:]
                            size = size[:size.find(']')+1]
                            size = [float(x) for x in size.lstrip('[').rstrip(']').split(",")]
                        else:
                            size = float(structure.split(",")[2])
                            rest = structure.split(",",3)[3]
                        interpreted_type = rest.split(",")[0]
                        if len(rest.split(",")) >= 2:
                            center = rest.split(",",1)[1]
                            center = [float(x) for x in center.lstrip('[').rstrip(']').split(",")]
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
                dicom.write_file(rs.filename, rs)