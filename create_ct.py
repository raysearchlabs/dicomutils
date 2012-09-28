#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy, dicom, time, uuid, sys, datetime

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

def get_empty_dataset(filename):
    file_meta = dicom.dataset.Dataset()
    file_meta.MediaStorageSOPClassUID = get_uid("CT Image Storage")
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = ImplementationClassUID
    ds = dicom.dataset.FileDataset(filename, {}, file_meta=file_meta, preamble="\0"*128)
    return ds

def get_default_ct_dataset(filename):
    DT = "%04i%02i%02i" % datetime.datetime.now().timetuple()[:3]
    TM = "%02i%02i%02i" % datetime.datetime.now().timetuple()[3:6]
    ds = get_empty_dataset(filename)
    get_sop_commom_module(ds)
    get_ct_image_module(ds)
    get_image_pixel_macro(ds)
    get_patient_module(ds)
    get_general_study_module(ds)
    get_general_series_module(ds)
    get_frame_of_reference_module(ds)
    get_general_equipment_module(ds)
    get_general_image_module(ds)
    get_image_plane_module(ds)
    return ds

def get_sop_commom_module(ds):
    # Type 1
    ds.SOPClassUID = get_uid("CT Image Storage")
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

def get_general_study_module(ds):
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

def get_general_series_module(ds):
    # Type 1
    ds.Modality = "CT"
    ds.SeriesInstanceUID = ""
    # Type 2
    ds.SeriesNumber = ""
    # Type 3
    ds.SeriesDate = DT
    ds.SeriesTime = TM
    #ds.SeriesDescription = ""
    #ds.PatientPosition = "HFS"

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

def get_general_image_module(ds):
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
    
def write_ct(filenamebase, ctData, voxelGrid, **kwargs):
    nVoxels = ctData.shape
    ctbaseuid = generate_uid()
    FoRuid = generate_uid()
    studyuid = generate_uid()
    seriesuid = generate_uid()
    for z in range(nVoxels[2]):
        filename = "%s-%i.dcm" % (filenamebase, z)
        ct = get_default_ct_dataset(filename)
        ct.SOPInstanceUID = "%s.%i" % (ctbaseuid, z)
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
        for k, v in kwargs.iteritems():
            if v != None:
                setattr(ct, k, v)
        dicom.write_file(filename, ct)

def get_centered_coordinates(voxelGrid, nVoxels):
    x,y,z=numpy.mgrid[:nVoxels[0],:nVoxels[1],:nVoxels[2]]
    x=(x-(nVoxels[0]-1)/2.0)*voxelGrid[0]
    y=(y-(nVoxels[1]-1)/2.0)*voxelGrid[1]
    z=(z-(nVoxels[2]-1)/2.0)*voxelGrid[2]
    return x,y,z

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create DICOM CT data.')
    parser.add_argument('filenameBase', metavar='filename-base', 
                        help='The base of the generated filenames. For example, specifying "foo" gives files called "foo-0.dcm", "foo-1.dcm" etc.') 
    parser.add_argument('--patient-position', dest='PatientPosition', choices = ['HFS', 'HFP', 'FFS', 'FFP', 'HFDR', 'HFDL', 'FFDR', 'FFDP'],
                        help='The patient position written in the images (default: not specified)')
    parser.add_argument('--voxelsize', dest='VoxelSize', default="1,2,4",
                        help='The size of a single voxel in mm. (default: 1,2,4)')
    parser.add_argument('--voxels', dest='Voxels', default="64,32,16",
                        help='The number of voxels in the dataset. (default: 64,32,16)')
    

    args = parser.parse_args()

    voxelGrid = [float(x) for x in args.VoxelSize.split(",")]
    nVoxels = [int(x) for x in args.Voxels.split(",")]
    x,y,z = get_centered_coordinates(voxelGrid, nVoxels)
    
    ctData = numpy.ones(nVoxels, dtype=numpy.int16)*1024
    ctData += numpy.arange(nVoxels[0]).reshape((nVoxels[0],1,1))
    ctData += numpy.arange(nVoxels[1]).reshape((1,nVoxels[1],1))*10
    ctData += numpy.arange(nVoxels[2]).reshape((1,1,nVoxels[2]))*100
    ctData -= 1000*(numpy.sqrt(x**2+y**2+z**2) < 30)


    write_ct(args.filenameBase, ctData, voxelGrid, PatientPosition = args.PatientPosition)
