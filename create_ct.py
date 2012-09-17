#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy, dicom, time, uuid, sys, datetime

# Be careful to pass good fp numbers...
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

    # SOP Commom Module
    # Type 1
    ds.SOPClassUID = get_uid("CT Image Storage")
    ds.SOPInstanceUID = ""
    # Type 3
    ds.InstanceCreationDate = DT
    ds.InstanceCreationTime = TM

    # CT Image Module
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
    # Image Pixel Macro
    # Type 1
    ds.Rows = 256
    ds.Columns = 256
    ds.PixelRepresentation = 0

    # Patient Module
    # Type 2
    ds.PatientsName = ""
    ds.PatientID = "Patient's ID"
    ds.PatientsBirthDate = ""
    ds.PatientsSex = "O"

    # General Study Module
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

    # General Series Module
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

    # Frame of Reference Module
    # Type 1
    ds.FrameofReferenceUID = ""
    # Type 2
    ds.PositionReferenceIndicator = ""

    # General Equipment Module
    # Type 1
    ds.Manufacturer = "pydicom"
    # Type 3
    ds.ManufacturersModelName = "https://github.com/rickardraysearch/dicomutils"
    ds.SoftwareVersions = "PyDICOM %s" % (dicom.__version__,)

    # General Image Module
    # Type 2
    ds.InstanceNumber = ""
    # Type 3
    ds.AcquisitionDate = DT
    ds.AcquisitionTime = TM
    ds.ImagesinAcquisition = 1
    ds.DerivationDescription = "Generated from numpy"

    # Image Plane Module
    # Type 1
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0]
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    # Type 2
    ds.SliceThickness = 1.0
    # Type 3
    # ds.SliceLocation = 0
    return ds
    
def write_ct(filenamebase, ctData, pixelGrid):
    nPixels = ctData.shape
    ctbaseuid = generate_uid()
    FoRuid = generate_uid()
    studyuid = generate_uid()
    seriesuid = generate_uid()
    for z in range(nPixels[2]):
        filename = "%s-%i.dcm" % (filenamebase, z)
        ct = get_default_ct_dataset(filename)
        ct.SOPInstanceUID = "%s.%i" % (ctbaseuid, z)
        ct.SeriesInstanceUID = seriesuid
        ct.StudyInstanceUID = studyuid
        ct.FrameofReferenceUID = FoRuid
        ct.Rows = nPixels[0]
        ct.Columns = nPixels[1]
        ct.PixelSpacing = [pixelGrid[0], pixelGrid[1]]
        ct.SliceThickness = pixelGrid[2]
        ct.ImagePositionPatient = [-(nPixels[0]-1)*pixelGrid[0]/2.0,
                                   -(nPixels[1]-1)*pixelGrid[1]/2.0,
                                   -(nPixels[2]-1)*pixelGrid[2]/2.0 + z*pixelGrid[2]]
        ct.PixelData=ctData[:,:,z].tostring(order='C')
        dicom.write_file(filename, ct)

def get_centered_coordinates(pixelGrid, nPixels):
    x,y,z=numpy.mgrid[:nPixels[0],:nPixels[1],:nPixels[2]]
    x=(x-(nPixels[0]-1)/2.0)*pixelGrid[0]
    y=(y-(nPixels[1]-1)/2.0)*pixelGrid[1]
    z=(z-(nPixels[2]-1)/2.0)*pixelGrid[2]
    return x,y,z

if __name__ == '__main__':
    pixelGrid = [2,4,1] # mm x y z
    nPixels = [64,32,128]
    x,y,z = get_centered_coordinates(pixelGrid, nPixels)
    
    ctData = numpy.ones(nPixels, dtype=numpy.int16)*1024
    ctData += numpy.arange(nPixels[0]).reshape((nPixels[0],1,1))
    ctData += numpy.arange(nPixels[1]).reshape((1,nPixels[1],1))*10
    ctData += numpy.arange(nPixels[2]).reshape((1,1,nPixels[2]))*100
    ctData -= 1000*(numpy.sqrt(x**2+y**2+z**2) < 50)
    
    write_ct(sys.argv[1], ctData, pixelGrid)
