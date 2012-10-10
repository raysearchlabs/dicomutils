dicomutils
==========

A set of utilities for working with DICOM files.

The main utility is currently build_dicom, which can generate simple synthetic CT data, 
RT Structure sets, RT Doses and RT Plans.

All output files will be placed in the current working directory, and named as `<MODALITY>_<SOPINSTANCEUID>.dcm`, e.g. `CT_2.25.119389864082697057857042902898482259876.84.dcm`.

Examples
========

Get help:
```bash
$ ./build_dicom.py --help
```

Generate a 50cm x 50cm x 50cm water phantom CT data with 5mm resolution and a RT Structure set with a box ROI:

```bash
$ ./build_dicom.py --patient-position HFS --values 1024 --voxelsize 5,5,5 --voxels 100,100,100 --modality CT \
      --structure external --modality RTSTRUCT
```

![Screenshot of 50x50x50 water phantom with outline] (https://github.com/raysearchlabs/dicomutils/wiki/simplebox.png)

Generate CT data with non-cubic voxels showing a 5cm radius sphere of water in vacuum, with an ROI covering it, an RT Dose object with 50 Gy to the sphere 
and a random RT plan:

```bash
$ ./build_dicom.py --patient-position HFS --values 0 --values sphere,1024,20 --voxelsize 1,2,4 --voxels 120,60,30 --modality CT \
      --structure sphere,Ball,50,EXTERNAL --modality RTSTRUCT \
      --modality RTPLAN \
      --values 0 --values sphere,50,20 --modality RTDOSE
```