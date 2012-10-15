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
$ ./build_dicom.py \
      --patient-position HFS --values 1024 --voxelsize 5,5,5 --voxels 100,100,100 --modality CT \
      --structure external --modality RTSTRUCT
```

![Screenshot of 50x50x50 water phantom with outline] (https://github.com/raysearchlabs/dicomutils/wiki/simplebox.png)

Generate CT data with two cavities (one denser), rois covering them, a box outline, an arbitrary plan 
and a lightfield "dose":

```bash
$ ./build_dicom.py \
      --patient-position HFS --values 1024 \
        --values "sphere,0,25,[50;86.6;0]" --values "sphere,2024,25,[50;-86.6;0]" \
        --voxelsize 4,4,4 --voxels 50,50,50 --modality CT \
      --structure external \
        --structure "sphere,Ball,25,CAVITY,[50;86.6;0]" \
        --structure "sphere,Ball2,25,CAVITY,[50;-86.6;0]" --modality RTSTRUCT \
      --beams 3 --mlc-shape "circle,25" --jaw-shape "60,60" \
        --nominal-energy 6 --modality RTPLAN \
      --values 0 --values lightfield --modality RTDOSE
```

![Screenshot of plan with lightfield dose] (https://raw.github.com/wiki/raysearchlabs/dicomutils/lightfieldplan.png)