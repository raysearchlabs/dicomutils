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
      --patient-position HFS --values 0 --voxelsize 5,5,5 --voxels 100,100,100 --modality CT \
      --structure external --modality RTSTRUCT
```

![Screenshot of 50x50x50 water phantom with outline] (https://github.com/raysearchlabs/dicomutils/wiki/simplebox.png)

Generate CT data with two cavities (one denser), rois covering them, a box outline, an arbitrary plan 
and a lightfield "dose":

```bash
$ ./build_dicom.py \
      --patient-position HFS --values 0 \
        --values "sphere,-100,25,[50;86.6;0]" --values "box,100,25,[50;-86.6;0]" \
        --voxelsize 4,3,4 --voxels 48,64,48 --modality CT \
      --structure external \
        --structure "sphere,Ball,25,CAVITY,[50;86.6;0]" \
        --structure "box,Cube,25,CAVITY,[50;-86.6;0]" --modality RTSTRUCT \
      --beams "[3;123;270]" \
        --mlc-shape "1,circle,30" --jaw-shape "1,[60;60]" \
        --mlc-shape "2,rectangle,60,60" --jaw-shape "2,[70;70;10;10]" \
        --mlc-shape "3,rectangle,40,80" --jaw-shape "3,[40;80]" \
        --nominal-energy 6 --modality RTPLAN \
      --values 0 --values lightfield --modality RTDOSE
```

![Screenshot of plan with lightfield dose] (https://raw.github.com/wiki/raysearchlabs/dicomutils/lightfieldplan.png)