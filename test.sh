#!/bin/bash

tempdir="$(mktemp -d)"

./build_dicom.py --outdir "$tempdir" \
      --patient-position HFS --values 0 --pixel_representation unsigned \
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

fails=0
for i in "$tempdir"/*.dcm ; do
    dciodvfy "$i" 2>&1 | grep -v "Error - Missing attribute Type 2C Conditional Element=<Laterality> Module=<GeneralSeries>" | grep ^Error && fails=$(($fails+1))
done

dcentvfy "$tempdir"/*.dcm || fails=$(($fails+1))
rm -rf "$tempdir"

if [ "$fails" == 0 ] ; then
    echo Pass
else
    echo FAIL - $fails fails
    exit -1
fi
