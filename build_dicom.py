#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import dicom, time, uuid, sys, datetime, os
import coordinates
from builders import StudyBuilder, TableTop, TableTopEcc
# Be careful to pass good fp numbers...
if hasattr(dicom, 'config'):
    dicom.config.allow_DS_float = True

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
parser.add_argument('--patient-position', dest='patient_position', choices = ['HFS', 'HFP', 'FFS', 'FFP', 'HFDR', 'HFDL', 'FFDR', 'FFDL'],
                    help='The patient position written in the images. Required for CT and MR. (default: not specified)')
parser.add_argument('--patient-id', dest='patient_id', default='Patient ID',
                    help='The patient ID.')
parser.add_argument('--patients-name', dest='patients_name', default='LastName^GivenName^^^',
                    help="The patient's name, in DICOM caret notation.")
parser.add_argument('--patients-birthdate', dest='patients_birthdate', default='',
                    help="The patient's birthdate, in DICOM DA notation (YYYYMMDD).")
parser.add_argument('--voxelsize', dest='VoxelSize', default="1,2,4",
                    help='The size of a single voxel in mm. (default: 1,2,4)')
parser.add_argument('--voxels', dest='NumVoxels', default="64,32,16",
                    help='The number of voxels in the dataset. (default: 64,32,16)')
parser.add_argument('--modality', dest='modality', default=[], choices=["CT", "MR", "RTDOSE", "RTPLAN", "RTSTRUCT"],
                    help='The modality to write. (default: CT)', action=ModalityGroupAction)
parser.add_argument('--nominal-energy', dest='nominal_energy', default=None,
                    help='The nominal energy of beams in an RT Plan.')
parser.add_argument('--values', dest='values', default=[], action='append', metavar='VALUE | SHAPE{,PARAMETERS}',
                    help="""Set the Hounsfield or dose values in a volume to the given value.\n\n\n
                    For syntax, see the forthcoming documentation or the source code...""")
parser.add_argument('--pixel_representation', dest='pixel_representation', default='signed', choices=['signed', 'unsigned'],
                    help="""signed: Stored pixel value type is int16, unsigned: Stored pixel value type is uint16.""")
parser.add_argument('--center', dest='center', default="[0;0;0]", help="""Center of the image, in dicom patient coordinates.""")
parser.add_argument('--sad', dest='sad', default=1000, help="The Source to Axis distance.")
parser.add_argument('--structure', dest='structures', default=[], action='append', metavar='SHAPE{,PARAMETERS}',
                    help="""Add a structure to the current list of structure sets.
                    For syntax, see the forthcoming documentation or the source code...""")
parser.add_argument('--beams', dest='beams', default='3',
                    help="""Set the number of equidistant beams to write in an RTPLAN.""")
parser.add_argument('--meterset', dest='meterset', default='1.0',
                    help="""Set the beam meterset weight, either as a single number for all beams or as [1;2;3;...] to specify the meterset weight for all beams in an RTPLAN.""")
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
parser.add_argument('--mlc-direction', dest='mlc_direction', default='MLCX',
                    help="""Set the direction of the MLC - MLCX or MLCY.""")
parser.add_argument('--mlc-shape', dest='mlcshapes', default=[], action='append',
                    help="""Add an opening to the current list of mlc openings.
                    For syntax, see the forthcoming documentation or the source code...""")
parser.add_argument('--jaw-shape', dest='jawshapes', default=[], action='append',
                    help="""Sets the jaw shape to x * y, centered at (xc, yc). Given as [x;y;xc;yc]. Defaults to conforming to the MLC.""")
parser.add_argument('--outdir', dest='outdir', default='.',
                    help="""Generate data to this directory. (default: working directory)""")
args = parser.parse_args(namespace = argparse.Namespace(studies=[[]]))
voxel_size = [float(x) for x in args.VoxelSize.split(",")]
num_voxels = [int(x) for x in args.NumVoxels.split(",")]
if args.pixel_representation == 'signed':
    pixel_representation = 1
else:
    pixel_representation = 0

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
for study in args.studies:
    sb = StudyBuilder()
    for series in study:
        if series.center.__class__ is str:
            series.center = [float(b) for b in series.center.lstrip('[').rstrip(']').split(";")]
        if series.modality == "CT":
            if 'PatientPosition' not in sb.current_study:
                parser.error("Patient position must be specified when writing CT images!")
            ib = sb.build_ct(
                num_voxels=num_voxels,
                voxel_size=voxel_size,
                pixel_representation=pixel_representation,
                center=np.array(series.center))
        if series.modality == "MR":
            if 'PatientPosition' not in sb.current_study:
                parser.error("Patient position must be specified when writing MR images!")
            ib = sb.build_mr(
                num_voxels=num_voxels,
                voxel_size=voxel_size,
                pixel_representation=pixel_representation,
                center=np.array(series.center))
        elif series.modality == "RTDOSE":
            ib = sb.build_dose(
                num_voxels=num_voxels,
                voxel_size=voxel_size,
                center=np.array(series.center))
        elif series.modality == "RTPLAN":
            isocenter = [float(b) for b in series.isocenter.lstrip('[').rstrip(']').split(";")]
            rp = sb.build_static_plan(nominal_beam_energy = series.nominal_energy,
                                      isocenter = isocenter,
                                      mlc_direction = series.mlc_direction,
                                      sad = series.sad)
        elif series.modality == "RTSTRUCT":
            rtstruct = sb.build_structure_set()
        else:
            assert "Unknown modality"

        for value in series.values:
            value = value.split(",")
            if len(value) == 1 and (value[0][0].isdigit() or value[0][0] == '-'):
                ib.clear(float(value[0]))
            else:
                shape = value[0]
                if shape == "sphere":
                    val = float(value[1])
                    radius = float(value[2])
                    if len(value) > 3:
                        center = [float(c) for c in value[3].lstrip('[').rstrip(']').split(";")]
                    else:
                        center = [0,0,0]
                    ib.add_sphere(radius=radius, center=center, stored_value=val, mode='set')
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
                    ib.add_box(size=size, center=center, stored_value=val, mode='set')
                elif shape == "lightfield":
                    for beam in sb.seriesbuilders['RTPLAN'][-1].beam_builders:
                        ib.add_lightfield(beam.rtbeam, beam.meterset)
        if series.patient_position != None:
            sb.current_study['PatientPosition'] = series.patient_position
        if series.patient_id != None:
            sb.current_study['PatientID'] = series.patient_id
        if series.patients_name != None:
            sb.current_study['PatientsName'] = series.patients_name
        if series.patients_birthdate != None:
            sb.current_study['PatientsBirthDate'] = series.patients_birthdate
        if series.modality == "CT":
            ib.build()
        elif series.modality == "MR":
            ib.build()
        elif series.modality == "RTDOSE":
            ib.build()
        elif series.modality == "RTPLAN":
            if all(d.isdigit() for d in series.beams):
                nbeams = int(series.beams)
                gantry_angles = [i * 360 / nbeams for i in range(nbeams)]
            else:
                gantry_angles = [int(b) for b in series.beams.lstrip('[').rstrip(']').split(";")]
                nbeams = len(gantry_angles)
            if all(d.isdigit() for d in series.collimator_angles):
                collimator_angles = [int(series.collimator_angles)] * nbeams
            else:
                collimator_angles = [int(b) for b in series.collimator_angles.lstrip('[').rstrip(']').split(";")]
            if all(d.isdigit() or d=='.' for d in series.meterset):
                meterset = [float(series.meterset)] * nbeams
            else:
                meterset = [float(b) for b in series.meterset.lstrip('[').rstrip(']').split(";")]
            if all(d.isdigit() for d in series.patient_support_angles):
                patient_support_angles = [int(series.patient_support_angles)] * nbeams
            else:
                patient_support_angles = [int(b) for b in series.patient_support_angles.lstrip('[').rstrip(']').split(";")]
            table_top = TableTop(*[float(b) for b in series.table_top.split(",")])
            table_top_eccentric = TableTopEcc(*[float(b) for b in series.table_top_eccentric.split(",")])
            for i in range(nbeams):
                rp.build_beam(gantry_angle = gantry_angles[i], meterset=meterset[i], collimator_angle=collimator_angles[i], patient_support_angle=patient_support_angles[i], table_top=table_top, table_top_eccentric=table_top_eccentric)
            for mlcshape in series.mlcshapes:
                mlcshape = mlcshape.split(",")
                if all(d.isdigit() for d in mlcshape[0]):
                    beams = [rp.beam_builders[int(mlcshape[0])-1]]
                    mlcshape=mlcshape[1:]
                else:
                    beams = rp.beam_builders
                if mlcshape[0] == "circle":
                    radius = float(mlcshape[1])
                    if len(mlcshape) > 2:
                        center = [float(c) for c in mlcshape[2].lstrip('[').rstrip(']').split(";")]
                    else:
                        center = [0,0]
                    for beam in beams:
                        beam.conform_to_circle(radius, center)
                elif mlcshape[0] == "rectangle":
                    X,Y = float(mlcshape[1]),float(mlcshape[2])
                    if len(mlcshape) > 3:
                        center = [float(c) for c in mlcshape[3].lstrip('[').rstrip(']').split(";")]
                    else:
                        center = [0,0]
                    for beam in beams:
                        beam.conform_to_rectangle(X, Y, center)
            for beam in beams:
                beam.conform_jaws_to_mlc()
            for jawshape in series.jawshapes:
                jawshape = jawshape.split(",")
                if len(jawshape) == 2:
                    beams = [rp.beam_builders[int(jawshape[0])-1]]
                    jawshape=jawshape[1:]
                else:
                    beams = rp.beam_builders
                jawsize = [float(c) for c in jawshape[0].lstrip('[').rstrip(']').split(";")]
                if len(jawsize) > 2:
                    center = [jawsize[2],jawsize[3]]
                else:
                    center = [0,0]
                for beam in beams:
                    beam.conform_jaws_to_rectangle(jawsize[0], jawsize[1], center)
            rp.build()
        elif series.modality == "RTSTRUCT":
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
                    rtstruct.add_sphere(name=name, radius=radius, center=center, interpreted_type=interpreted_type)
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
                    rtstruct.add_box(name=name, size=size, center=center, interpreted_type=interpreted_type)
                elif shape == "external":
                    rtstruct.add_external_box()
            rtstruct.build()
sb.write(args.outdir)
