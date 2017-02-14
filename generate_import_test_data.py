#!/usr/bin/python
# -*- coding: utf-8 -*-

import builders
reload(builders)
import modules
reload(modules)
from builders import StudyBuilder
import os
import sys


def get_pixel_representation(s='signed'):
    assert s == 'signed' or s == 'unsigned'
    if s == 'signed':
        return 1
    elif s == 'unsigned':
        return 0


def generate_mr_unsigned_in_short_range(out_dir, patient_id):
    sb = StudyBuilder(patient_position="HFS",
                      patient_id=patient_id,
                      patients_name="MrUnsigned^InShortRange",
                      patients_birthdate="20121212")
    ct = sb.build_mr(num_voxels=[48, 64, 10],
                     voxel_size=[4, 3, 4],  # [mm]
                     pixel_representation=get_pixel_representation('unsigned'))

    ct.clear(stored_value=0)
    ct.add_box(size=[25, 50, 5], center=[0, 0, 0], stored_value=12345, mode='set')

    # Smallest value is: 0 (in short range)
    # Largest value is: 12345 (in short range).

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    sb.write(out_dir)
    print out_dir


def generate_mr_unsigned_not_in_short_range(out_dir, patient_id):
    sb = StudyBuilder(patient_position="HFS",
                      patient_id=patient_id,
                      patients_name="MrUnsigned^NotInShortRange",
                      patients_birthdate="20121212")
    ct = sb.build_mr(num_voxels=[48, 64, 10],
                     voxel_size=[4, 3, 4],  # [mm]
                     pixel_representation=get_pixel_representation('unsigned'))

    ct.clear(stored_value=0)
    ct.add_box(size=[25, 50, 5], center=[0, 0, 0], stored_value=54321, mode='set')

    # Smallest value is: 0 (in short range)
    # Largest value is: 54321 (greater than 32767).

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    sb.write(out_dir)
    print out_dir


def generate_mr_signed_in_short_range(out_dir, patient_id):
    sb = StudyBuilder(patient_position="HFS",
                      patient_id=patient_id,
                      patients_name="MrSigned^InShortRange",
                      patients_birthdate="20121212")
    ct = sb.build_mr(num_voxels=[48, 64, 10],
                     voxel_size=[4, 3, 4],  # [mm]
                     pixel_representation=get_pixel_representation('signed'))

    ct.clear(stored_value=0)
    ct.add_box(size=[25, 50, 5], center=[0, 0, 0], stored_value=12345, mode='set')

    # Smallest value is: 0 (in short range)
    # Largest value is: 12345 (in short range).

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    sb.write(out_dir)
    print out_dir


def generate_ct_unsigned_rescaled_in_short_range(out_dir, patient_id):
    sb = StudyBuilder(patient_position="HFS",
                      patient_id=patient_id,
                      patients_name="CtUnsigned^InShortRange",
                      patients_birthdate="20121212")
    ct = sb.build_ct(num_voxels=[48, 64, 10],
                     voxel_size=[4, 3, 4],  # [mm]
                     pixel_representation=get_pixel_representation('unsigned'),
                     rescale_slope=1,
                     rescale_intercept=-1024)

    ct.clear(stored_value=0)
    ct.add_box(size=[25, 50, 5], center=[0, 0, 0], stored_value=12345, mode='set')

    # Smallest rescaled value is: 1 * 0 - 1024 = -1024 (in short range)
    # Largest rescaled value is: 1 * 54321 - 1024 = 11321 (in short range).

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    sb.write(out_dir)
    print out_dir


def generate_ct_unsigned_rescaled_not_in_short_range(out_dir, patient_id):
    sb = StudyBuilder(patient_position="HFS",
                      patient_id=patient_id,
                      patients_name="CtUnsigned^NotInShortRange",
                      patients_birthdate="20121212")
    ct = sb.build_ct(num_voxels=[48, 64, 10],
                     voxel_size=[4, 3, 4],  # [mm]
                     pixel_representation=get_pixel_representation('unsigned'),
                     rescale_slope=1,
                     rescale_intercept=-1024)

    ct.clear(stored_value=0)
    ct.add_box(size=[25, 50, 5], center=[0, 0, 0], stored_value=54321, mode='set')

    # Smallest rescaled value is: 1 * 0 - 1024 = -1024 (in short range)
    # Largest rescaled value is: 1 * 54321 - 1024 = 53297 (greater than 32767).

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    sb.write(out_dir)
    print out_dir


def generate_ct_signed_rescaled_in_short_range(out_dir, patient_id):
    sb = StudyBuilder(patient_position="HFS",
                      patient_id=patient_id,
                      patients_name="CtSigned^InShortRange",
                      patients_birthdate="20121212")
    ct = sb.build_ct(num_voxels=[48, 64, 10],
                     voxel_size=[4, 3, 4],  # [mm]
                     pixel_representation=get_pixel_representation('signed'),
                     rescale_slope=1,
                     rescale_intercept=-1024)

    ct.clear(stored_value=0)
    ct.add_box(size=[25, 50, 5], center=[0, 0, 0], stored_value=12345, mode='set')

    # Smallest rescaled value is: 1 * 0 - 1024 = -1024 (in short range)
    # Largest rescaled value is: 1 * 12345 - 1024 = 11321 (in short range).

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    sb.write(out_dir)
    print out_dir


def generate_ct_signed_rescaled_not_in_short_range(out_dir, patient_id):
    sb = StudyBuilder(patient_position="HFS",
                      patient_id=patient_id,
                      patients_name="CtSigned^RescaledNotInShortRange",
                      patients_birthdate="20121212")
    ct = sb.build_ct(num_voxels=[48, 64, 10],
                     voxel_size=[4, 3, 4],  # [mm]
                     pixel_representation=get_pixel_representation('signed'),
                     rescale_slope=5,
                     rescale_intercept=-1024)

    ct.clear(stored_value=0)
    ct.add_box(size=[25, 50, 5], center=[0, 0, 0], stored_value=12345, mode='set')

    # Smallest rescaled value is: 5 * 0 - 1024 = -1024 (in short range)
    # Largest rescaled value is: 5 * 12345 - 1024 = 60701 (greater than 32767).

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    sb.write(out_dir)
    print out_dir


def main(argv):
    generate_mr_unsigned_in_short_range(
        out_dir='./mr_unsigned_in_short_range',
        patient_id='1233321')
    generate_mr_unsigned_not_in_short_range(
        out_dir='./mr_unsigned_not_in_short_range',
        patient_id='1234321')
    generate_mr_signed_in_short_range(
        out_dir='./mr_signed_in_short_range',
        patient_id='1235321')
    generate_ct_unsigned_rescaled_in_short_range(
        out_dir='./ct_unsigned_rescaled_in_short_range',
        patient_id='1236321')
    generate_ct_unsigned_rescaled_not_in_short_range(
        out_dir='./ct_unsigned_rescaled_not_in_short_range',
        patient_id='1237321')
    generate_ct_signed_rescaled_in_short_range(
        out_dir='./ct_signed_rescaled_in_short_range',
        patient_id='1238321')
    generate_ct_signed_rescaled_not_in_short_range(
        out_dir='./ct_signed_rescaled_not_in_short_range',
        patient_id='1239321')


if __name__ == "__main__":
    main(sys.argv[1:])
