import shutil
try:
    shutil.rmtree("orientationtests")
except:
    pass
import builders
reload(builders)
import modules
reload(modules)
from builders import StudyBuilder

import os
if not os.path.exists("orientationtests"):
    os.mkdir("orientationtests")

def build_orientation(patient_position, row_direction, column_direction, frame_of_reference_uid = None):
    sb = StudyBuilder(patient_position=patient_position, patient_id="OrientationTests", patient_name="Orientation^Tests", patient_birthdate = "20121212")
    if frame_of_reference_uid != None:
        sb.current_study['FrameOfReferenceUID'] = frame_of_reference_uid

    print "building %s..." % (patient_position,)
    print "ct"
    ct = sb.build_ct(
        num_voxels=[7, 7, 7],
        voxel_size=[4, 4, 4],
        pixel_representation=0,
        rescale_slope=1,
        rescale_intercept=-1024,
        row_direction=row_direction,
        column_direction=column_direction)
    ct.clear(real_value = -1000)
    ct.add_box(size = [4,4,4], center = [0,0,0], real_value = 0)
    ct.add_box(size = [20,4,4], center = [0,-8,-8], real_value = 0)
    ct.add_box(size = [4,20,4], center = [8,0,-8], real_value = 0)
    ct.add_box(size = [4,4,20], center = [8,8,0], real_value = 0)
    ct.add_sphere(radius = 4, center = [-8,-8,-8], real_value = 0)

    print "rtstruct"
    rtstruct = sb.build_structure_set(ct)
    rtstruct.add_external_box()
    rtstruct.add_box(size = [4,4,4], center = [0,0,0], name='CenterVoxel', interpreted_type='SITE')
    rtstruct.add_box(size = [20,4,4], center = [0,-8,-8], name='x=-8 to 8, y=z=-8', interpreted_type='SITE')
    rtstruct.add_box(size = [4,20,4], center = [8,0,-8], name='y=-8 to 8 x=8, z=-8', interpreted_type='SITE')
    rtstruct.add_box(size = [4,4,20], center = [8,8,0], name='z=-8 to 8, x=y=8', interpreted_type='SITE')
    rtstruct.add_sphere(radius=4, center = [-8,-8,-8], name='x=y=z=-8', interpreted_type='SITE')
    rtstruct.build()

    print "rtplan"
    rtplan = sb.build_static_plan(structure_set = rtstruct, sad=20)
    b1 = rtplan.build_beam(gantry_angle = 0, collimator_angle=30, meterset = 100)
    b1.conform_to_rectangle(4, 4, [0,0])
    b2 = rtplan.build_beam(gantry_angle = 120, meterset = 100)
    b2.conform_to_rectangle(4, 4, [4,4])
    rtplan.build()

    print "rtdose"
    rtdose = sb.build_dose(planbuilder = rtplan)
    for beam in rtplan.beam_builders:
        rtdose.add_lightfield(beam.rtbeam, beam.meterset)

    return sb

orientations = [([0,1,0],[1,0,0]),([0,-1,0],[-1,0,0]),([0,1,0],[-1,0,0]),([0,-1,0],[1,0,0])]
patientpositions = ['HFS','HFP','FFS','FFP','HFDR', 'HFDL', 'FFDR', 'FFDL']
sbs = []
FoR = None
for o in orientations:
    for p in patientpositions:
        sb = build_orientation(p, o[0], o[1], FoR)
        sbs.append(sb)
        FoR = sbs[0].current_study['FrameOfReferenceUID']
        d = "orientationtests/" + p + "/" + "%s%s%s%s%s%s" % tuple(x for y in o for x in y)
        os.makedirs(d)
        sb.write(d)
