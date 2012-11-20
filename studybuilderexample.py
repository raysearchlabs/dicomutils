import dicom
import builders
reload(builders)
import modules
reload(modules)
from builders import StudyBuilder

import os
if not os.path.exists("/tmp/studybuilder"):
    os.mkdir("/tmp/studybuilder")

sb = StudyBuilder(patient_position="HFS", patient_id="123", patients_name="Kalle^Kula", patients_birthdate = "20121212")
ct = sb.build_ct(num_voxels = [48,64,48], voxel_size = [4,3,4], rescale_slope = 1, rescale_intercept = -1024)
ct.clear(real_value = 0)
print ct.pixel_array.max(),ct.pixel_array.min()
ct.add_sphere(radius = 25, center = [0,0,0], real_value = -1000, mode = 'set')
print ct.pixel_array.max(),ct.pixel_array.min()
ct.add_box(size = [25,50,25], center = [0,0,0], stored_value = 300, mode = 'add')
print ct.pixel_array.max(),ct.pixel_array.min()

assert sb.seriesbuilders['CT'] == [ct]

rtstruct = sb.build_structure_set()
rtstruct.add_external_box()
rtstruct.add_sphere(radius = 25, center = [0,0,0], name='Sph-PTV', interpreted_type='PTV')
rtstruct.add_box(size = [25,50,25], center = [0,0,0], name='Box-Organ', interpreted_type='CAVITY')

rtplan = sb.build_static_plan()
b1 = rtplan.build_beam(gantry_angle = 0, meterset = 100)
b1.conform_to_circle(25, [0,0])
b2 = rtplan.build_beam(gantry_angle = 120, meterset = 91)
b2.conform_to_rectangle(25, 50, [0,0])
b3 = rtplan.build_beam(gantry_angle = 240, meterset = 71)
b3.conform_to_rectangle(50, 25, [50,-50])
assert rtplan.beam_builders == [b1,b2,b3]

rtplan.build()

rtdose = sb.build_dose()
for beam in rtplan.beam_builders:
    rtdose.add_lightfield(beam.rtbeam, beam.meterset)


sb.write("/tmp/studybuilder")
