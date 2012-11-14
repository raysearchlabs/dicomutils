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
ct = sb.build_ct(voxels = [48,64,48], voxel_size = [4,3,4], rescale_slope = 1, rescale_intercept = -1024)
ct.clear(real_value = 0)
ct.add_sphere(radius = 25, center = [0,0,0], stored_value = 24, mode = 'set')
ct.add_box(size = [25,50,25], center = [0,0,0], real_value = 1000, mode = 'add')

assert sb.seriesbuilders['CT'] == [ct]

rtplan = sb.build_static_plan()
b1 = rtplan.build_beam(gantry_angle = 0)
b2 = rtplan.build_beam(gantry_angle = 120)
b3 = rtplan.build_beam(gantry_angle = 240)
assert rtplan.beam_builders == [b1,b2,b3]

rtstruct = sb.build_structure_set(ct)
rtstruct.add_external_box()
rtstruct.add_sphere(radius = 25, center = [0,0,0], name='Sph-PTV', interpreted_type='PTV')
rtstruct.add_box(size = [25,50,25], center = [0,0,0], name='Box-Organ', interpreted_type='CAVITY')

sb.write("/tmp/studybuilder")

