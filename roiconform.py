import builders
reload(builders)
import modules
reload(modules)
from builders import StudyBuilder
import matplotlib.pyplot as pp

import os
tmpdir = os.path.join(os.getenv("TEMP"), "studybuilder")
if not os.path.exists(tmpdir):
    os.mkdir(tmpdir)

sb = StudyBuilder(patient_position="HFS", patient_id="123", patients_name="Kalle^Kula", patients_birthdate = "20121212")
ct = sb.build_ct(num_voxels = [48,64,75], voxel_size = [4,3,4], rescale_slope = 1, rescale_intercept = -1024)
ct.clear(real_value = 0)
print ct.pixel_array.max(),ct.pixel_array.min()
ct.add_sphere(radius = 25, center = [0,0,0], real_value = -1000, mode = 'set')
print ct.pixel_array.max(),ct.pixel_array.min()
ct.add_box(size = [25,50,25], center = [0,0,0], stored_value = 300, mode = 'add')
print ct.pixel_array.max(),ct.pixel_array.min()

assert sb.seriesbuilders['CT'] == [ct]

rtstruct = sb.build_structure_set()
rtstruct.add_external_box()
sph = rtstruct.add_sphere(radius=70, center = [-50,0,-100], name='Sph-Organ', interpreted_type='CAVITY')
sph2 = rtstruct.add_sphere(radius = 25, center = [0,0,0], name='Sph-PTV', interpreted_type='PTV')

rtplan = sb.build_static_plan()
b1 = rtplan.build_beam(gantry_angle = 180, collimator_angle = 15, meterset = 100)
b1.conform_to_rectangle(1,1,[0,0])

rtplan.build(finalize_mlc = False)

modules.conform_mlc_to_roi(b1.rtbeam, sph.roi_contour, sb.current_study)
modules.conform_mlc_to_roi(b1.rtbeam, sph2.roi_contour, sb.current_study)
b1.finalize_mlc()

sb.write(tmpdir)
print tmpdir

import plotting as p

p.plot_cp(rtplan.datasets[0].Beams[0], rtplan.datasets[0].Beams[0].CPs[0])
p.plot_roi_in_cp(rtplan.datasets[0].Beams[0], rtplan.datasets[0].Beams[0].CPs[0], sph.roi_contour, sb.current_study)
pp.axis('image')