# -*- coding: utf-8 -*-

from enthought.traits.api \
    import HasTraits, Str, Regex, List, Instance, PythonValue, CInt, Property, on_trait_change
from enthought.traits.ui.api \
    import TreeEditor, TreeNode, View, Item, VSplit, \
           HGroup, Handler, Group
from enthought.traits.ui.menu \
    import Menu, Action, Separator
from enthought.traits.ui.wx.tree_editor \
    import NewAction, CopyAction, CutAction, \
           PasteAction, DeleteAction, RenameAction
from collections import defaultdict

import dicom

# DATA CLASSES

class SOPInstance(HasTraits):
    dicom_dataset = PythonValue
    sopinstanceuid = Property( depends_on = [ 'dicom_dataset' ] )
    
    label = Property( depends_on = [ 'dicom_dataset' ] )
    
    def _get_label(self):
        return "%s [%s]" % (dicom.UID.UID_dictionary[self.dicom_dataset.SOPClassUID][0].replace(" Storage",""), 
                            self.dicom_dataset.SOPInstanceUID)

    def _get_sopinstanceuid(self):
        if self.dicom_dataset != None:
            return str(self.dicom_dataset.SOPInstanceUID)
        else:
            return "N/A"
    # series = Instance(Series)

class Image(SOPInstance):
    pass

class RTImage(SOPInstance):
    # rtplan = Instance(RTPlan)
    # rttreatmentrecord = Instance(RTTreatmentRecord)
    pass

class RTStructureSet(SOPInstance):
    images = List(Image)
    # doses = List(RTDose)

class RTDose(SOPInstance):
    structure_sets = List(RTStructureSet)
    
    def _get_label(self):
        return "%s %s [%s]" % (dicom.UID.UID_dictionary[self.dicom_dataset.SOPClassUID][0].replace(" Storage",""), 
                               getattr(self.dicom_dataset, 'DoseSummationType', '<unknown summation>'),
                               self.dicom_dataset.SOPInstanceUID)
    # plan = Instance(RTPlan)

class ControlPoint(HasTraits):
    dicom_dataset = PythonValue
    index = Property( depends_on = [ 'dicom_dataset' ] )
    label = Property( depends_on = [ 'index' ] )
    
    def _get_label(self):
        return "CP %i" % (self.index,)

    def _get_index(self):
        if hasattr(self.dicom_dataset, 'ControlPointIndex'):
            return self.dicom_dataset.ControlPointIndex
        else:
            return -1
    # beam = Instance(Beam)

class Beam(HasTraits):
    control_points = List(ControlPoint)
    dicom_dataset = PythonValue
    name = Property( depends_on = [ 'dicom_dataset' ] )
    
    def _get_name(self):
        return self.dicom_dataset.BeamName
    # plan = Instance(RTPlan)

    @on_trait_change("dicom_dataset")
    def update_control_points(self, obj, name, new):
        self.control_points = [ControlPoint(dicom_dataset = cp, beam = self) for cp in new.ControlPoints]

class RTPlan(SOPInstance):
    doses = List(RTDose)
    structure_sets = List(RTStructureSet)
    rtimages = List(RTImage)
    beams = List(Beam)
    
    def _get_label(self):
        return "%s %s %s [%s]" % (dicom.UID.UID_dictionary[self.dicom_dataset.SOPClassUID][0].replace(" Storage",""), 
                               getattr(self.dicom_dataset, 'RTPlanLabel', '<missing label>'),
                               getattr(self.dicom_dataset, 'RTPlanName', '<missing name>'),
                               self.dicom_dataset.SOPInstanceUID)

    @on_trait_change("dicom_dataset")
    def update_beams(self, obj, name, new):
        self.beams = [Beam(dicom_dataset = beam, plan = self) for beam in new.Beams]

    # rttreatmentrecords = List # RTTreatmentRecord

class RTTreatmentRecord(SOPInstance):
    plan = Instance(RTPlan)
    images = List(RTImage)

specialsops = defaultdict(lambda: SOPInstance)
specialsops.update({
    '1.2.840.10008.5.1.4.1.1.481.1': RTImage,
    '1.2.840.10008.5.1.4.1.1.481.2': RTDose,
    '1.2.840.10008.5.1.4.1.1.481.3': RTStructureSet,
    '1.2.840.10008.5.1.4.1.1.481.4': RTTreatmentRecord,
    '1.2.840.10008.5.1.4.1.1.481.5': RTPlan,
    '1.2.840.10008.5.1.4.1.1.481.6': RTTreatmentRecord,
    '1.2.840.10008.5.1.4.1.1.481.7': RTTreatmentRecord,
    '1.2.840.10008.5.1.4.1.1.481.8': RTPlan,
    '1.2.840.10008.5.1.4.1.1.481.9': RTTreatmentRecord,
})

class Series(HasTraits):
    sopinstances = List(SOPInstance)
    # study = Instance(Study)
    dicom_datasets = List(PythonValue)

    series_instance_uid = Str
    
    label = Property( depends_on = [ 'dicom_dataset' ] )
    
    def _get_label(self):
        return "Series %s [%s]" % (self.dicom_datasets[0].Modality, self.series_instance_uid) 
    
    @on_trait_change("dicom_datasets[]")
    def update_sopinstances(self, obj, name, old, new):
        print "Series.update_sopinstances()"
        sopuids = defaultdict(lambda: [])
        for x in self.dicom_datasets:
            sopuids[x.SOPInstanceUID].append(x)
        sd = {x.sopinstanceuid: x for x in self.sopinstances}
        newuids = [uid for uid in sopuids if uid not in sd]
        goneuids = [uid for uid in sd if uid not in sopuids]
        for uid in newuids:
            assert len(sopuids[uid]) == 1
            cls = specialsops[sopuids[uid][0].SOPClassUID]
            print "building %s" % (cls,)
            self.sopinstances.append(cls(dicom_dataset = sopuids[uid][0], series = self))
        for uid in goneuids:
            self.sopinstances.pop(sd[uid])

class Study(HasTraits):
    series = List(Series)
    # patient = Instance(Patient)
    dicom_datasets = List(PythonValue)
    study_instance_uid = Str
    
    label = Property( depends_on = [ 'dicom_dataset' ] )
    
    def _get_label(self):
        return "Study %s [%s]" % (self.dicom_datasets[0].StudyID, self.study_instance_uid) 
    
    @on_trait_change("dicom_datasets[]")
    def update_series(self, obj, name, old, new):
        print "Study.update_series()"
        seriesuids = defaultdict(lambda: [])
        for x in self.dicom_datasets:
            seriesuids[x.SeriesInstanceUID].append(x)
        sd = {x.series_instance_uid: x for x in self.series}
        newuids = [uid for uid in seriesuids if uid not in sd]
        goneuids = [uid for uid in sd if uid not in seriesuids]
        updateduids = [uid for uid in sd if uid in seriesuids]
        for uid in newuids:
            self.series.append(Series(dicom_datasets = seriesuids[uid], 
                                      study = self,
                                      series_instance_uid = uid))
        for uid in goneuids:
            self.series.pop(sd[uid])
        for uid in updateduids:
            sd[uid].dicom_datasets = seriesuids[uid]

class Patient(HasTraits):
    studies = List(Study)
    dicom_datasets = List(PythonValue)
    
    patient_id = Str
    name = Str

    label = Property( depends_on = [ 'dicom_dataset' ] )
    def _get_label(self):
        return "%s <%s>" % (self.name, self.patient_id)

    @on_trait_change("dicom_datasets[]")
    def update_studies(self, obj, name, old, new):
        print "Patient.update_studies()"
        studyuids = defaultdict(lambda: [])
        for x in self.dicom_datasets:
            studyuids[x.StudyInstanceUID].append(x)
        sd = {x.study_instance_uid: x for x in self.studies}
        newuids = [uid for uid in studyuids if uid not in sd]
        goneuids = [uid for uid in sd if uid not in studyuids]
        updateduids = [uid for uid in sd if uid in studyuids]
        for uid in newuids:
            self.studies.append(Study(dicom_datasets = studyuids[uid], 
                                      patient = self,
                                      study_instance_uid = uid))
        for uid in goneuids:
            self.studies.pop(sd[uid])
        for uid in updateduids:
            sd[uid].dicom_datasets = studyuids[uid]

class PatientList(HasTraits):
    patients = List(Patient)

class Selection(HasTraits):
    plan = RTPlan
    beam = Beam
    control_point = ControlPoint
    dose = RTDose
    structure_set = RTStructureSet
    image = Image
    rtimage = RTImage
    series = Series
    study = Study
    patient = Patient
    

class Root(HasTraits):
    patientlist = PatientList
    filenames = List(Str)
    selection = Selection

    _loaded_files = {}

    def get_patient_with_id(self, id):
        for patient in self.patientlist.patients:
            if patient.patient_id == id:
                return patient
        return None

    @on_trait_change("filenames[]")
    def filenames_changed(self, obj, name, old, new):
        for filename in new:
            self._loaded_files[filename] = dicom.read_file(filename)
            patient = self.get_patient_with_id(self._loaded_files[filename].PatientID)
            if patient == None:
                self.patientlist.patients.append(Patient(dicom_datasets = [self._loaded_files[filename]],
                                                         name = self._loaded_files[filename].PatientName,
                                                         patient_id = self._loaded_files[filename].PatientID))
            else:
                patient.dicom_datasets.append(self._loaded_files[filename])
        for filename in old:
            patient = self.get_patient_with_id(self._loaded_files[filename].PatientID)
            patient.dicom_datasets.pop(self._loaded_files[filename])
            if len(patient.dicom_datasets) == 0:
                self.patientlist.patients.pop(patient)

RTImage.add_class_trait('rtplan', Instance(RTPlan))
RTDose.add_class_trait('rtplan', Instance(RTPlan))
RTImage.add_class_trait('rttreatmentrecord', Instance(RTTreatmentRecord))
RTPlan.add_class_trait('rttreatmentrecord', Instance(RTTreatmentRecord))
SOPInstance.add_class_trait('series', Instance(Series))
Series.add_class_trait('study', Instance(Study))
Study.add_class_trait('patient', Instance(Patient))
Beam.add_class_trait('plan', Instance(RTPlan))
ControlPoint.add_class_trait('beam', Instance(Beam))

# View for objects that aren't edited
no_view = View()

# Actions used by tree editor context menu

def_title_action = Action(name='Default title',
                          action = 'object.default')

import sys

root = Root(patientlist=PatientList())
for fn in sys.argv[1:]:
    try:
        dicom.read_file(fn)
        root.filenames.append(fn)
        print "added %s" % (fn,)
    except:
        continue
print "\n".join(root.filenames)

patient_action = Action(name='Patient', action='handler.dump_patient(editor,object)')

class TreeHandler ( Handler ):
    def dump_patient ( self, editor, object ):
        print 'dump_patient(%s)' % ( object, )

def on_tree_select(obj):
    print "on_tree_select(%s)" % (obj)
    if obj.__class__ is ControlPoint:
        root.selection.control_point = obj
        root.selection.beam = obj.beam
        root.selection.plan = obj.beam.plan
        root.selection.series = obj.beam.plan.series
        root.selection.study = obj.beam.plan.series.study
        root.selection.patient = obj.beam.plan.series.study.patient
    elif obj.__class__ is Beam:
        root.selection.control_point = None
        root.selection.beam = obj
        root.selection.plan = obj.plan
        root.selection.series = obj.plan.series
        root.selection.study = obj.plan.series.study
        root.selection.patient = obj.plan.series.study.patient
    elif obj.__class__ is RTPlan:
        root.selection.control_point = None
        root.selection.beam = None
        root.selection.plan = obj
        root.selection.series = obj.series
        root.selection.study = obj.series.study
        root.selection.patient = obj.series.study.patient
    elif obj.__class__ is Series:
        root.selection.control_point = None
        root.selection.beam = None
        root.selection.plan = None
        root.selection.series = obj
        root.selection.study = obj.study
        root.selection.patient = obj.study.patient
    elif obj.__class__ is Study:
        root.selection.control_point = None
        root.selection.beam = None
        root.selection.plan = None
        root.selection.series = None
        root.selection.study = obj
        root.selection.patient = obj.patient
    elif obj.__class__ is Patient:
        root.selection.control_point = None
        root.selection.beam = None
        root.selection.plan = None
        root.selection.series = None
        root.selection.study = None
        root.selection.patient = obj
    else:
        root.selection.control_point = None
        root.selection.beam = None
        root.selection.plan = None
        root.selection.series = None
        root.selection.study = None
        root.selection.patient = None
    print (root.selection.control_point, 
           root.selection.beam, 
           root.selection.plan, 
           root.selection.series, 
           root.selection.study, 
           root.selection.patient, 
           )
        

# Tree editor
tree_editor = TreeEditor(
    nodes = [
        TreeNode(node_for  = [ PatientList ],
                 auto_open = True,
                 children  = 'patients',
                 label     = '=Patients',
                 view      = no_view),
        TreeNode(node_for  = [ Patient ],
                 auto_open = True,
                 children  = 'studies',
                 label     = 'label',
                 view      = no_view),
        TreeNode(node_for  = [ Study ],
                 auto_open = True,
                 children  = 'series',
                 label     = 'label',
                 view      = no_view),
        TreeNode(node_for  = [ Series ],
                 auto_open = False,
                 children  = 'sopinstances',
                 label     = 'label',
                 view      = no_view),
        TreeNode(node_for  = [ RTPlan ],
                 auto_open = False,
                 children  = 'beams',
                 label     = 'label',
                 view      = no_view),
        TreeNode(node_for  = [ SOPInstance ],
                 auto_open = False,
                 children  = '',
                 label     = 'label',
                 view      = no_view),
        TreeNode(node_for  = [ Beam ],
                 auto_open = False,
                 children  = 'control_points',
                 label     = 'name',
                 view      = no_view),
        TreeNode(node_for  = [ ControlPoint ],
                 auto_open = False,
                 children  = '',
                 label     = 'label',
                 view      = no_view),
    ],
    on_select = on_tree_select
)

# The main view
view = View(
           Group(
               Item(
                    name = 'patientlist',
                    id = 'patientlist',
                    editor = tree_editor,
                    resizable = True ),
                orientation = 'vertical',
                show_labels = True,
                show_left = False, ),
            title = 'Patients',
            id = \
             'dicomutils.viewer.tree',
            dock = 'horizontal',
            drop_class = HasTraits,
            handler = TreeHandler(),
            buttons = [ 'Undo', 'OK', 'Cancel' ],
            resizable = True,
            width = .3,
            height = .3 )



if __name__ == '__main__':
    root.configure_traits( view = view )
    
