import configparser
from pyforms import BaseWidget
from pyforms.controls import ControlFile
from pyforms.controls import ControlButton
from pyforms.controls import ControlText
from pyforms.controls import ControlLabel
from pyforms.controls import ControlNumber
from pyforms.controls import ControlCombo

from Audiofile import Audiofile
from AudioStream import AudioStream
from OLTW import OLTW
from Feature import LibrosaMFCC
from Queueing import Queueing
from Connector import Connector

import os.path
import threading

# GUI Program to control the alignment system
class OPAQ(BaseWidget):

    CONFIG_PATH = "config.ini"

    # Settings
    fs = 48000
    
    fl = 250
    hopl = 100
    searchwidth = 200
    pathmeasure = 'weightedmean'
    delin = 'diagonal'

    def __init__(self, *args, **kwargs):
        super().__init__('opera.guru - Automated Queuing System')
        self.set_margin(6)

        self.loadSettings()

        self.createAlignmentTab()
        self.createSettingsTab()

        self._formset = [{
            'Alignment': [
                'ref_audio',
                'ref_queue',
                ('prepare_button', 'run_button'),
                ('targ_out', 'ref_out', 'chunk_out')],
            'Settings': [
                'control_fs',
                'control_fl',
                'control_hopl',
                'control_searchwidth',
                'control_pathmeasure',
                'control_delin'
            ]
        }]

    def before_close_event(self):
        self.saveSettings()

    def createAlignmentTab(self):

        self.ref_audio = ControlFile('Reference Audio')
        self.ref_audio.changed_event = self.selectRefAudio

        self.ref_queue = ControlFile('Referene Queueing')
        self.ref_queue.changed_event = self.selectRefQueue

        self.prepare_button = ControlButton('Prepare')
        self.prepare_button.value = self.onPrepare

        self.run_button = ControlButton('Run')
        self.run_button.value = self.onRun
        self.run_button.enabled = False

        self.targ_out = ControlLabel('Target: ')
        self.ref_out = ControlLabel('ref time')
        self.chunk_out = ControlLabel('Chunk: ')

    def createSettingsTab(self):
        self.control_fs = ControlNumber(default=self.fs, maximum=256000, label="Sample rate")
        self.control_fs.changed_event = lambda: setattr(self, 'fs', int(self.control_fs.value))

        self.control_fl = ControlNumber(default=self.fl, maximum=10000, label="Frame length (ms)")
        self.control_fl.changed_event = lambda: setattr(self, 'fl', int(self.control_fl.value))

        self.control_hopl = ControlNumber(default=self.hopl, maximum=self.fl, label="Hop length (ms)")
        self.control_hopl.changed_event = lambda: setattr(self, 'hopl', int(self.control_hopl.value))

        self.control_searchwidth = ControlNumber(default=self.searchwidth, maximum=2000, label="Search width (frames)")
        self.control_searchwidth.changed_event = lambda: setattr(self, 'searchwidth', int(self.control_searchwidth.value))

        self.control_pathmeasure = ControlCombo(default=self.pathmeasure)
        self.control_pathmeasure.add_item('Raw', 'raw')
        self.control_pathmeasure.add_item('Minimum mean', 'minmean')
        self.control_pathmeasure.add_item('Progressive mean', 'progminmean')
        self.control_pathmeasure.add_item('Rolling mean', 'rollingminmean')
        self.control_pathmeasure.add_item('Weighted mean', 'weightedmean')
        self.control_pathmeasure.add_item('Rolling weighted mean', 'rollingweightedmean')
        self.control_pathmeasure.value = self.pathmeasure
        self.control_pathmeasure.changed_event = lambda: setattr(self, 'pathmeasure', self.control_pathmeasure.value)

        self.control_delin = ControlCombo(default=self.delin)
        self.control_delin.add_item('None', 'none')
        self.control_delin.add_item('Axes', 'axes')
        self.control_delin.add_item('Diagonal', 'diagonal')
        self.control_delin.value = self.delin
        self.control_delin.changed_event = lambda: setattr(self, 'delin', self.control_delin.value)
        
        # Not really used anymore
        self.endtime = 60 # (s)

    def loadSettings(self):
        """
            Load settings from file
        """
        config = configparser.ConfigParser()

        if os.path.isfile(self.CONFIG_PATH):
        
            config.read(self.CONFIG_PATH)
            settings = config['settings']

            self.fs = int(settings['fs'])

            self.fl = int(settings['fl'])
            self.hopl = int(settings['hopl'])
            self.searchwidth = int(settings['searchwidth'])
            self.pathmeasure = settings['pathmeasure']
            self.delin = settings['delin']

    def saveSettings(self):
        """
            Save settings to file
        """
        config = configparser.ConfigParser()
        config['settings'] = {
            'fs': self.fs,
            'fl': self.fl,
            'hopl': self.hopl,
            'searchwidth': self.searchwidth,
            'pathmeasure': self.pathmeasure,
            'delin': self.delin
        }
        with open(self.CONFIG_PATH, 'w') as configFile:
            config.write(configFile)


    def selectRefAudio(self):
        print(self.ref_audio.value)

    def selectRefQueue(self):
        print(self.ref_queue.value)

    def onPrepare(self):
        threading.Thread(target=self.prepare).start()

    def prepare(self):

        mfcc = LibrosaMFCC(self.fs, self.fl ,self.hopl, ncoeff=120, cskip=10)
        self.oltw = OLTW(mfcc, inc_measure=self.pathmeasure, searchwidth=self.searchwidth, matrixsize=14000, save_analytics=True, start_deadzone=0, use_future=False, n_frames=36000, d_measure='manhattan', delin=self.delin, diagonalCost=2.5)
        print(f"Starting with {self.pathmeasure}, {self.delin}")

        # Load connector
        self.connector = Connector()

        self.queueing = Queueing(self.connector)
        self.queueing.loadCSV(self.ref_queue.value)

        reference = Audiofile(self.ref_audio.value, self.fs)
        self.oltw.loadReference(reference.signal)

        self.oltw.init()

        self.auds = AudioStream(self.fs, self.fl, self.hopl, self.oltw.processFrame, self.onStep)

        # (re)set labels
        self.run_button.enabled = True
        self.prepare_button.label = "Reset"
        self.targ_out.value = "Target: "
        self.ref_out.value = "Reference: "
        self.chunk_out.value = "Chunk: "

    def onStep(self, target_timestamp, reference_timestamp):
        
        # Update UI labels
        self.targ_out.value = "Target: " + str(target_timestamp)
        self.ref_out.value = "Reference: " + str(reference_timestamp)

        chunk = self.queueing.checkTimestamp(reference_timestamp)
        self.chunk_out.value = "Chunk: " + str(chunk)
    
    def onRun(self):
        algo = threading.Thread(target=self.run)
        algo.start()

        # Reset button
        self.run_button.label = "Stop"
        self.run_button.value = self.onStop

    def run(self):
        self.auds.record(self.endtime)

    def onStop(self):
        self.auds.stop()

        # Reset button
        self.run_button.enabled = False
        self.run_button.label = "Run"
        self.run_button.value = self.onRun

if __name__ == '__main__':

    from pyforms import start_app
    start_app(OPAQ, geometry=(200, 200, 400, 200))
