import numpy as np
import Audiofile
import time
import sounddevice as sd
import sys

# Represents audio input in the form of a live microphone recording
class AudioStream:

    buffer = None       # Audio buffer
    buffer_index = 0    # Index in audio buffer up to which has been processed    

    starttime = 0
    stepstarttime = 0
    behindcorrection = 0

    def __init__(self, fs, fl, hopl, alignmentFunc, queueFunc):
        """
            fs: frequency
            fl: frame length (ms)
            hopl: hop length (ms)
            alignmentFunc: Function to call with collected frames
            queueFunc: Function to call to check if a queueing request should be sent
        """

        self.fs = fs
        self.flms = fl                       # Frame size (ms)
        self.flf = 1000 / self.flms          # Frame frequency (Hz)
        self.fl = int(fs / self.flf)    # Frame length (number of array indices)

        self.hoplms = hopl
        self.hoplf = 1000 / self.hoplms
        self.hopl = int(fs / self.hoplf)

        self.buffer = np.zeros(100000, dtype=np.float32)
        self.buffer_index = self.fl - self.hopl

        self.alignmentFunc = alignmentFunc
        self.queueFunc = queueFunc

    def simulate(self, audiofile, realtime=False, starttime=0, endtime=0):
        """
            Simulate a recording from an existing audiofile
            audiofile: audiofile to receive in parts
            realtime: Incorporate sleep to simulate real time performance 
            endtime: Timestamp (s) to simulate until
        """

        if endtime != 0:
            self.buffer = audiofile.section(starttime, endtime)
        else:
            self.buffer = audiofile.signal

        n_frames = int(self.buffer.shape[0] / self.hopl)

        self.starttime = time.time()
        self.stepstarttime = 0

        for i in range(n_frames):

            self.buffer_index += self.hopl
            frame = self.buffer[self.buffer_index - self.fl:self.buffer_index]

            targtimestamp, reftimestamp = self.alignmentFunc(frame)
            currtime = time.time() - self.starttime

            self.queueFunc(targtimestamp, reftimestamp)
            ratio = reftimestamp / targtimestamp

            if realtime:
                stepduration = currtime - self.stepstarttime
                stepdelta = (self.hoplms / 1000) - stepduration - self.behindcorrection

                if stepdelta < 0:
                    print('FALLEN BEHIND!')
                    self.behindcorrection = stepdelta
                else:
                    time.sleep(stepdelta)
                    self.behindcorrection = 0

                self.stepstarttime = time.time() - self.starttime

    def recordCallback(self, indata, frames, rectime, status):

        currtime = time.time() - self.starttime
        stepduration = currtime - self.stepstarttime
        stepdelta = (self.hoplms / 1000) - stepduration

        # discard previous values
        self.buffer[:] = np.roll(self.buffer, -self.hopl)

        self.buffer[-self.hopl:] = indata[:,0]

        targtimestamp, reftimestamp = self.alignmentFunc(self.buffer)
        self.queueFunc(targtimestamp, reftimestamp)
        ratio = reftimestamp / targtimestamp

    def record(self, starttime=0):
        """
        Start a live recording using the system microphone
        """

        self.starttime = time.time()
        self.stepstarttime = starttime

        self.buffer = np.zeros((self.fl))
        self.stream = sd.InputStream(samplerate=self.fs, blocksize=self.hopl, channels=1, dtype=np.int16, callback=self.recordCallback)
        self.stream.start()

    def stop(self):
        """
        Stop a live recording
        """
        self.stream.stop()