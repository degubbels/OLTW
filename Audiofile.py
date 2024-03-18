import wave
import sounddevice as sd
import soundfile as sf
import numpy as np

# Represents a saved audio recording
class Audiofile:

    def __init__(self, path, fs):
        self.signal = self.loadWav(path)
        self.fs = fs

    def loadWav(self, path):
        """
            Load WAV file
            returns numpy array of audio signal
        """
        data, _ = sf.read(path)

        # Merge channels if present
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data[:, 0] + data[:, 1] / 2

        return data

    def play(self):
        sd.play(self.signal, self.fs)

    def section(self, start, end=0):

        if end == 0:
            return self.signal[int(start * self.fs) :]
        else:
            return self.signal[int(start * self.fs) : int(end * self.fs)]