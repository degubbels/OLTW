import numpy as np
import scipy as scp
from scipy import fft

import librosa
from librosa import feature

# Various audio features

class Feature:

    veclen = 0
    tune_a = 440

    def __init__(self, fs, fl, hopl):
        """
            fs: frequency
            fl: frame length (ms)
            hopl: hop length (ms)
        """

        self.fs = fs
        self.flms = fl                       # Frame size (ms)
        self.flf = 1000 / self.flms          # Frame frequency (Hz)
        self.fl = int(fs / self.flf)    # Frame length (number of array indices)

        self.hoplms = hopl
        self.hoplf = 1000 / self.hoplms
        self.hopl = int(fs / self.hoplf)

        self.hammingWindow = self.hamming(self.fl)

    def apply(self, sig):
        print('Not implemented')
        pass

    def hamming(self, N):
        """
            Calculate hamming window of length N
        """

        n = np.arange(N)
        a0 = 25/46
        window = a0 - ((1-a0) * np.cos((2 * np.pi * n) / N))
        return window

    def range(self, sig):
        """
            Apply feature to a range of frames
        """

        n_frames = int(sig.shape[0] / self.hopl)

        r = np.zeros((n_frames, self.veclen))

        for i in range(n_frames):
            if i * self.hopl + self.fl <= sig.shape[0]:
                r[i] = self.apply(sig[i * self.hopl : i * self.hopl + self.fl])

        return r


# Naive semitone-spaced onset feature
class Semitone(Feature):

    low = 0
    high = 108
    veclen = high - low

    def __init__(self, fs, fl, hopl, low=0, high=108):
        """
            fs: Frequency
            fl: Frame length (ms)
            hopl: Hop length (ms)
            low: Lowest semitone 
            high: Highest semitone
            semitones are measured as above c0
        """

        super().__init__(fs, fl, hopl)

        self.low = low
        self.high = high
        self.veclen = high - low

        # Pre-calculate the bin boundaries
        self.bins = self.calculateBoundaries()

    def apply(self, sig):

        # Restrict to one frame long
        sig = sig[:self.fl]

        # Take Fourier transform to get spectrum
        s = np.abs(np.fft.fft(sig * self.hammingWindow))[:int(self.fl / 2)]

        # Output feature vector
        fv = np.zeros(self.veclen)

        for i in range(self.veclen - 1):
            fv[i] = np.sum(s[self.bins[i] : self.bins[i + 1]])
        fv[self.veclen - 1] = np.sum(s[self.bins[self.veclen - 1:]])

        return fv

    def calculateBoundaries(self, return_freqs=False):
        """
            Calculates the semitone bin boundaries
            Every index gives the starting frequency of said bin
        """

        # The initial bin always starts a 0 frequency
        bins = np.zeros(self.veclen, dtype=int)
        freqs = np.zeros(self.veclen)

        for i in range(1, self.veclen):

            # Starting frequency for bin i above low
            frequency = self.tune_a * 2**((i + self.low - 57.5) / 12)
            freqs[i] = frequency

            # Fit to spectrum scale
            bins[i] = int(frequency / self.flf)

        if return_freqs:
            return bins, freqs
        else:
            return bins

class SemitoneOnset(Semitone):

    def __init__(self, fs, fl, hopl, low=0, high=108):
        super().__init__(fs, fl, hopl, low=low, high=high)

    def range(self, sig):

        """
            Apply feature to a range of frames
        """

        n_frames = int(sig.shape[0] / self.hopl)

        r = np.zeros((n_frames, self.veclen))

        for i in range(n_frames):
            if i * self.hopl + self.fl <= sig.shape[0]:

                current = super().apply(sig[i * self.hopl : i * self.hopl + self.fl])
                if i > 0:
                    # Get energy increase (ignoring negative energy delta)
                    delta = current - previous
                    r[i] = np.where(delta >= 0, delta, 0)
                else:
                    r[i] = current

                previous = current

        return r

class ChromaOnset(SemitoneOnset):
    
    octaves = 9

    def __init__(self, fs, fl, hopl, octaves=9):

        self.octaves = octaves

        super().__init__(fs, fl, hopl, low=0, high=self.octaves*12)

    def range(self, sig):
        semitoneOnsets =  super().range(sig)

        rv = np.zeros((semitoneOnsets.shape[0], 12))
        for f in range(semitoneOnsets.shape[0]):
            for i in range(12 * self.octaves):
                rv[f, i%12] = rv[f, i%12] + semitoneOnsets[f, i]

        return rv

# Naive Chroma feature
class Chroma(Feature):

    veclen = 12
    octaves = 9

    def __init__(self, fs, fl, hopl, octaves=9):

        super().__init__(fs, fl, hopl)

        self.octaves = octaves

        # Initialise semitone feature for bin calculation
        self.semitone = Semitone(fs, fl, hopl, 0, self.octaves*self.veclen)

    def apply(self, sig):

        semitones = self.semitone.apply(sig)

        fv = np.zeros(self.veclen)
        for i in range(self.veclen * self.octaves):
            fv[i%12] = fv[i%12] + semitones[i]

        return fv

# Mel Frequency Cepstral Coefficients
# After Logan, 2000
class MFCC(Feature):

    veclen = 120

    def __init__(self, fs, fl, hopl, c=20, cskip=0):
        """
            c: how many coefficients
            cskip: how many coefficients to skip
        """
        super().__init__(fs, fl, hopl)
        self.c = c
        self.cskip = cskip
        # Pre-calculate the bin boundaries
        self.bins = self.calcMelBins()

    def apply(self, sig):

        # Restrict to one frame long
        sig = sig[:self.fl]

        # Take Fourier transform to get spectrum
        s = np.abs(np.fft.fft(sig * self.hammingWindow))[:int(self.fl / 2)]

        # Log of amplitude spectrum
        log_s = np.log(s)

        # Use overlapping windows
        fbank = np.zeros((self.veclen, self.fl))

        for k in range(1, self.veclen-2):
            for i in range(self.bins[k-1], self.bins[k]+1):
                fbank[k, i] = ((i - self.bins[k-1] + 1) / (self.bins[k] - self.bins[k-1] + 1)) * log_s[i]
            for i in range(self.bins[k]+1, self.bins[k+1]+1):
                fbank[k, i] = (1 - ((i - self.bins[k]) /  (self.bins[k+1] - self.bins[k] + 1))) * log_s[i]

        fbanksum = np.zeros((self.veclen))
        for k in range(0, self.veclen):
            fbanksum[k] = np.sum(fbank[k])

        # Discrete cosine transform
        return scp.fft.dct(np.abs(fbanksum))

    def calcMelBins(self):

        max_freq = self.fs / 2
        max_mel = self.melFromFreq(max_freq)

        bins = np.zeros(self.veclen, dtype=int)

        for i, m in enumerate(np.linspace(0, max_mel, self.veclen)):

            if m != 0:
                    
                f = self.freqFromMel(m)
                # Spectrum frequency index
                bins[i] = int(f / self.flf)

        return bins[:self.veclen-1]

    # Mel formula for frequency f (from en.wikipedia.org/wiki/Mel_scale)
    def melFromFreq(self, f):
        return 2595 * np.log10(1 + (f / 700))
    
    # Inverse mel formula
    def freqFromMel(self, m):
        return 700 * (10**(m/2595) - 1)

class LibrosaChroma(Feature):

    def range(self, sig):

        return np.transpose(librosa.feature.chroma_stft(sig, self.fs, n_fft=self.fl, hop_length=self.hopl)) * 1000

# Efficient MFCC Implementation from Librosa: McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.
class LibrosaMFCC(Feature):

    veclen = 20

    def __init__(self, fs, fl, hopl, ncoeff=20, cskip=0):
        """
            c: how many coefficients
            cskip: how many coefficients to skip
        """
        super().__init__(fs, fl, hopl)
        self.ncoeff = ncoeff
        self.cskip = cskip
        self.veclen = ncoeff - cskip

    def apply(self, sig):

        # Restrict to one frame long
        sig = sig[:self.fl]

        return self.range(sig)[-1]

    def range(self, sig):

        return np.transpose(librosa.feature.mfcc(y=sig, sr=self.fs, n_mfcc=self.ncoeff, n_fft=self.fl, hop_length=self.hopl, )[self.cskip:])
