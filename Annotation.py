import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

class Annotation:

    def loadTimeMeasure(self, reference, target, as_array=False, align_start_time=False, delimiter=",", shift=0, ratio=1.0, id_first=False):
        """
            Load alignment of two annotations in timestamp, measure format
            Format of ISMIR19 dataset
            
            paramaters:
            | target: alignment of target recording
            | reference: alignment of reference recording
            | as_array: target and reference are supplied as numpy arrays
            | align_start_time: normalise timestamps to start at 0
            | delimiter: csv file delimiter
            | shift: time (s) to shift reference timestamps by
            | ratio: Playback speed
            | id_first: if the first column contains id instead of time
        """

        # Load annotations
        if as_array:
            self.target = target
            self.reference = reference
        else:
            self.target = np.loadtxt(open(target, "rb"), delimiter=delimiter)
            self.reference = np.loadtxt(open(reference, "rb"), delimiter=delimiter)

        col_t = 0
        col_i = 1
        if id_first:
            col_t = 1
            col_i = 0

        # Check that the measures match
        for i in range(self.target.shape[0]):
            if self.target[i, col_i] != self.reference[i, col_i]:
                raise Exception(f'Annotation measures do not match! index {i}: [{self.target[i, 1]}] != [{self.reference[i, 1]}]')

        if align_start_time:

            # Subtract first timestamp from all
            self.target[:, col_t] = self.target[:, col_t] - self.target[0, col_t]
            self.reference[:, col_t] = self.reference[:, col_t] - self.reference[0,col_t]

        if shift != 0:
            self.reference[:, col_t] = self.reference[:, col_t] - shift

        if ratio != 1.0:
            self.reference[:, col_t] = self.reference[:, col_t] * ratio

        # save measures
        self.measures = self.target[:, col_i]

        # Calculate alignment
        # in t, u pairs
        self.alignment = np.zeros((self.target.shape[0], 2))
        self.alignment[:, 0] = self.target[:, col_t]
        self.alignment[:, 1] = self.reference[:, col_t]

    def getFrameAlignment(self, hopl, start_time=0, end_time=0):

        frameAlignment = self.alignment * 1000 / hopl
        frameAlignment = np.array(frameAlignment, dtype=int)

        if start_time != 0:
            start_frame = int(start_time * 1000 / hopl)
            for i in range(frameAlignment.shape[0]):
                if frameAlignment[i, 0] >= start_frame:
                    frameAlignment = frameAlignment[i:]
                    break
        
        if end_time != 0:
            end_frame = int(end_time * 1000 / hopl)
            for i in range(frameAlignment.shape[0]):
                if frameAlignment[i, 0] > end_frame:
                    frameAlignment = frameAlignment[:i]
                    break
        
        return frameAlignment

    def getAlignmentDelta(self, alignment, hopl, endtime=0):
        """
            The delta of the given alignment to the annotation
            parameters:
            | alignment: 
        """

        if endtime != 0:
            annotated = self.getFrameAlignment(hopl, end_time=endtime)
            l = annotated.shape[0]
        else:
            l = self.alignment.shape[0]
            annotated = self.getFrameAlignment(hopl)

        # Calculate difference
        delta = np.zeros((l))

        maxl = l
        for i in range(l):

            # Stop when we surpass the calculated alignment
            if annotated[i, 0] >= alignment.shape[0] or annotated[i, 1] >= alignment.shape[0]:
                maxl = i
                break

            # Calculate difference between reference index as found the alignment vs the annotation for each target point in the annotation
            delta[i] = alignment[annotated[i, 0]] - annotated[i, 1]
        
        # Normalise to seconds
        delta /= 1000/hopl

        delta = delta[:maxl]
        return delta

    def plotAlignmentDelta(self, alignment, hopl, size=(20,20), endtime=0, label='', color=None, linewidth=2, no_x=False):


        if endtime != 0:
            delta = self.getAlignmentDelta(alignment, hopl, endtime=endtime)
        else:
            delta = self.getAlignmentDelta(alignment, hopl)

        if no_x:
            xvals = np.arange(delta.shape[0])
        else:
            xvals = self.target[:delta.shape[0], 0]

        fig, plot = plt.subplots(1,1, figsize=size)
        plot.plot(xvals, np.zeros((delta.shape[0])), color='black', linewidth=1)
        if label:
            plot.plot(xvals, delta, label=label, color=color, linewidth=linewidth)
        else:
            plot.plot(xvals, delta)
        plot.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plot.grid()

        if label:
            plot.legend()
            
        return fig, plot, xvals