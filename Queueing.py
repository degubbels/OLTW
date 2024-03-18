import numpy as np
import time

# Tracks progress of timestamps passed in the reference
class Queueing:

    points = None      # Queueing points (id, timestamp ())
    subs = None

    performanceId = 0

    currentIndex = 0

    connector = None

    def __init__(self, connector=None):
        if connector:
            self.connector = connector

    def loadCSV(self, path, startoffset=0):
        """
            Load queue data from csv file with (id, timestamp) column format
        """

        self.points = np.loadtxt(open(path, "rb"), delimiter=',')

        # Apply start offset
        if startoffset != 0:
            n_skip = 0
            for i in range(len(self.points)):
                if self.points[i, 1] < startoffset:
                    n_skip +=1
                else:
                    self.points[i, 1] -= startoffset
            self.points = self.points[n_skip:]


    def saveCSV(self, path):
        """
            Save queue data to csv file
        """

        np.savetxt(open(path, 'wb'), self.points, delimiter=',',fmt='%.0f,%.3f')

    def loadSRT(self, path, startoffset=0):
        """
            Load queue data from srt subtitle file into (id, timestamp, line) format
        """

        with open(path, 'r') as file:
            lines = file.readlines()

            # Strip newlines
            for i in range(len(lines)):
                lines[i] = lines[i].rstrip()

            # Maximum number of subtitle lines
            n_max = int(len(lines) / 4)

            self.points = np.zeros((n_max, 2))
            self.subs = [""]

            i = 0
            # For every subtitle line
            for j in range(n_max):

                if i >= len(lines):
                        break
                
                # Get id
                self.points[j, 0] = lines[i]
                i += 1

                # Get start time (end time is discarded)
                starttime, _ = lines[i].split(" --> ")
                h, m, sms = starttime.split(':')
                s, ms = sms.split(',')
                self.points[j, 1] = (3600 * int(h) + 60 * int(m) + int(s) + (int(ms) / 1000)) - startoffset
                i += 1

                # Get subtitle lines
                sub = lines[i]
                i += 1
                if i >= len(lines):
                        break

                while len(lines[i]) > 0:
                    sub += ' ' + lines[i]
                    i += 1
                    if i >= len(lines):
                        break
                self.subs.append(sub)

                # skip empty line(s)
                while len(lines[i]) == 0:
                    i += 1
                    if i >= len(lines):
                        break

            # remove empty
            self.points = self.points[:j]


    def getNext(self):
        """
            Return next queueing timestamp
        """

        return self.points[self.currentIndex, 1]

    def checkTimestamp(self, timestamp):
        """
            Check if a new queueing point has been passed
        """


        if not(self.currentIndex >= len(self.points)):
            if timestamp >= self.points[self.currentIndex, 1]:
                self.advance()

        if self.currentIndex >= 1:
            return int(self.points[self.currentIndex-1, 0])
        else:
            return 0

    def advance(self):
        """
            Send next queueing point
            timestamp: current live reference timestamp
        """

        self.currentIndex += 1

        self.sendQueueRequest()

    def sendQueueRequest(self):
        """
            Send next action to the connected service
        """
        if self.connector:
            self.connector.SendNext()