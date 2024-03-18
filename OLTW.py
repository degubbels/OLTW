from wave import Error
import numpy as np
import math
from Feature import LibrosaMFCC

from pathos.multiprocessing import ProcessingPool as Pool

def mptest(a):
    return a

def euclidean(A, B):
    """
        Calculate euclidean distance between feature vectors A and B
    """

    return np.sqrt(np.sum(np.square(A - B)))

def diagTestRatio(i, r, t, u, T, U):
    L = 16

    v = 0
    for l in range(L):
        v += (1/L) * euclidean(U[i][int((u-l) / r)], T[t-l])
    return v

class OLTW:
    """
        Implementation of the online-Dynamic Time Warp algorithm
        Base modeled after (Dixon, 2005): Live tracking of musical performances using on-line time warping
    """

    ## Settings
    # Base parameters
    searchwidth = 400               # Number of frames to search
    maxRunCount = 3                 # Max number of successive steps in one direction
    diagonalcost = 1                # correction for diagonal path length cost
    INC_MEASURE = 'raw'             # What measure to use for finding increment direction: raw | minmean | progminmean | rollingminmean | weightedmean | rollingweightedmean
    USE_FUTURE = True
    D_MEASURE = 'cube'              # Distance measure to use for normalising LCM: manhattan | square | cube
    DELIN = True                    # Delinearisation method: none | diagonal | axes

    # Backtracing
    USE_BACKTRACING = False
    USE_BACKTRACING_MULT = True
    BACKTRACE_EVERY = 8             # How many frames between each backtrace
    BACKTRACE_LENGTH = 50           # How far to backtrace for
    BACKTRACE_SECONDARY_MULT = 5    # By how much to multiply the above parameters for the long backtraces

    START_DEADZONE = 0              # Number of frames to continue diagonal after searchwidth

    ## Running parameters
    t = -1                          # Target index
    u = 0                           # Reference index
    previous = ''                   # 'row'/'col'/'both' - The last increment direction
    runcount = 0                    # The current number of consecutive equal increment directions

    evalt = -1
    evalu = -1

    initialised = False

    ## Data
    Ml = searchwidth                # Cost Matrix size
    M  = None                       # Accumulated cost matrix
    LCM = None                      # Accumulated local cost matrix

    maxTl = 10000                   # Max target length (frames)

    Tl = 0                          # Length of target array
    T = None                        # Target audio feature array

    Ul = 0                          # Length of reference array
    U = None                        # Reference audio feature array

    alignment = None                # Alignment array,  store only for every t, the first corresponding u


    ## Analytics
    SAVE_ANALYTICS = True           # Whether to record and save analytics data
    PCM = None                      # primitive cost matrix

    # Delinearisation options
    DELIN_LEN = 0
    DELIN_TEMPO_DELTA = 0.08
    DELIN_N_TEMPO = 3

    delin_hits = 0
    delin_misses = 0
    DELIN_DIAG_PARAM = 'hop'

    multiProcess = True
    multiProcessingPool = None
    

    def __init__(self, feature, use_backtracing=False, inc_measure='raw', diagonalCost=1, searchwidth=400,
                matrixsize=800, n_frames=8000, save_analytics=False, start_deadzone=0, use_future=False,
                d_measure='cube', delin='none', delin_len=16, delinDiagParam='hop', n_processes=16, multiProcess=False):
        """
            feature: The audio feature to use for cost calculation
            use_backtracing: Should the backtracing steps be performed
            diagonalCost: Cost modifier for diagonal steps
            searchwidth: How long far back from the path to calculate
            matrixsize: How much of the cost matrix to store, must be at least searchwidth
            save_analytics: Should calculation metrics be recorded and saved
        """

        self.feature = feature
        self.USE_BACKTRACING = use_backtracing
        self.INC_MEASURE = inc_measure
        self.diagonalcost = diagonalCost
        self.searchwidth = searchwidth
        self.Ml = matrixsize
        self.maxTl = n_frames
        self.SAVE_ANALYTICS = save_analytics
        self.START_DEADZONE = start_deadzone
        self.USE_FUTURE = use_future
        self.D_MEASURE = d_measure
        self.DELIN = delin
        self.DELIN_LEN = delin_len
        self.DELIN_DIAG_PARAM = delinDiagParam
        self.multiProcessingPool = Pool(n_processes)
        self.multiProcess = multiProcess

    def loadTarget(self, audio):
        """
                Load the target audio feature array from the give audio signal
        """

        self.T = self.feature.range(audio)
        self.Tl = self.T.shape[0]

    def loadReference(self, audio, shift=0, feature=None):
        """
            Load the reference audio feature array from the give audio signal
            shift: shift vector over (only makes sense for chrome, need better solution)
            feature: use alternative feature
        """

        if feature == None:
            feature = self.feature

        if self.DELIN == 'diagonal' and self.DELIN_DIAG_PARAM == 'fs':

            self.delin_features = []
            self.U = []

            for tempo in np.arange(1-(self.DELIN_N_TEMPO*self.DELIN_TEMPO_DELTA), 1+((self.DELIN_N_TEMPO)*self.DELIN_TEMPO_DELTA)+0.01, self.DELIN_TEMPO_DELTA):

                feature = LibrosaMFCC(round(self.feature.fs*tempo, 3), self.feature.flms, self.feature.hoplms, ncoeff=120, cskip=20)
                print("Generating feature for tempo="+str(tempo)+ "="+str(feature.fs))
                self.U.append(feature.range(audio))

            self.Ul = self.U[self.DELIN_N_TEMPO].shape[0]
        elif self.DELIN == 'diagonal' and self.DELIN_DIAG_PARAM == 'hop':

            self.delin_features = []
            self.U = []

            for tempo in np.arange(1-(self.DELIN_N_TEMPO*self.DELIN_TEMPO_DELTA), 1+((self.DELIN_N_TEMPO)*self.DELIN_TEMPO_DELTA)+0.01, self.DELIN_TEMPO_DELTA):

                feature = LibrosaMFCC(self.feature.fs, round(self.feature.flms * tempo, 3), round(self.feature.hoplms * tempo, 3), ncoeff=120, cskip=20)
                print("Generating feature for tempo="+str(tempo)+ "="+str(feature.hopl))
                self.U.append(feature.range(audio))

            self.Ul = self.U[self.DELIN_N_TEMPO].shape[0]
        else:
            self.U = feature.range(audio)

            self.Ul = self.U.shape[0]

        if shift != 0:
            self.U = np.roll(self.U, shift, 1)

    def init(self):
        """
            Initialise algorithm for running.
            Create data structures, including for analytics
        """

        ## Init datastructures
        # store only for every t, the first corresponding u
        self.alignment = np.zeros(self.maxTl, dtype=np.int32)

        # store cost as a loop-around matrix of size searchWidth^2
        # With correct implementation, uncalculated cells should never be acessed, though we'd better not rely on it
        self.M = np.full((self.Ml, self.Ml), 10**20, dtype=float)      # Accumulated cost matrix
        self.LCM = np.full((self.Ml, self.Ml), 10**20, dtype=float)    # Accumulated local cost matrix

        if self.SAVE_ANALYTICS:
            self.PCM = np.full((self.Ml, self.Ml), 10**20, dtype=float)    # primitive cost matrix

            # Keep track of the current cost as a confidence metric
            self.matchCostPrimitive = np.zeros(self.maxTl)
            self.matchCostLocal = np.zeros(self.maxTl)

            self.closeCount= np.zeros((self.maxTl))

            self.backtraces = np.full((self.maxTl, 2, 2), None)
            self.retraces = np.full((self.maxTl, 1000, 2), None)

        if self.DELIN == 'diagonal':
            self.CSM = np.full((self.DELIN_N_TEMPO*2+1, self.Ml, self.Ml), 10**20, dtype=float)
            self.UCM = np.full((self.Ml, self.Ml), 10**20, dtype=float)    # column-wise delinearisation cost matrix
        if self.DELIN == 'axes':
            self.TCM = np.full((self.Ml, self.Ml), 10**20, dtype=float)    # row-wise delinearisation cost matrix
            self.UCM = np.full((self.Ml, self.Ml), 10**20, dtype=float)    # column-wise delinearisation cost matrix

        if self.SAVE_ANALYTICS or self.INC_MEASURE in ['minmean', 'progminmean', 'rollingminmean']:
            # Track all minimum cost points
            self.n_minPoints = 5
            self.minPoints = np.zeros((self.n_minPoints, self.maxTl, 2))
            self.minMean = np.zeros((self.maxTl, 2))

        if self.SAVE_ANALYTICS or self.INC_MEASURE == 'progminmean':
            self.progMinMean = np.zeros((self.maxTl, 2))

        if self.SAVE_ANALYTICS or self.INC_MEASURE == 'rollingminmean':
            self.rollingMinMeanLength = 100
            self.rollingMinMean = np.zeros((self.maxTl, 2))
        
        if self.SAVE_ANALYTICS or self.INC_MEASURE in ['weightedmean', 'rollingweightedmean']:

            self.weightarr = np.zeros((self.maxTl, self.searchwidth*2 + -1 - 2*self.DELIN_LEN, 3))
            self.weightedMean = np.zeros((self.maxTl, 2))

        if self.SAVE_ANALYTICS or self.INC_MEASURE == 'rollingweightedmean':
            self.rollingWeightedMeanLength = 50
            self.rollingWeightedMean = np.zeros((self.maxTl, 2))

        self.T = np.zeros((self.maxTl, self.feature.veclen))
        self.initialised = True


    def simulate(self):
        """
            Base algorithm loop
        """

        if not self.initialised:
            print('not initialised!')
            return

        # Calculate first costs
        self.evalPathCost(self.t, self.u)

        # Stop when either feature array is done
        while self.t < self.Tl - 1 and self.u < self.Ul - 1:
            self.step()

        self.alignment = self.alignment[:self.t]

    def processFrame(self, frame):
        """
            Process a new incoming audio frame
            returns the current reference-equivalent timestamp
        """
        
        # Get feature
        self.Tl +=1
        featureframe = self.feature.apply(frame)
        self.T[self.Tl] = featureframe

        while self.t < self.Tl - 1 and self.u < self.Ul - 1:
            self.step()

        # Process timestamps
        targtimestamp = (self.t+1) / (1000 / self.feature.hoplms)
        reftimestamp = (self.u+1) / (1000 / self.feature.hoplms)

        return (targtimestamp, reftimestamp)

    def step(self):
        """
            Perform one iteration step of the algorithm
        """

        inc = self.getInc(self.t, self.u)

        # Advance row
        if  inc != 'col':

            # Advance target, first corresponding reference index in recorded
            self.t = self.t+1

            self.alignment[self.t] = self.u

            if self.SAVE_ANALYTICS:
                self.matchCostLocal[self.t] = self.LCM[self.t % self.Ml, self.u % self.Ml]
                self.matchCostPrimitive[self.t] = self.PCM[self.t % self.Ml, self.u % self.Ml]

            if self.USE_FUTURE:
                for k in range(max(self.u - self.searchwidth, 0), self.u + self.searchwidth + 1):
                    self.evalPathCost(self.t, k)
            else:
                for k in range(max(self.u - self.searchwidth, 0), self.u + 1):
                    self.evalPathCost(self.t, k)
            
            # Use backtracing
            if self.USE_BACKTRACING and self.t > self.searchwidth:
                if self.t % self.BACKTRACE_EVERY == 0:
                    self.backtrace(self.t, self.u)
                    if self.t >= self.Tl - 1 or self.u >= self.Ul - 1:
                        return
                    
        # Advance column
        if inc != 'row':

            # Advance column
            self.u = self.u + 1
            
            if not self.USE_FUTURE:
                for k in range(max(self.t - self.searchwidth, 0), self.t + 1):
                    self.evalPathCost(k, self.u)

        current = inc

        if current == self.previous:
            self.runcount += 1
        else:
            self.runcount = 1
        
        if current != 'both':
            self.previous = current



    def getInc(self, t, u):
        """
            Find out next direction to increment
        """
        
        # Before reaching searchwidth, go diagonal to avoid inaccurate alignment due to lack of information
        if (t < self.searchwidth + self.START_DEADZONE):
            return 'both'

        # Make a step in the other direction after too many consecutive steps in one direction
        if self.runcount > self.maxRunCount:
            if self.previous == 'row':
                return 'col'
            else:
                return 'row'

        if self.USE_FUTURE:
            
            arr = self.LCM[t, u-self.searchwidth: u + self.searchwidth + 1]
            minpoint = u-self.searchwidth + np.argmin(arr)

            if minpoint <= self.u - 2:
                return 'row'
            if minpoint > self.u + 2:
                return 'col'
            else:
                return 'both'
        else:
            
            sortarr, frameweightarr = self.getFrameArray(t, u, sorted=True, append_unsorted=True)

            if self.SAVE_ANALYTICS or self.INC_MEASURE in ['minmean', 'progminmean', 'rollingminmean']:
                # Collect points for inspection
                for i in range(self.n_minPoints):
                    self.minPoints[i, t] =  [sortarr[i,1], sortarr[i,2]]

                # get meanpoint
                self.minMean[t, 0] = np.mean(self.minPoints[:, t, 0])
                self.minMean[t, 1] = np.mean(self.minPoints[:, t, 1])

            # Find out how many points are close
            if self.SAVE_ANALYTICS:
                self.closeCount[t] = sortarr.shape[0]
                for i in range(sortarr.shape[0]):
                    if sortarr[i,0] > sortarr[0,0]*1.02:
                        self.closeCount[t] = i
                        break

            if self.SAVE_ANALYTICS or self.INC_MEASURE == 'progminmean':
                if self.progMinMean[t-1, 0] == 0:
                    self.progMinMean[t] = self.minMean[t]
                else:
                    self.progMinMean[t, 0] = np.mean([self.minMean[t, 0], self.progMinMean[t-1, 0]])
                    self.progMinMean[t, 1] = np.mean([self.minMean[t, 1], self.progMinMean[t-1, 1]])

            if self.SAVE_ANALYTICS or self.INC_MEASURE == 'rollingminmean':
                if self.minMean[t-self.rollingMinMeanLength, 0] == 0:
                    self.rollingMinMean[t] = self.minMean[t]
                else:
                    self.rollingMinMean[t, 0] = np.mean(self.minMean[t-self.rollingMinMeanLength:t+1, 0])
                    self.rollingMinMean[t, 1] = np.mean(self.minMean[t-self.rollingMinMeanLength:t+1, 1])

            if self.SAVE_ANALYTICS or self.INC_MEASURE in ['weightedmean', 'rollingweightedmean']:
                self.weightarr[t] = frameweightarr
                index = int(np.average(np.arange(self.weightarr.shape[1]), 0, (np.max(self.weightarr[t,:,0])-self.weightarr[t,:,0].T)**8))
                self.weightedMean[t, 0] = self.weightarr[t,index,1]
                self.weightedMean[t, 1] = self.weightarr[t,index,2]

            if self.SAVE_ANALYTICS or self.INC_MEASURE == 'rollingweightedmean':
                if self.weightedMean[t-self.rollingWeightedMeanLength, 0] == 0:
                    self.rollingWeightedMean[t] = self.weightedMean[t]
                else:
                    self.rollingWeightedMean[t, 0] = np.mean(self.weightedMean[t-self.rollingWeightedMeanLength:t+1, 0])
                    self.rollingWeightedMean[t, 1] = np.mean(self.weightedMean[t-self.rollingWeightedMeanLength:t+1, 1])
                

            measurepos = None
            if self.INC_MEASURE == 'raw':
                measurepos = sortarr[0, 1:]
            elif self.INC_MEASURE == 'minmean':
                measurepos = self.minMean[t]
            elif self.INC_MEASURE == 'progminmean':
                measurepos = self.progMinMean[t]
            elif self.INC_MEASURE == 'rollingminmean':
                measurepos = self.rollingMinMean[t]
            elif self.INC_MEASURE == 'weightedmean':
                measurepos = self.weightedMean[t]
            elif self.INC_MEASURE == 'rollingweightedmean':
                measurepos = self.rollingWeightedMean[t]
                
            # Calculate direction
            tdelta = t - measurepos[0]
            udelta = u - measurepos[1]

            # Find direction compared to current
            if abs(tdelta - udelta) < 1:
                return 'both'
            elif tdelta > udelta:
                # closer to u
                return 'col'
            else:
                return 'row'

    def backtrace(self, t, u):
        """
            Perform backtracing procedure
        """

        # Every so often, perform a longer backtrace
        if self.USE_BACKTRACING_MULT and (self.t % (self.BACKTRACE_EVERY * self.BACKTRACE_SECONDARY_MULT)) == 0:
            bt, bu = self.traceBackPath(self.t, self.u, self.BACKTRACE_LENGTH * self.BACKTRACE_SECONDARY_MULT)
        else:
            bt, bu = self.traceBackPath(self.t, self.u, self.BACKTRACE_LENGTH)        
        
        # Retrace back to current target frame
        self.u = self.retrace(bt, bu)

    def traceBackPath(self, t, u, l):
        """
            t, u: Start for backtrace (end of path)
            l: Number of backtracing steps

            returns:
            t: new t
            u: new u
        """

        t_start = t

        if self.SAVE_ANALYTICS:
            self.backtraces[t_start, 0] = [t, u]

        for i in range(l):

            if t < 2 or u < 2:
                break

            # Find mincost predecessor
            costa = self.LCM[(t-1) % self.Ml, (u-1) % self.Ml]
            costb = self.LCM[(t-1) % self.Ml, u % self.Ml]
            costc = self.LCM[t % self.Ml, (u-1) % self.Ml]

            if costb < costa:
                costab = costb
                mincost = 'b'
            else:
                costab = costa
                mincost = 'a'
            if costc < costab:
                cost = costc
                mincost = 'c'
            else:
                cost = costab

            if mincost == 'a':
                t -= 1
                u -= 1
            elif mincost == 'b':
                t -= 1
            else:
                u -= 1

        if self.SAVE_ANALYTICS:
            self.backtraces[t_start, 1] = [t, u]

        return t, u

    def retrace(self, bt, bu):
        """
            Retrace back to current frame after backtrace
            returns new u
        """

        # Count consecutive steps in the same direction
        prevdir = ''
        dircount = 0

        if self.SAVE_ANALYTICS:
            stepCounter = 0
            self.retraces[self.t, 0] = [bt, bu]

        # Forward back up to current t
        while bt < self.t:

            inc = self.getInc(bt, bu)

            # Track maxruncount (separate from main loop)
            if inc == prevdir:
                dircount += 1
                if dircount > self.maxRunCount:
                    dircount = 0
                    inc = 'both'
            else:
                dircount = 0

            if inc == 'row':
                bt +=1

            if inc == 'col':
                bu +=1

            if inc == 'both':
                bt += 1
                bu += 1

            prevdir = inc

            if bu > self.u:
                self.u += 1

                for k in range(max(self.t - self.searchwidth, 0), self.t + 1):
                    self.evalPathCost(k, self.u)

            if self.SAVE_ANALYTICS:
                stepCounter += 1
                self.retraces[self.t, stepCounter] = [bt, bu]

        return bu

    def evalPathCost(self, t, u):
        """
            Calculate costs for cell
        """

        if self.evalt > t and self.evalu > u:
            pass
            # print(f'DOUBLE CALCULATION!!!! at  t={t}<{self.evalt}, u={u}<{self.evalu}')

        cost = 0            

        if self.DELIN == 'diagonal':

            if self.multiProcess:

                L = 16

                if t > 2*L and u > 2*L:

                    ratios = np.arange(1-(self.DELIN_N_TEMPO*self.DELIN_TEMPO_DELTA), 1+((self.DELIN_N_TEMPO)*self.DELIN_TEMPO_DELTA)+0.01, self.DELIN_TEMPO_DELTA)
                    n_ratios = ratios.shape[0]

                    values = self.multiProcessingPool.map(diagTestRatio, np.arange(n_ratios), ratios, [t]*n_ratios, [u]*n_ratios, [self.T]*n_ratios, self.U)

                    cost = np.min(values)
                    self.UCM[t % self.Ml, u % self.Ml] = cost
                else:
                    cost = 1000
                    self.PCM[t % self.Ml, u % self.Ml] = cost

                    if t == u:
                        cost = 0
            else:
                L = 16
                
                if t > 2*L and u > 2*L:

                    vmin = 10*100
                    for i, tempo in enumerate(np.arange(1-(self.DELIN_N_TEMPO*self.DELIN_TEMPO_DELTA), 1+((self.DELIN_N_TEMPO)*self.DELIN_TEMPO_DELTA)+0.01, self.DELIN_TEMPO_DELTA)):

                        v = 0
                        for l in range(L):
                            v += (1/L) * self.d(self.U[i][int((u-l) / tempo)], self.T[t-l])
                        if v < vmin:
                            vmin = v
                
                    # cost = np.min(dL)
                    cost = vmin
                    self.UCM[t % self.Ml, u % self.Ml] = cost
                    self.PCM[t % self.Ml, u % self.Ml] = self.CSM[self.DELIN_N_TEMPO, t % self.Ml, u % self.Ml]

                else:
                    cost = 1000
                    self.PCM[t % self.Ml, u % self.Ml] = cost

                    if t == u:
                        cost = 0
        elif self.DELIN == 'axes':
            L = 16
            # Subtract previous values to reduce linear structures
            if t > L and u > L:

                cost = self.d(self.T[t], self.U[u])
            
                self.PCM[t % self.Ml, u % self.Ml] = cost

                costt = cost - 0.1  * self.PCM[(t-1) % self.Ml, u % self.Ml]\
                            - 0.25 * self.PCM[(t-2) % self.Ml, u % self.Ml]\
                            - 0.2  * self.PCM[(t-4) % self.Ml, u % self.Ml]\
                            - 0.15 * self.PCM[(t-8) % self.Ml, u % self.Ml]\
                            - 0.1  * self.PCM[(t-16) % self.Ml, u % self.Ml]\
                                +100
                self.TCM[t % self.Ml, u % self.Ml] = costt

                costu = costt - 0.1 * self.TCM[t % self.Ml, (u-1) % self.Ml]\
                            - 0.25 * self.TCM[t % self.Ml, (u-2) % self.Ml]\
                            - 0.2  * self.TCM[t % self.Ml, (u-4) % self.Ml]\
                            - 0.15 * self.TCM[t % self.Ml, (u-8) % self.Ml]\
                            - 0.1  * self.TCM[t % self.Ml, (u-16) % self.Ml]\
                                + 100

                cost = (costu**2) / 100

                self.UCM[t % self.Ml, u % self.Ml] = cost

                if (t<40 or u<40):
                    if t==u:
                        cost = 0
                    else:
                        cost = 1000


            else:
                cost = 1000
                self.PCM[t % self.Ml, u % self.Ml] = cost
                self.TCM[t % self.Ml, u % self.Ml] = cost
                self.UCM[t % self.Ml, u % self.Ml] = cost

                if t==u:
                    cost = 0

        else:
            cost = self.d(self.T[t], self.U[u])

            if self.SAVE_ANALYTICS:
                self.PCM[t % self.Ml, u % self.Ml] = cost

        if t == 0 and u == 0:
            self.M[t % self.Ml, u % self.Ml] = cost
        elif u < 1:
            self.M[t % self.Ml, u % self.Ml] = cost + self.M[(t-1) % self.Ml, u % self.Ml]
        elif t < 1:
            self.M[t % self.Ml, u % self.Ml] = cost + self.M[t % self.Ml, (u-1) % self.Ml]
        else:
            self.M[t % self.Ml, u % self.Ml] = cost + min(self.M[(t-1) % self.Ml, u % self.Ml],
                              self.M[t % self.Ml, (u-1) % self.Ml],
                              self.diagonalcost * self.M[(t-1) % self.Ml, (u-1) % self.Ml])


        # Calculate length-normalised cost
        if self.D_MEASURE == 'manhattan':
            self.LCM[t % self.Ml, u % self.Ml] = self.M[t % self.Ml, u % self.Ml] / max((t + u), 1)
        elif self.D_MEASURE == 'square':
            self.LCM[t % self.Ml, u % self.Ml] = self.M[t % self.Ml, u % self.Ml] / max(math.sqrt(t**2 + u**2), 1)
        elif self.D_MEASURE == 'cube':
            self.LCM[t % self.Ml, u % self.Ml] = self.M[t % self.Ml, u % self.Ml] / max((t**3 + u**3)**(1/3), 1)
        elif self.D_MEASURE == 'sqrt':
            self.LCM[t % self.Ml, u % self.Ml] = self.M[t % self.Ml, u % self.Ml] / max((math.sqrt(t) + math.sqrt(u))**2, 1)
        else:
            raise Error("uknown distance measure")

        if t > self.evalt:
            self.evalt = t
        if u > self.evalu:
            self.evalu = u

    def d(self, A, B):
        """
            Calculate cost between features
        """

        return self.euclidean(A, B)

    def euclidean(self, A, B):
        """
            Calculate euclidean distance between feature vectors A and B
        """

        return np.sqrt(np.sum(np.square(A - B)))

    def getFrameArray(self, t, u, sorted=True, append_unsorted=False):
        """
            Collect a linearised array of cells for the current frame position
            sorted: Should the array be sorted before returning
            append_unsorted: If sorted is true; should the unsorted array be return as second output
        """

        ## Collect all values to check
        # Most recent reference
        arr = np.array([self.LCM.take(u, axis=1, mode='wrap').take(range(t-self.searchwidth+1+self.DELIN_LEN, t), mode='wrap'),
                        np.arange(t - self.searchwidth+1+self.DELIN_LEN, t),
                        np.full((self.searchwidth - 1 - self.DELIN_LEN), u)]).transpose()
        # Current position
        arr = np.append(arr, np.array([[self.LCM[t % self.Ml, u % self.Ml], t, u]]), axis=0)
        # Most recent target
        arr = np.append(arr, np.array([self.LCM.take(t, axis=0, mode='wrap').take(range(u-1, u-self.searchwidth+self.DELIN_LEN, -1), mode='wrap'),
                        np.full((self.searchwidth - 1 - self.DELIN_LEN), t),
                        np.arange(u-1, u-self.searchwidth + self.DELIN_LEN, -1)]).transpose(), axis=0)

        if sorted:
            # sort by first column to find lowest cost, second and third column represent matrix coords
            sortarr = arr[arr[:, 0].argsort()]

            if append_unsorted:
                return sortarr, arr
            else:
                return sortarr

        return arr

    def getZeroLCM(self):
        """
            Provide a plottable copy of the  local cost matrix with zero background
        """

        tlcm = np.copy(self.LCM)
        # Set unfilled fields to zero
        tlcm[tlcm > 10**10] = None
        return tlcm

    def getAlignment(self):
        return self.alignment[:self.t]

if __name__ == '__main__':
    from pathos.helpers import freeze_support
    freeze_support() # help Windows use multiprocessing
    OLTW()