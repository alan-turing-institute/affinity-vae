import numpy as np 
import math


class KLD_weight:
    """
    Parameters
    ----------
    """

    def __init__(self, n_epoch, kl_weight_method, start= 0.0, stop= 1.0, n_cycle= 4, ratio= 0.5):

        self.n_epoch = n_epoch
        self.start = start
        self.stop = stop
        self.n_cycle = n_cycle
        self.ratio = ratio


        print(kl_weight_method, n_epoch, start,stop,n_cycle,ratio)


        if kl_weight_method == 'flat':
            self.beta = self._frange_flat()

        elif kl_weight_method == 'cycle_linear':
            self.beta = self._frange_cycle_linear()

        elif kl_weight_method == 'cycle_sigmoid':
            print('yes')
            self.beta = self._frange_cycle_sigmoid()            
        
        elif kl_weight_method == 'cycle_cosine':
            self.beta = self._frange_cycle_cosine()   
        
        elif kl_weight_method == 'ramp':
            self.beta = self._frange_flat() 

        else: 
            raise RuntimeError("Select a valid method for KL_weight. Available options are : flat, cycle_linear, cycle_sigmoid, cycle_cosine, ramp")



    def _frange_flat(self):
        L = np.ones( self.n_epoch) * self.stop
        return L

    def _frange_cycle_linear(self):
        L = np.ones(self.n_epoch)
        period = self.n_epoch/self.n_cycle
        step = (self.stop-self.start)/(period*self.ratio) # linear schedule

        for c in range(self.n_cycle):

            v , i = self.start , 0
            while v <= self.stop and (int(i+c*period) < self.n_epoch):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L    


    def _frange_cycle_sigmoid(self):
        L = np.ones(self.n_epoch)
        period = self.n_epoch/self.n_cycle
        step = (self.stop-self.start)/(period*self.ratio) # step is in [0,1]
        
        # transform into [-6, 6] for plots: v*12.-6.

        for c in range(self.n_cycle):

            v , i = self.start , 0
            while v <= self.stop:
                L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
                v += step
                i += 1
        return L    


    #  function  = 1 − cos(a), where a scans from 0 to pi/2

    def _frange_cycle_cosine(self):
        L = np.ones(self.n_epoch)
        period = self.n_epoch/self.n_cycle
        step = (self.stop-self.start)/(period*self.ratio) # step is in [0,1]
        
        # transform into [0, pi] for plots: 

        for c in range(self.n_cycle):

            v , i = self.start , 0
            while v <= self.stop:
                L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
                v += step
                i += 1
        return L    

    def _frange_ramp(self):
        L = np.ones(self.n_epoch)
        v , i = self.start , 0
        while v <= self.stop:
            L[i] = v
            v += step
            i += 1
        return L




