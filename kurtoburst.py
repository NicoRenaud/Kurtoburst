import numpy as np
import pandas

from scipy import signal
from scipy.stats import kurtosis

from statsmodels.robust.scale import mad

import pywt
import pybursts
import matplotlib.pyplot as plt

class Kurtoburst(object):
    """ Class to detect the peak using the kurtoburst method."""

    def __init__(self,filename):

        self.filename = filename
        self.raw_data = pandas.read_csv(self.filename)
        #self.colname = ' AU12_r'
        #sself.colname = ' AU06_r'
        #self.colname = ' AU07_r'
        self.colname = ' AU02_r'
        self.time = np.array(self.raw_data[' timestamp'][::10])
        self.input = np.array(self.raw_data[self.colname][::10])
        self.len = len(self.input)

        self.nwin = 51
        self.wave_type = 'sym3'
        self.TK = 0.5
        self.TT = 0.15

        self.burst_s = 2             # burst s parameter
        self.burst_gamma = 0.05   # burst gamma parameters

    def remove_baseline(self):
        """Remove the base line using a Savitzky-Golay method"""

        print(" \t Apply Savitzky-Golay filter \t %d" %self.nwin)
        base_savgol = signal.savgol_filter(self.input, self.nwin, 1)
        self.input_nobase = self.input - base_savgol

    def denoise(self):
        """denoise the data using the 2stage kurtosis denoising"""

        #make sure the data has a len dividible by 2^2
        self.len_swt = self.len
        while not (self.len_swt/4).is_integer():
            self.len_swt -= 1

        inp = self.input_nobase[:self.len_swt]
        self.wave = pywt.Wavelet(self.wave_type)
        nLevel = pywt.swt_max_level(self.len_swt)
        self.coeffs = pywt.swt(inp,self.wave,level=2)

        print(" \t Denoise STW coefficients \t %1.2f %1.2f" %(self.TK,self.TT))
        (cA2, cD2), (cA1, cD1) = self.coeffs

        # rolling kurtosis
        k2 = self._rolling_kts(cD2,self.nwin)
        k1 = self._rolling_kts(cD1,self.nwin)

        # thresholding
        cD2[k2<self.TK] = 0
        cD1[k1<self.TK] = 0

        cA2[k2<self.TK] = 0
        cA1[k1<self.TK] = 0

        # universal threshold
        sigma_roll_1 = mad(cD1[cD1!=0])*np.ones(self.len_swt)
        uthresh_roll_1 = self.TT * sigma_roll_1 * np.sqrt(2*np.log(self.len_swt))
        cD1[abs(cD1)<uthresh_roll_1] = 0

        # universal threshold
        sigma_roll_2 = mad(cD2[cD2!=0])*np.ones(self.len_swt)
        uthresh_roll_2 = self.TT * sigma_roll_2 * np.sqrt(2*np.log(self.len_swt))
        cD2[abs(cD2)<uthresh_roll_2] = 0

        # final threshold
        cA1[cD1 == 0] = 0
        cA2[cD2 == 0] = 0
        self.denoised_coeffs = [(cA1,cD1),(cA2,cD2)]

        # denoise the data
        #self.input_denoised = self._iswt(self.denoised_coeffs,self.wave)
        self.input_denoised = pywt.iswt(self.denoised_coeffs,self.wave)

    def get_burst(self):
        """Detect bursts of activity."""

        print('\t Detect bursts \t\t\t %d %1.2f' %(self.burst_s,self.burst_gamma))

        # compute the cum sum of the positive values of the datan ...
        _tmp = np.copy(self.input_denoised)
        _tmp[_tmp<0] = 0
        _tmp += 1E-12
        self.input_cummulative = np.cumsum(_tmp)

        # decimation ...
        self.T_cummulative = np.copy(self.time[0:-1:10])
        self.input_cummulative = self.input_cummulative[0:-1:10]

        # burst calculation
        self.burst = pybursts.kleinberg(self.input_cummulative,s=int(self.burst_s),gamma=self.burst_gamma)


        Tbursts = []
        for b in self.burst:
            if b[0] == 1:
                ti = self.T_cummulative[np.argwhere(self.input_cummulative==b[1])[0]]
                tf = self.T_cummulative[np.argwhere(self.input_cummulative==b[2])[0]]
                Tbursts.append([ti[0],tf[0]])

        ########################################
        ##   detect the peaks
        ########################################
        x_peak_bursts = []
        y_peak_bursts = []
        print(Tbursts)
        if len(Tbursts)>0:
            for i in range(len(Tbursts)-1):
                ind_init = np.argmin(abs(self.time-Tbursts[i][1]))
                ind_final = np.argmin(abs(self.time-Tbursts[i+1][0]))

                x_peak_bursts.append( self.time[ ind_init + np.argmax(self.input_denoised[ind_init:ind_final])] )
                y_peak_bursts.append(  self.input[ind_init + np.argmax(self.input_denoised[ind_init:ind_final])] )
        else:
            print('\t no peaks found in the bursts')

        self.xpeak = x_peak_bursts
        self.ypeak = y_peak_bursts



    @staticmethod
    def _rolling_kts(y,N):
        """Compute the rolling kurtosis."""

        # number of points
        nPTS,N2 = len(y), int(N/2)

        # define the out
        kts = np.zeros(nPTS)

        # for all points comopute snr
        for i in range(nPTS):
            s,e = i-N2, i+N2
            if s<0:
                s = 0
            if s > nPTS-1:
                s = nPTS-1
            win = np.ones(len(y[s:e]))
            kts[i] = kurtosis(win*y[s:e])
        return kts

    def plot(self):
        plt.plot(self.time,self.input)
        #plt.plot(self.time,self.input_nobase-1,linewidth=0.5)
        plt.scatter(self.xpeak,self.ypeak,c='orange')
        ypeak = np.zeros_like(self.time)
        for p in self.xpeak:
            ypeak[self.time==p] = 0.5
        plt.plot(self.time,ypeak-1,c='orange')
        #plt.plot(self.T_cummulative,self.input_cummulative)
        #plt.plot(self.time[:self.len_swt],self.input_denoised,c='black')
        plt.show()




if __name__ == '__main__':
    filename = '003_VL.csv'
    kb = Kurtoburst(filename)
    kb.remove_baseline()
    kb.denoise()
    tb = kb.get_burst()
    kb.plot()
    print(tb)