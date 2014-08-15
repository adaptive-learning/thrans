from sklearn.metrics import roc_curve, auc
import math
import numpy as np
import pylab as plt
# copy from simulated-experiments... TODO unify...

class Evaluator:
    LL_BOUND = 0.01 # trochu hack, vyhodnotit?

    def __init__(self, log):
        self.log = log
        self.n = len(log)
        sse, llsum = 0.0, 0.0
        for pred, answer in log:
            sse += (pred - answer) ** 2
            if answer: llsum += math.log(max(pred, Evaluator.LL_BOUND))
            else:      llsum += math.log(max(1-pred, Evaluator.LL_BOUND))
        self.ll = llsum
        self.rmse = math.sqrt(sse / self.n)
        estimated, correct = zip(*log)
        fpr, tpr, thresholds = roc_curve(correct, estimated)
        self.auc = auc(fpr, tpr)
        self.compute_bins_stats()
        self.compute_brier_decomposition()
    
    def print_metrics(self, digits = 6):
        print "AUC\t", round(self.auc, digits)
        print "LL\t", int(self.ll)
        print "RMSE\t", round(self.rmse, digits)
        print "Brier reliability\t", round(self.reliability, digits)
        print "Brier resolution\t", round(self.resolution, digits)
        print "Brier uncertainty\t", round(self.uncertainty, digits)
#        brier = self.reliability - self.resolution + self.uncertainty
#        print "Brier test:", brier, self.rmse**2, brier-self.rmse**2

        return round(self.rmse, digits)
        
    def compute_bins_stats(self, k = 10):
        sum_answer, sum_pred = np.zeros(k), np.zeros(k)
        self.count = np.zeros(k)
        for pred, answer in self.log:
            if pred >= 1:   # hack
                pred = 0.999
            b = int(pred * k) # find bin
            sum_answer[b] += answer
            sum_pred[b] += pred
            self.count[b] += 1
        self.mean_obs = sum(sum_answer) / self.n
        self.freq = np.zeros(k)
        self.bin_mean = np.zeros(k)
        for i in range(k):
            if self.count[i]:
                self.freq[i] = sum_answer[i] / self.count[i]
                self.bin_mean[i] = sum_pred[i] / self.count[i]
            else:
                self.bin_mean[i] = (i+0.5) * 1.0 / k

    #assumes compute_bins_stats was called previously
    def compute_brier_decomposition(self):
        rel, res = 0, 0
        for i in range(len(self.count)):
            rel += self.count[i] * (self.freq[i] - self.bin_mean[i])**2
            res += self.count[i] * (self.freq[i] - self.mean_obs)**2
        self.reliability = rel / self.n
        self.resolution = res / self.n
        self.uncertainty = self.mean_obs*(1-self.mean_obs)
            
    def calibration_graphs(self):
        plt.figure()
        plt.plot(self.bin_mean, self.freq)
        plt.plot((0,1), (0,1))
        plt.figure()
        plt.bar(range(len(self.count)), self.count, 0.2)
