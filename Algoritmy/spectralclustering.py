##
#Spectral clustering0
##

from kmean import *
import numpy as np
import numpy.linalg as la
import random
import math
import pylab as plt
import scipy.stats as stats
import sys
import heapq
import scipy.cluster.vq as clust
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


class Spectral_Clusterer:
    def __init__(self,data):         #square matrix with values in <0,inf> 0 - far, inf - close 
        if (data.shape[0]!=data.shape[1]):
            raise ValueError("Data metrix", "Data metrix must be square, but shape is "+str(data.shape))
        #if (np.min(data)<0):
        #    raise ValueError("Data metrix", "Min value is <0")
        
        self.W = data
        self.point_number = data.shape[0]
        for i in range(self.point_number):
            self.W[i,i] = 0;
            
    
    ###
    # norm = 0   -   Unnormalized spectral clustering
    # norm = 1   -   Normalized spectral clustering according to Shi and Malik (2000)
    # norm = 2   -   Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)
    def run(self, cluster_number, norm = 0, KMiter=20, KMlib = True):
        self.cluster_number = cluster_number
        
        self.D = np.zeros((self.point_number,self.point_number))
        for i in range(self.point_number):
            self.W[i,i] = 0;
            self.D[i,i] = np.sum(self.W[i])
        self.L = self.D - self.W
        
        if norm == 1:
            D2 = np.diag(1./self.D.diagonal())
            self.L = np.dot(D2,self.L)
            
        if norm == 2:
            D2 = np.diag(self.D.diagonal()**(-.5))
            self.L = np.dot(D2,np.dot(self.L,D2))
        
        self.eig_val, self.eig_vect = la.eig(self.L)
        self.sortEig()
        
        #print self.eig_val
        #print np.sum(self.eig_vect[:,0]), np.sum(self.eig_vect[:,1])
        
        self.points = self.eig_vect[:, 0:self.cluster_number]
        
        if norm == 2:
            for i in range(self.point_number):
                self.points[i] = self.points[i]/la.norm(self.points[i])
        
        #print np.sum(self.eig_vect[:,0]), np.sum(self.eig_vect[:,1])
        
        #sys.stdout.write("KMean ")
        #sys.stdout.flush()
        if KMlib:
            codebook, cost = clust.kmeans(self.points, cluster_number, iter=KMiter)
            self.sol, cost2 = clust.vq(self.points, codebook)
        else:
            KM = KMean(self.points)
            self.KMdata = np.zeros(KMiter)
            bestCost = np.inf
            for i in range(KMiter): 
                KM.run(cluster_number)
                sys.stdout.write(".")
                sys.stdout.flush()
                self.KMdata[i] = KM.cost
                #print i, ": ", KM.cost
                if bestCost > KM.cost:
                    bestCost = KM.cost
                    self.sol = KM.pointsNearestCluster
                    
    
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        return self.sol
    
    
    def kNearest(self, k):
        filter = np.zeros((self.point_number,self.point_number), dtype=bool)
        for i in range(self.point_number):
            #x = heapq.nlargest(k, self.W[i])[k-1]
            #filter[i] = filter[i] | (self.W[i]>=x)
            #filter[:,i] = filter[:,i] | (self.W[i]>=x)
            
            knn = np.argsort(self.W[i])[-k:]
            filter[i,knn] = True
            filter[knn,i] = True
        
        #print np.sum(filter,1)
        self.W[filter==False] = 0
        
    def kNearestMutual(self, k):
        filter = np.empty((self.point_number,self.point_number), dtype=bool)
        filter.fill(True)
        for i in range(self.point_number):
            x = heapq.nlargest(k, self.W[i])[k-1]
            filter[i] = filter[i] & (self.W[i]>=x)
            filter[:,i] = filter[:,i] & (self.W[i]>=x)
            
        self.W[filter==False] = 0
            
    def sortEig(self):
        s = self.eig_val.argsort()
        self.eig_val = self.eig_val[s]
        self.eig_vect = self.eig_vect[:,s]
        
        
def all_perms(elements):
    if len(elements) <=1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                yield perm[:i] + elements[0:1] + perm[i:]

def swapSol(sol1, sol2, clusterNumber):
    best_pr = 0.0
    for p in all_perms(range(clusterNumber)):
        hits = 0.0
        for i in range(clusterNumber):
            hits += sum((sol1==i) & (sol2==p[i]))
        pr = hits/sol1.size
        if pr > best_pr:
            best_pr = pr
            best = p
    
    sol3 = np.copy(sol2)
    for i in range(clusterNumber):
        sol3[sol2 == best[i]] = i
    return sol3

def clusterSpearmanSC(cor, sol=None, clusterNumber = 2, SCtype = 1, KMiter=20, kcut=0, plot = False, mutual=False):
    #print "Clustering ..."
    SC = Spectral_Clusterer(np.copy(cor))
    if kcut>0:
        if mutual:
            SC.kNearestMutual(kcut)
        else:
            SC.kNearest(kcut)
    
    solSC = SC.run(clusterNumber, SCtype, KMiter)
    if sol is not None:
        solSC = swapSol(sol,solSC,clusterNumber)        
        return np.sum(sol==solSC)*1.0/len(sol), solSC, SC
    return solSC, SC

def demo():
    data= np.array([
            [0,1,1,0,0],
            [1,0,1,0,0],
            [1,1,0,0,0],
            [0,0,0,0,1],
            [0,0,0,1,0]
        ])
    data= np.array([
        [0,1,1,0,0,0,0,0,0,1],
        [1,0,1,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,1],
        [0,0,0,1,0,1,0,0,0,1],
        [0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1],
        [0,0,0,0,0,0,1,0,1,0],
        [0,0,0,0,0,0,1,1,0,0],
        [1,0,0,1,1,0,1,0,0,0]
    ])
    
    #data = np.zeros((15,15))
    #data[:5,:5] = np.ones((5,5))
    #data[10:15,10:15] = np.ones((5,5))
    #data[5:10,5:10] = np.ones((5,5))
    
    SC = Spectral_Clusterer(data)
    print SC.run(3)
    
    
    #fig = plt.figure()
    #ax = Axes3D(fig)
    
    #ax.scatter(pos[:5,0],pos[:5,1],pos[:5,2],c="r")
    #ax.scatter(pos[5:,0],pos[5:,1],pos[5:,2],c="b")
    #plt.xlim(0,100)
    #plt.ylim(0,100)
    #plt.plot(SC.points[SC.sol==0,0],SC.points[SC.sol==0,1],"go")
    #plt.plot(SC.points[SC.sol==1,0],SC.points[SC.sol==1,1],"ro")
    #plt.show()
#demo()

