# -*- coding: utf-8 -*-

import numpy as np

import metrics
import forcer
import regularizer
import transformer
import derivatives
import linAlg
import feedback

from matplotlib import pyplot as plt
                    
class lddmmReg3D:

    _f = 0
    _r = 0
    _t = 0
        
    def __init__(self, mew, lam):        
      
        self._f = forcer.forcer3D()
        self._r = regularizer.regularizer3D(mew, lam)
        self._t = transformer.transformer3D()
        
    def lddmmRegister(self, ref, tmp, sig, eps, wPath, maxIter, thresh, T):

        rows = ref.shape[0]
        cols = ref.shape[1]
        deps = ref.shape[2]
        self._r.updateDinv([rows, cols, deps], diffOp='NC')

        """
        Initialize vector fields
        """                      
        v_total = np.zeros( (T+1,) + tmp.shape + (3,) )
        uf_total = np.zeros(v_total.shape)
        ub_total = np.zeros(v_total.shape)
        refTxm = np.tile(ref, (T+1, 1, 1, 1))
        tmpTxm = np.tile(tmp, (T+1, 1, 1, 1))
        dt = np.ones((T+1))/T
        disp = np.empty_like(v_total)

        """
        Initialize list to hold objective function values throughout
        optimization
        """
        ssdList = []
        ssdList.append(metrics.SSD(ref, tmpTxm[-1]))
        currentTotalDist = ssdList[-1]
        prevTotalDist= float('Inf')
        
        """
        Figures for feedback
        """
        dSlc = 64
        ssdFig = plt.figure(1, figsize=(10, 8))
        ssdForceFig = plt.figure(2, figsize=(12, 12))
        imgFigB = plt.figure(3, figsize=(12, 12))
        imgFigF = plt.figure(4, figsize=(12, 12))
        jDetFig = plt.figure(5, figsize=(9, 9))
        
        plt.ion()
        plt.show()

        """
        The registration itself
        """ 
        k = 0 
        stepClock = 0
        while k < maxIter and abs((prevTotalDist - currentTotalDist)
                                        /currentTotalDist) > thresh:
        
            prevTotalDist = currentTotalDist
            
            #adapt step size if oscillating
            if stepClock >= 7:
                list1 = np.array(ssdList[-6:])
                list2 = np.array(ssdList[-7:-1])
                diff = list1 - list2
                diff[diff >= 0.0] = 1
                diff[diff < 0.0] = 0
                diff = np.sum(diff)
                if diff >= 3:
                    eps = eps/1.5
                    print "step size adjust, eps: " + str(eps)
                    stepClock = 0

            #Calculate the force on the images               
            force = self._f.computeForceLDDMMssd(tmpTxm, refTxm)*2.0/sig
            for i in range(0, T+1):
                jDet = linAlg.det3by3FieldAnalytical(
                        derivatives.Jacobian( uf_total[i] ))
                force[i, ..., 0] = jDet*force[i, ..., 0]
                force[i, ..., 1] = jDet*force[i, ..., 1]
                force[i, ..., 2] = jDet*force[i, ..., 2]
                
            feedback.lddmmForceFigs(ssdForceFig, force, dSlc)

            #Regularize the force             
            for i in range(0, T+1):
#                v_total[i] = v_total[i] - eps*( 2*v_total[i] +
#                    self._r.regularize(force[i]) )
                v_total[i] = v_total[i] - eps*self._r.regularize(force[i])
                 
            #Reparameterze time
#            if (k+1)%10 == 0:
#                
#                v_norms = np.empty((T+1))
#                norm_ttl = np.empty((T+1))
#                for i in range(0, T+1):
#                    v_norms[i] = np.linalg.norm(
#                                    self._r.convolve(v_total[i])   )
#                    if i == 0:
#                        norm_ttl[0] = dt[0]*v_norms[0]
#                    else:
#                        norm_ttl[i] = norm_ttl[i-1] + dt[i]*v_norms[i]
#                     
#                for i in range(0, T+1):
#                    dt[i] = (T+1)*norm_ttl[i]/norm_ttl[-1]
#                    
#                dt[1:] = dt[1:] - dt[:-1]
#                
#                new_norms = np.empty((T+1))
#                new_ttl = np.empty((T+1))
#                for i in range(0, T+1):
#                    v_total[i] = (norm_ttl[-1]/v_norms[i]/(T+1))*v_total[i]
#                    new_norms[i] = np.linalg.norm(
#                                    self._r.convolve(v_total[i]))
#                    if i ==0:
#                        new_ttl[0] = dt[0]*new_norms[0]
#                    else:
#                        new_ttl[i] = new_ttl[i-1] + dt[i]*new_norms[i]
#                
#                for i in range(0, T+1):                    
#                    print "be: " + str(v_norms[i]) + "\t" +              (
#                          "af: " + str(new_norms[i]) + "\t" + str(dt[i])  )
#                          
#                print "old ttl: " + str(norm_ttl[-1]) + "\t" + (
#                      "new ttl: " + str(new_ttl[-1])            )
                    
            #Integrate the velocity fields
            for i in range(0, T+1):
                disp[i] = dt[i]*v_total[i]
                
            #Alternate between using first 20 and second 20
            for i in range(1, T+1):
                
                ub_total[i,...,0] = -disp[i,...,0] + self._t.applyTransform(
                                           ub_total[i-1,...,0], -disp[i]    )
                ub_total[i,...,1] = -disp[i,...,1] + self._t.applyTransform(
                                           ub_total[i-1,...,1], -disp[i]    )
                ub_total[i,...,2] = -disp[i,...,2] + self._t.applyTransform(
                                           ub_total[i-1,...,2], -disp[i]    )
                
                uf_total[-(i+1),...,0] = disp[-i,...,0] + self._t.applyTransform(
                                                uf_total[-i,...,0], disp[-i]     )
                uf_total[-(i+1),...,1] = disp[-i,...,1] + self._t.applyTransform(
                                                uf_total[-i,...,1], disp[-i]     )
                uf_total[-(i+1),...,2] = disp[-i,...,2] + self._t.applyTransform(
                                                uf_total[-i,...,2], disp[-i]     )

            #Interpolate the new images                
            for i in range(1, T+1):
                tmpTxm[i] = self._t.applyTransform(tmp, ub_total[i])
                refTxm[i-1] = self._t.applyTransform(ref, uf_total[i-1])
                
            feedback.lddmmImgFigs(imgFigB, tmpTxm, dSlc)
            feedback.lddmmImgFigs(imgFigF, refTxm, dSlc)
            
            jDetFull = linAlg.det3by3FieldAnalytical(
                        derivatives.Jacobian(ub_total[-1]) ) - 1
            feedback.jdAvgFig(jDetFig, jDetFull, dSlc)
           
            v_ttl = 0
#            for i in range(0, T+1):
#                v_ttl = v_ttl + np.linalg.norm(
#                                    self._r.convolve(v_total[i])    )
            ssdList.append(metrics.SSD(ref, tmpTxm[-1]) + v_ttl)            
            feedback.objGraphs(ssdFig, ssdList)  

            print "iteraton: " + str(k)
            print "force mag:" + str(np.linalg.norm(force))
            for i in range(0, T+1):
                print metrics.SSD(refTxm[i], tmpTxm[i])
            print np.linalg.norm(v_total)
            print '--------------------------------------------'
            
            currentTotalDist = ssdList[-1]    
            k += 1
            stepClock += 1
    
        return uf_total, ub_total, refTxm, tmpTxm
        
