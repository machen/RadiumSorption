# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 09:46:33 2014

@author: Michael
"""

#Mucking about with problem set, curve fitting via minimization of sum of squared error
import numpy as np, matplotlib.pyplot as plt, pandas as pd, scipy.optimize as sp

def freundlich(Ci,K,n): #Calculates the Solid concentration given K (adsorption coefficient), n (parameter), Ci (Solution concentration)
    return K*Ci**(1.0/n)

def langmuir(Ci,k,b):
    return k*Ci*b/(1.0+k*Ci) #Calculates the Solid concentration using a langmuir isotherm with k (adsorption coefficient), b (adsoprtion max), Ci (Solution concentration)
    
def errorCalc(actual,calc): #Calculates the sum squared error for two data sets
    return np.sqrt(np.sum((actual-calc)**2)/np.size(actual,0))
    
directory = ""
name = "FHY_2014_11_25"
filename = name+".csv"
raw = pd.read_csv(directory+filename)
Cw = raw['Cw']
Cs = raw['Cs']

try:
    dCw = raw['dCw'] #Fractional uncertainties
    dCs = raw['dCs'] #Fractional unceratinties
    dCw = Cw*dCw #Convert fractional uncertainties to straight uncertainties
    dCs = Cs*dCs #Convert fractional uncertainties to straight uncertainties
except KeyError:
    pass

filename = "FHY_2014_12_7.csv"
raw = pd.read_csv(directory+filename)
Cwnew = raw['Cw']
Csnew = raw['Cs']
dCwnew = raw['dCw'] #Fractional uncertainties
dCsnew = raw['dCs'] #Fractional unceratinties
dCwnew = Cwnew*dCwnew #Convert fractional uncertainties to straight uncertainties
dCsnew = Csnew*dCsnew #Convert fractional uncertainties to straight uncertainties


freundFit = sp.curve_fit(freundlich,Cw,Cs,[8E-7,0.42],maxfev=10000)
langFit = sp.curve_fit(langmuir,Cw,Cs,[10**-5,12000],maxfev=10000)
kdFit = np.polyfit(Cw,Cs,1)


CwM = np.arange(0.5*min(Cw),1.1*max(Cw))

f1 = plt.figure(1)
plt.clf()
perr = plt.errorbar(Cw[0:-2],Cs[0:-2],yerr = dCs[0:-2],xerr = dCw[0:-2],fmt=None)
perr2 = plt.errorbar(Cwnew,Csnew,yerr = dCsnew,xerr = dCwnew,fmt=None)
p1, = plt.plot(Cw[0:-2],Cs[0:-2],"xb")
p12, = plt.plot(Cwnew,Csnew,"xg")
p2, = plt.plot(CwM,freundlich(CwM,freundFit[0][0],freundFit[0][1]),'-r')
p3, = plt.plot(CwM,langmuir(CwM,langFit[0][0],langFit[0][1]),'-k')
p4, = plt.plot(CwM,np.polyval(kdFit,CwM),'-c')
plt.xlabel('Cw')
plt.ylabel('Cs')
plt.title('Sorption Isotherm Fits')
plt.legend((p1,p12,p2,p3,p4),('Experiment Nov 25','Experiment Dec 7','Freundlich Fit','Langmuir fit','Kd Fit'),loc=0)
f1.set_size_inches(12.5,6.5)
plt.savefig(name+".pdf",dpi=600)
freundError = errorCalc(Cw,freundlich(Cw,freundFit[0][0],freundFit[0][1]))
langError = errorCalc(Cw,langmuir(Cw,langFit[0][0],langFit[0][1]))