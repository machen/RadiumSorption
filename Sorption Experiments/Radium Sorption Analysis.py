# -*- coding: utf-8 -*-
"""13
Created on Fri Jun 26 :16:13 2015

Plotting and fitting of isotherm data for radium sorption

@author: machen
"""

import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import linregress

saveData = False

#load Data
dataFileName = "Isotherm_Data"
dataFilePath = "FHY_07_21_2015/"
rawData = pd.read_csv(dataFilePath+dataFileName+".csv",header=0)

#Plot Data
Cw = rawData.ix[:,'Cw'].values
Cs = rawData.ix[:,'Cs'].values
dCw = rawData.ix[:,'sCw'].values
dCs = rawData.ix[:,'sCs'].values

plotfit = linregress(Cw, Cs)
plotfit = {'Slope':plotfit[0],'Intercept':plotfit[1],'R2':plotfit[2]**2,'p':plotfit[3],'StdErr':plotfit[4]}

f1 = plt.figure(1)
plt.clf()
perr = plt.errorbar(Cw, Cs, yerr = dCs, xerr = dCw, fmt='none')
p1, = plt.plot(Cw, Cs, "ob")
p2, = plt.plot([0.1*np.min(Cw),1.3*np.max(Cw)],np.polyval((plotfit['Slope'],plotfit['Intercept']),[0.3*np.min(Cw),1.3*np.max(Cw)]),'-r')
#plt.title('Linear Fit '+dataFileName+' Kd = '+str(plotfit['Slope']))
plt.title('pH 3 Ferrihydrite Isotherm Kd = '+"%.2f"%plotfit['Slope'])
plt.xlabel('Cw (DPM/mL)')
plt.ylabel('Cs (DPM/mg Fe)')
plt.legend((p1,p2,),('Data','Fit R2: '+ "%.3f"%plotfit['R2']),loc=0)
if saveData:
    plt.savefig(dataFilePath+"Isotherm Plot.svg",dpi=1200,type='svg')

##Langmuir Isotherm (1/Cs vs 1/Cw)
#
#f2 = plt.figure(2)
#plt.clf()
#
#langfit = linregress(1/Cw,1/Cs)
#langfit = {'Slope':langfit[0], 'Intercept':langfit[1], 'R2':langfit[2]**2, 'p':langfit[3], 'StdErr':langfit[4]}
#
#perr1 = plt.errorbar(1/Cw,1/Cs, yerr = (1/Cs**2)*dCs, xerr = (1/Cw**2)*dCw, fmt='none')
#p3, = plt.plot(1/Cw, 1/Cs, "ob")
#p4, = plt.plot(np.arange(0,0.05,0.001),np.polyval((langfit['Slope'],langfit['Intercept']),np.arange(0,0.05,0.001)),'-r')
#plt.title('Langmuir Isotherm 06_17_2015')
#plt.xlabel('1/Cw')
#plt.ylabel('1/Cs')
#plt.legend((p3,p4,),('Data', 'Fit R2: '+str(langfit['R2'])),loc=0)
#if saveData:
#    plt.savefig(dataFilePath+'Langmuir Isotherm plot.png',dpi=600)
#
##Trying logCs vs logCw
#
#f3 = plt.figure(3)
#plt.clf()
#logfit = linregress(np.log(Cw),np.log(Cs))
#logfit = {'Slope':logfit[0], 'Intercept':logfit[1], 'R2':logfit[2]**2, 'p':logfit[3], 'StdErr':logfit[4]}
#
#perr_log = plt.errorbar(np.log(Cw), np.log(Cs), yerr = (1/Cs)*dCs, xerr=(1/Cw)*dCw, fmt='none')
#p5, = plt.plot(np.log(Cw),np.log(Cs),"ob")
#p6, = plt.plot(np.arange(0, 10, 1),np.polyval((logfit['Slope'],logfit['Intercept']),np.arange(0,10,1)), '-r')
#plt.title('log Cs vs log Cw')
#plt.xlabel('log Cw')
#plt.ylabel('log Cs')
#plt.legend((p5,p6,), ('Data', 'Fit R2: '+str(logfit['R2'])),loc=0)
#if saveData:
#    plt.savefig(dataFilePath+'LogCs vs LogCw plot.png',dpi=600)

plt.show()                       
