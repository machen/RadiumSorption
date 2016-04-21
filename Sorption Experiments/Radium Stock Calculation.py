# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:23:36 2015

@author: tiffwang
"""

import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns, re
from scipy.stats import linregress

saveData=True

#Load Radium Stock Scintillation Data
dataFileName = "Radium_Stock_1"
dataFilePath = "Radium_Stock/"
radiumData=pd.read_csv(dataFilePath+dataFileName+".csv",header=1)

#Load Calibration Data
calFile = "ScintCal_07_27_2015.csv"
calData = pd.read_csv(calFile,header=1)

#Create Calibration Curve to convert CPM to DPM
xData = calData.ix[:,'CPM'].values
yData = calData.ix[:,'Ra226 (DPM)'].values
calFit = linregress(xData,yData)
calFit = {'Slope':calFit[0],'Intercept':calFit[1],'R2':calFit[2]**2,'p':calFit[3],'StdErr':calFit[4]}
intErr = calFit['StdErr']*np.sqrt(1/len(xData)+np.mean(xData)**2/np.sum((xData-np.mean(xData))**2))
slopeErr = calFit['StdErr']*np.sqrt(1/np.sum((xData-np.mean(xData))**2))
calFit['IntErr']=intErr
calFit['SlopeErr']=slopeErr
f1 = plt.figure(1)
p1, = plt.plot(calData.loc[:,'Ra226 (DPM)'],calData.loc[:,'CPM'],'ob')
p2, = plt.plot(np.polyval((calFit['Slope'],calFit['Intercept']),np.arange(0,31000,1000)),np.arange(0,31000,1000),'-r')
plt.title('Scintillation Counter Calibration: '+calFile)
plt.xlabel(r'$Ra^{226}$ Activity (DPM)')
plt.ylabel('Scintillation CPM')
plt.legend((p1,p2),('Data','Fit R2: '+str(calFit['R2'])),loc=0)
if saveData:
    plt.savefig(dataFilePath+"Calibration Curve.png")
plt.show()

#Apply Calibration to get Radium Stock Concentration 
radiumDPM = np.polyval((calFit['Slope'],calFit['Intercept']),radiumData['CPM']/filterPass)
radiumData['DPM'] = pd.Series(radiumDPM, index=radiumData.index)
if saveData: 
    radiumData.to_csv(dataFilePath+" Radium Stock.csv")