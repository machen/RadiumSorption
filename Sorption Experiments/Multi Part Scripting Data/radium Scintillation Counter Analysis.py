# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:14:28 2015
Script to convert results from scintillation counter to known concentrations through the use of a standard curve
@author: Michael
"""



#Variables titled with a small "d" in front of them represent absolute error of a value

import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns, re
from scipy.stats import linregress

pat = re.compile('(\d*)\_(\w)\_(\w*)') #pattern searches for NUMBER_LETTER_LETTERS, essentially looking for samples

#Parameters

scintVol = 2 #mL
dscintVol = 0.1 #mL

sampVol = 100 #mL
dsampVol = .172 #mL
minlConc = 2245 #ppm Fe
dminlConc = 23.7 #ppm Fe
minlVol = 1.78E-3 #Liters
dminlVol = 0.002E-3 #Liters
molesFe = minlConc*minlVol/55850 #moles Fe as Mineral Specified
dmolesFe = np.sqrt((dminlConc/minlConc)**2+(dminlVol/minlVol)**2)*molesFe
FHYmass = 168700 #formula mass in mg
minlMass = 20 #molesFe*FHYmass #mg of mineral 
dminlMass = 1
countEff = 7308.7/7656
stockCPM = 18982.2/2.0/countEff #DPM/mL for radium stock added to samples
dstockCPM = 0.0037*stockCPM 

saveData = True #Do you want to overwrite current data sets?

#Load data to analyze
dataFileName = "PYR_OX_08_17_2015"
dataFilePath = dataFileName+"/"
rawData = pd.read_csv(dataFilePath+dataFileName+".csv",header=1)

#Substitute a calibration curve, assume that sample actual concentrations just a matter of efficiency

##Load file containing calibration info for the scintillation counter
#calFile = "ScintCal_ReactStd.csv"
#calData = pd.read_csv(calFile,header=1)
#
##Create Calibration Curve to convert CPM to DPM
#xData = calData.ix[:,'CPM'].values
#yData = calData.ix[:,'Ra226 (DPM)'].values
#xDataRange = np.linspace(min(xData),max(xData))
#calFit = linregress(xData,yData)
#calFit = {'Slope':calFit[0],'Intercept':calFit[1],'R2':calFit[2]**2,'p':calFit[3],'StdErr':calFit[4]}
#intErr = calFit['StdErr']*np.sqrt(1/len(xData)+np.mean(xData)**2/np.sum((xData-np.mean(xData))**2))
#slopeErr = calFit['StdErr']*np.sqrt(1/np.sum((xData-np.mean(xData))**2))
#calFit['IntErr']=intErr
#calFit['SlopeErr']=slopeErr
#f1 = plt.figure(1)
#p1, = plt.plot(calData.loc[:,'Ra226 (DPM)'],calData.loc[:,'CPM'],'ob')
#p2, = plt.plot(np.polyval((calFit['Slope'],calFit['Intercept']),xDataRange),xDataRange,'-r')
#plt.title('Scintillation Counter Calibration: '+calFile)
#plt.xlabel(r'$Ra^{226}$ Activity (DPM)')
#plt.ylabel('Scintillation CPM')
#plt.legend((p1,p2),('Data','Fit R2: '+str(calFit['R2'])),loc=0)
#if saveData:
#    plt.savefig(dataFilePath+"Calibration Curve.png")
#plt.show()
#
##Apply Calibration to Data to get Cw, includes a simple multiplication to adjust for filtration loss
#sampleDPM = np.polyval((calFit['Slope'],calFit['Intercept']),rawData.ix[:,1])
#stockDPM =np.polyval((calFit['Slope'],calFit['Intercept']),stockCPM)
sampleDPM = rawData.ix[:,1]/countEff
Cw = sampleDPM/scintVol #DPM Radium 226 per mL of supernatent
xData = rawData.ix[:,1]
xErr= rawData.ix[:,2]/100.0 #Relative error of counts
dsampleDPM = xErr*sampleDPM #np.sqrt(calFit['IntErr']**2+(xData*calFit['Slope']*np.sqrt((xErr/xData)**2+(calFit['SlopeErr']/calFit['Slope'])**2))**2)
dCw = np.sqrt((dscintVol/scintVol)**2+(dsampleDPM/sampleDPM)**2)*Cw
rawData['Cw (DPM/mL)'] = pd.Series(Cw,index=rawData.index)
rawData['dCw (DPM/mL)'] = pd.Series(dCw,index=rawData.index)

#Calculate Cs from the data

stockVol = rawData.ix[:,'Volume Added'].values #mL of Stock added
dstockVol = rawData.ix[:, 'Vol Err'].values 
totAct = stockVol * stockCPM
#totAct = rawData.ix[:,'Total Activity'].values
dtotAct = np.sqrt((dstockVol/stockVol)**2+(dstockCPM/stockCPM)**2)*totAct
rawData['Total Activity (DPM)'] = pd.Series(totAct,index=rawData.index)
rawData['dTotal Activity (DPM)'] = pd.Series(dtotAct,index=rawData.index)
Cs = (totAct-Cw*sampVol)/minlMass
dCs = Cs*np.sqrt((np.sqrt(dtotAct**2+(np.sqrt((dCw/Cw)**2+(dsampVol/sampVol)**2)*Cw*sampVol)**2)/(totAct-Cw*sampVol))**2+(dminlMass/minlMass)**2)
rawData['Cs (DPM/mg Fe)'] = pd.Series(Cs,index=rawData.index)
rawData['dCs (DPM/mg Fe)'] = pd.Series(dCs,index = rawData.index)
if saveData:
    rawData.to_csv(dataFilePath+dataFileName+" Calculations.csv")

f2 = plt.figure(2)
plt.clf()
ax1 = f2.add_subplot(111)
perr = ax1.errorbar(Cw,Cs,yerr = dCs, xerr = dCw, fmt="none")
Dat, = ax1.plot(Cw,Cs,"xb")
ax1.ticklabel_format(axis='y',style = 'sci',scilimits=(-2,2))
ax1.ticklabel_format(axis='x',style='sci',scilimits=(-2,2))
plt.show()
