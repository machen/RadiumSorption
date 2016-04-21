# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:50:12 2015

@author: Michael
"""

#Script meant to supplant original radium analysis set. Works with template Excel file. Script plots results only for initial verification, use  master table script.

import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns, re
from scipy.stats import linregress

sns.set_context("poster")
saveData = False #Flag to control data overwrite
fileLocation = "Test\\" #Working folder, which contains the input folder and output
dataFile = 'ExperimentDataTemplate.xlsx' #Name of files
resultFile = 'TestResult.xlsx'

paramSource = pd.read_excel(fileLocation+dataFile,sheetname='Parameters',index_col=0,header=0)

class Parameter:
    def __init__(self,value,error=None,units=None):
        self.value = value
        self.error = error
        self.units = units
    def __repr__(self):
        return "Parameter()"
    def __str__(self):
        return " %0.2f +/- %0.2f %s"%(self.value,self.error,self.units)
    
class Sample:
    def __init__(self,sampleID,scintData,finalpH):
        """INPUTS:
        sampleID: String Identifying sample
        scintData = Pandas DataFrame containing all of the scintillation data
        """
        self.sampleID = sampleID
        self.pH = finalpH
        equilVals = []
        equilErrs = []
        for index in reversed(scintData.loc[scintData.loc[:,'Sample']==self.sampleID,:].index):
            newVal = scintData.loc[index,'CPM']/60
            newErr = scintData.loc[index,'% Error']/100
            if len(equilVals)==0:
                equilVals.append(newVal)
                equilErrs.append(newErr)
                continue
            if np.std(equilVals+newVal) > np.mean(equilErrs+newErr)*np.mean(equilVals+newVal):
                break
            else:
                equilVals.append(newVal)
                equilErrs.append(newErr)
        self.equilVal = np.mean(equilVals)
        self.equilErr = np.mean(equilErrs)

#Load parameters

scintVol = Parameter(paramSource.ix['Scintillation Volume','Value'],paramSource.ix['Scintillation Volume','Error'],paramSource.ix['Scintillation Volume','Units'])
sampVol = Parameter(paramSource.ix['Sample Volume','Value'],paramSource.ix['Sample Volume','Error'],paramSource.ix['Sample Volume','Units'])
minlType = Parameter(paramSource.ix['Mineral Type','Value'])
minlMass = Parameter(paramSource.ix['Mineral Mass','Value'],paramSource.ix['Mineral Mass','Error'],paramSource.ix['Mineral Mass','Units'])
countEff = Parameter(paramSource.ix['Counting efficiency','Value'], units = paramSource.ix['Counting efficiency','Units'])
stockAct = Parameter(paramSource.ix['Stock Activity','Value'],error=paramSource.ix['Stock Activity','Error'],units = paramSource.ix['Stock Activity','Units'])

#Load raw data into memory

rawData = pd.read_excel(fileLocation+dataFile,sheetname='Data',index_col = 0,header = 0)
sampleDPM = rawData.ix[:,'Scintillation Count (CPM)']/countEff.value
dsampleDPM = rawData.ix[:,'% Error']/100*sampleDPM
Cw = sampleDPM/scintVol.value
dCw = np.sqrt((scintVol.error/scintVol.value)**2+(dsampleDPM/sampleDPM)**2)*Cw
stockVol = rawData.ix[:,'Stock Volume Added'].values
dstockVol = rawData.ix[:,'Stock Vol Err'].values
rawData['Cw (DPM/mL)'] = pd.Series(Cw,index=rawData.index)
rawData['dCw (DPM/mL)'] = pd.Series(dCw,index=rawData.index)
totAct = stockVol * stockAct.value
dtotAct = np.sqrt((dstockVol/stockVol)**2+(stockAct.error/stockAct.value)**2)*totAct
rawData['Total Activity (DPM)'] = pd.Series(totAct,index=rawData.index)
rawData['dTotal Activity (DPM)'] = pd.Series(dtotAct,index=rawData.index)
Cs = (totAct-Cw*sampVol.value)/minlMass.value
dCs = Cs*np.sqrt((np.sqrt(dtotAct**2+(np.sqrt((dCw/Cw)**2+(sampVol.error/sampVol.value)**2)*Cw*sampVol.value)**2)/(totAct-Cw*sampVol.value))**2+(minlMass.error/minlMass.value)**2)
rawData['Cs (DPM/mg)'] = pd.Series(Cs,index=rawData.index)
rawData['dCs (DPM/mg)'] = pd.Series(dCs,index = rawData.index)
fSorb = Cs*minlMass.value/totAct
dfSorb = np.sqrt((dCs/Cs)**2+(minlMass.error/minlMass.value)**2+(dtotAct/totAct)**2)
rawData['fSorb (DPM/DPM)'] = pd.Series(fSorb,index=rawData.index)
rawData['dfSorb (DPM/DPM)'] = pd.Series(dfSorb,index=rawData.index)

#Average exisiting data by using the established pattern
pat = re.compile('([0-9a-zA-Z]*)\_([a-zA-Z])(\_[0-9a-zA-Z]*)?') #pattern searches for THINGS_LETTER_THINGS, essentially looking for samples. Pattern used for sample averaging.
nRows = np.size(rawData,0)
expIDs = [] #List of sample IDs with multiplicate notation removed

#Convert sample IDs from multiple notation to single notation (to combine multiplicates)
for testID in rawData.index:
    IDmatches = re.search(pat,testID)
    if IDmatches == None or len(IDmatches.groups())==1:
        continue
    elif IDmatches.groups()[2]!=None:
        expID = IDmatches.group(1)+IDmatches.group(3)
    else:
        expID = IDmatches.group(1)
    expIDs.append(expID)
rawData['expID'] = pd.Series(expIDs,index=rawData.index)
#Use the generated multiplicate names as an index for averaging values

procData = pd.DataFrame(np.zeros((len(set(expIDs)),7)),index=set(expIDs),columns=['Cw','sCw','Cs','sCs','pH','spH','nVals'])

for ID in procData.index:
    idPat = re.compile(ID)
    pHvals = []
    Cwvals = []
    Csvals = []
    fSorbvals = []
    for sample in rawData.index:
        if rawData.ix[sample,'Include'] == False:
            continue
        if idPat.match(rawData.ix[sample,'expID']):
            pHvals.append(rawData.ix[sample,'Final pH'])
            Cwvals.append(rawData.ix[sample,'Cw (DPM/mL)'])
            Csvals.append(rawData.ix[sample,'Cs (DPM/mg)'])
            fSorbvals.append(rawData.ix[sample,'fSorb (DPM/DPM)'])
    procData.loc[ID,'pH'] = np.mean(pHvals)
    procData.loc[ID,'spH'] = np.std(pHvals)
    procData.loc[ID,'Cw'] = np.mean(Cwvals)
    procData.loc[ID,'sCw'] = np.std(Cwvals)
    procData.loc[ID,'Cs']= np.mean(Csvals)
    procData.loc[ID,'sCs'] = np.std(Csvals)
    procData.loc[ID,'fSorb'] = np.mean(fSorbvals)
    procData.loc[ID,'sfSorb'] = np.std(fSorbvals)
    procData.loc[ID,'nVals'] = len(pHvals)

f1 = plt.figure(1)
plt.clf()
ax1 = f1.add_subplot(111)
colors = sns.color_palette()
dataErr = ax1.errorbar(Cw,Cs,xerr=dCw,yerr=dCs,marker='.',linestyle = 'none',color=colors[0],label='Raw Data')
avgErr = ax1.errorbar(procData.loc[:,'Cw'],procData.loc[:,'Cs'],yerr=procData.loc[:,'sCs'],xerr=procData.loc[:,'sCw'],marker='.',linestyle='none',label='Averaged Data',color=colors[1])
plt.xlabel('Cw (DPM/mL)') 
plt.ylabel('Cs (DPM/mg)')
plt.legend(loc=0)


f2 = plt.figure(2)
plt.clf()
ax2 = f2.add_subplot(111)
dataErr2 = ax2.errorbar(rawData.ix[:,'Final pH'],rawData.ix[:,'fSorb (DPM/DPM)'],yerr=rawData.ix[:,'dfSorb (DPM/DPM)'],marker='.',linestyle = 'none',color=colors[0],label='Raw Data')
avgErr2 = ax2.errorbar(procData.loc[:,'pH'],procData.loc[:,'fSorb'],yerr=procData.loc[:,'sfSorb'],marker='.',linestyle='none',label='Averaged Data',color=colors[1])
plt.xlabel('pH')
plt.ylabel('Fraction Sorbed')
plt.legend(loc=0)
plt.ylim([0,1])

#Write averaged data to new file    
if saveData:
    writer = pd.ExcelWriter(fileLocation+resultFile)
    paramSource.to_excel(writer,sheet_name='Parameters')
    rawData.to_excel(writer,sheet_name='Processed Data',index_label = 'Sample ID')
    procData.to_excel(writer,sheet_name='Result',index_label = 'Sample')
    writer.save()
    f1.savefig(fileLocation+'CwvCsPlot.png',dpi=1500,format='png')
    f2.savefig(fileLocation+'pHvfSorb.png',dpi=1500,format='png')
