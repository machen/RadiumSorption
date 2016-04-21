# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 17:44:02 2016

@author: Michael
"""

#Script to plot surface complexation results. 

import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
from win32com.client import Dispatch
from string import Template
import itertools, os

#New script will allow user to run a series of PHREEQC calculation using a template file, allowing for iteration over multiple values

class simulation:
    def __init__(self,parameters,templateFile,database,pHrange=[2,10]):
        self.param = parameters #Dictionary of parameters
        self.pHrange = pHrange #Range of pHs each simulation should run over
        with open(templateFile,'r') as tempFile:
            self.templStr = Template(tempFile.read())
        self.db = database #Thermodynamic database to use for simulations
        self.data = []

    def runPHREEQC(self,inputStr):
        #Run a specific instance of phreeqc, and return the output array, which is specified in the template file
        dbase = Dispatch('IPhreeqcCOM.Object')
        dbase.LoadDatabase(self.db)
        dbase.RunString(inputStr)
        out = dbase.GetSelectedOutputArray()
        res = out[1]
        return res
    
    def generateData(self):
        #Runs the PHREEQC simulation for the range of pH values specified in the intialized pHrange variable
        for i in np.arange(self.pHrange[0],self.pHrange[1]+0.1,0.1):
            val = {'pH':i}
            inputParam=self.param
            inputParam.update(val)
            subRes = self.runPHREEQC(self.templStr.substitute(inputParam))
            self.data.append(subRes)
        self.data = pd.DataFrame(self.data,columns=['pH','Ra'])
        fSorb = (inputParam['totRa']-self.data.ix[:,'Ra'].values)/inputParam['totRa']
        self.data['fSorb'] = pd.Series(fSorb,index=self.data.index)
    def getData(self):
        return self.data
        
def extractData(path):
    #Function retrieves data from an excel spreadsheet
    fileLoc = path
    data = pd.read_excel(fileLoc)
    return data
             
totRa = 5.979e-010
k1 = 6.66
k2 = -5.67
db = "C:\Program Files (x86)\USGS\Phreeqc Interactive 3.1.4-8929\database\sit.dat"
tmp = "Single site model NoElectroStatics.txt"
titleString = "Single site model, No Electrostatics, K1 = 0, K2 = -5.67, 0.1-10 mol of sites"
#x = simulation({'totRa':totRa,'k1':k1,'k2':k2},[2,10],tmp,db)
#x.generateData()
sns.set_palette("deep",n_colors = 6)
expData = extractData('RaFHY Experimental Data.xlsx')
exp5 = expData.ix[expData['Total Activity']==5,:]
exp10 = expData.ix[expData['Total Activity']==10,:]
exp50 = expData.ix[expData['Total Activity']==50,:]
exp100 = expData.ix[expData['Total Activity']==100,:]
exp500 = expData.ix[expData['Total Activity']==500,:]

f1 = plt.figure(num=1,figsize=(10,8))
f1.clf()
ax = f1.add_subplot(111)

labelStr = "1 site model, K1: {k1} K2: {k2} Sites (mol): {sites}"
K2val = np.array([-5.67])
K1val = np.array([1])
siteMolVal = np.array([0.3])
#K1val = np.linspace(1,3,num=2,endpoint=True)
#siteMolval = np.logspace(-3,0,num=12,endpoint=True, base=10)
#siteMolVal = np.linspace(0.02,0.05,num=2, endpoint=True)
ncol = np.size(K1val)*np.size(K2val)*np.size(siteMolVal)
cmap = sns.cubehelix_palette(n_colors=ncol,dark=0.3,rot=0.4,light=0.8,gamma=1.3)
palette = itertools.cycle(cmap)
for K1 in K1val:
    for K2 in K2val:
        for siteMol in siteMolVal:
            x = simulation({'totRa':totRa,'k1':K1,'k2':K2,'sites':siteMol},tmp,db)
            x.generateData()
            simRes = x.getData()
            ax.plot(simRes.ix[:,'pH'],simRes.ix[:,'fSorb'],'-',label=labelStr.format(k1=K1,k2=K2,sites=siteMol),color=next(palette))
            print K1,K2,siteMol
        

exp5Plot = ax.errorbar(exp5.ix[:,'pH'],exp5.ix[:,'fSorb'],xerr=exp5.ix[:,'spH'],yerr=exp5.ix[:,'sfSorb'],fmt='o',label='Experimental Data 5 Bq Total')
exp10Plot = ax.errorbar(exp10.ix[:,'pH'],exp10.ix[:,'fSorb'],xerr=exp10.ix[:,'spH'],yerr=exp10.ix[:,'sfSorb'],fmt='o',label='Experimental Data 10 Bq Total')
exp50Plot = ax.errorbar(exp50.ix[:,'pH'],exp50.ix[:,'fSorb'],xerr=exp50.ix[:,'spH'],yerr=exp50.ix[:,'sfSorb'],fmt='o',label='Experimental Data 50 Bq Total')
exp100Plot = ax.errorbar(exp100.ix[:,'pH'],exp100.ix[:,'fSorb'],xerr=exp100.ix[:,'spH'],yerr=exp100.ix[:,'sfSorb'],fmt='o',label='Experimental Data 100 Bq Total')
exp500Plot = ax.errorbar(exp500.ix[:,'pH'],exp500.ix[:,'fSorb'],xerr=exp500.ix[:,'spH'],yerr=exp500.ix[:,'sfSorb'],fmt='o',label='Experimental Data 500 Bq Total')


ax.legend(loc=0)
ax.set_title(titleString)
ax.set_xlabel('pH')
ax.set_ylabel('Fraction Sorbed')
ax.set_ylim([-0.01,1.0])
plt.show()
    


#Original script which focused on pulling in and plotting data from already produced data
"""
def extractData(path,PHREEQC=False):
    #Function retrieves data from an excel spreadsheet, also includes the error
    fileLoc = path
    data = pd.read_excel(fileLoc)
    return data

#tetraMod = extractData('RaFHY_Sajih4SiteModel.xlsx')
#SCM = extractData('RaFHY_SajihSCM.xlsx')
SCM = extractData('RaEqTestOUTPUT.xlsx',PHREEQC=True)
expData = extractData('RaFHY Experimental Data.xlsx')
#testMod = extractData('RaFHY_SajihSCM_TEST.xlsx')
exp5 = expData.ix[expData['Total Activity']==5,:]
exp10 = expData.ix[expData['Total Activity']==10,:]
exp50 = expData.ix[expData['Total Activity']==50,:]
exp100 = expData.ix[expData['Total Activity']==100,:]
exp500 = expData.ix[expData['Total Activity']==500,:]

f1 = plt.figure(num=1,figsize=(10,8))
plt.clf()
ax = f1.add_subplot(111)
#tetplot = ax.plot(tetraMod['pH'],tetraMod['fSorb'],'-',label='Tetradentate Model, Sajih')
SCMplot = ax.plot(SCM.ix[:,'pH'],SCM.ix[:,'fSorb'],'-',label='Simple Complexation Model, Sajih constants, 1 site type')
#testplot = ax.plot(testMod['pH'],testMod['fSorb'],'-',label='Simple Complexation Model, Weak log K: '+weakK+', Strong log K: '+strongK)
#weakTest = ax.plot(testMod['pH'],testMod['fWeak'],'--',label='Weak sites')
#strTest = ax.plot(testMod['pH'],testMod['fStrong'],'--',label='Strong Sites')
exp5Plot = ax.errorbar(exp5.ix[:,'pH'],exp5.ix[:,'fSorb'],xerr=exp5.ix[:,'spH'],yerr=exp5.ix[:,'sfSorb'],fmt='o',label='Experimental Data 5 Bq Total') #Need to fix this coloring scheme
exp10Plot = ax.errorbar(exp10.ix[:,'pH'],exp10.ix[:,'fSorb'],xerr=exp10.ix[:,'spH'],yerr=exp10.ix[:,'sfSorb'],fmt='o',label='Experimental Data 10 Bq Total')
exp50Plot = ax.errorbar(exp50.ix[:,'pH'],exp50.ix[:,'fSorb'],xerr=exp50.ix[:,'spH'],yerr=exp50.ix[:,'sfSorb'],fmt='o',label='Experimental Data 50 Bq Total')
exp100Plot = ax.errorbar(exp100.ix[:,'pH'],exp100.ix[:,'fSorb'],xerr=exp100.ix[:,'spH'],yerr=exp100.ix[:,'sfSorb'],fmt='o',label='Experimental Data 100 Bq Total')
exp500Plot = ax.errorbar(exp500.ix[:,'pH'],exp500.ix[:,'fSorb'],xerr=exp500.ix[:,'spH'],yerr=exp500.ix[:,'sfSorb'],fmt='o',label='Experimental Data 500 Bq Total')
ax.legend(loc=0)
ax.set_title(str(datetime.date.today()))
ax.set_xlabel('pH')
ax.set_ylabel('Fraction Sorbed')
ax.set_ylim([-0.01,1.0])
plt.show()
#f1.savefig('Radium Complexation All.pdf',dpi=900)
"""