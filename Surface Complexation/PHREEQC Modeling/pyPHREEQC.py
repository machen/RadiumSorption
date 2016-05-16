# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 17:44:02 2016

@author: Michael
"""

#Script to plot surface complexation results. 

import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
from win32com.client import Dispatch
from string import Template
import itertools, os, copy

#New script will allow user to run a series of PHREEQC calculation using a template file, allowing for iteration over multiple values

class simulation:
    def __init__(self,parameters,templateFile,database,pHrange=[2,10],masterTable=None):
        self.param = parameters #Dictionary of parameters, which includes everything you want to input to the template
        self.pHrange = pHrange #Range of pHs each simulation should run over
        with open(templateFile,'r') as tempFile:
            self.templStr = Template(tempFile.read()) #Template allows for easy pythonic substitution of the model parameters into a given run of the model (template files formatted assuming this is used)
        self.templFile = templateFile #Location of the template, which is the key linked to the master table of data that have already been run
        self.db = database #Thermodynamic database to use for simulations
        if masterTable:
            self.masterTable = masterTable #Can preload the master table, and keep it updated during execution instead of having to read/write each time
        else:
            self.masterTable = self.loadMaster() #Loads master table into memory
        self.data = pd.DataFrame() #Should only contian data that are specified by the parameters and pH range

    def runPHREEQC(self,inputStr):
        """inputStr: String that you want PHREEQC to run, should have valid PHRREQC syntax"""
        #Run a specific instance of phreeqc, and return the output array, which is specified in the template file. 
        dbase = Dispatch('IPhreeqcCOM.Object') 
        dbase.LoadDatabase(self.db)
        dbase.RunString(inputStr)
        out = dbase.GetSelectedOutputArray()
        res = out
        return res
    
    def generateData(self):
        #Runs the PHREEQC simulation for the range of pH values specified in the intialized pHrange variable. Checks to see if those values are already in the master table.
        for i in np.arange(self.pHrange[0],self.pHrange[1]+0.1,0.1):
            val = {'pH':i}
            inputParam=copy.deepcopy(self.param) #Create a separate copy of the parameters so that we don't accidentally change the param table in the simulation
            inputParam.update(val)
            chk = self.checkMaster(inputParam) #Calls chkMaster() to see if the data is in the table already or not
            if chk.empty:
                subRes = self.runPHREEQC(self.templStr.substitute(inputParam)) #Run phreeqc with the parameters if the data is not yet there
                subRes = pd.DataFrame([subRes[1]],columns=subRes[0])
                fSorb = (inputParam['totRa']-subRes.ix[:,'Ra(mol/kgw)'].values)/inputParam['totRa']
                subRes['fSorb'] = pd.Series(fSorb,index=subRes.index)
            else:
                subRes = chk #If the data is there, append the result that would have come out of running PHRREQC instead. Note that I assume that a given simulation instance may have both already run and unrun data
            self.data = self.data.append(subRes,ignore_index=True) #Note that self.data is a pandas dataframe containing ALL of the results from selected output
    def loadMaster(self):
        #Assumes your template is saved with some kind of extension of the form ".xxx", and loads the master table into the simulation
        #Loads a master table into the simulation that can be used to see if a simulation has been run or not
        masterTabPath = self.templFile[:-4]+'.csv' #Mastertable name is just the template name as a .csv
        if os.path.isfile(masterTabPath): #Tries to find the table in the current directory ONLY 
            masterTab= pd.read_csv(masterTabPath,header=0) #Load the table as a csv, which is how the table is saved
            return masterTab
        else:
            masterTab = pd.DataFrame() #Can't find it, make an empty table
            return masterTab
    def checkMaster(self,params):
        matchData = copy.deepcopy(self.masterTable) #Creates a separate copy of the mastertable, which is then sliced according to the parameters in params
        if not matchData.empty: #Need to make sure matchData isn't empty before trying to slice it
            for key in params: #Iterate over all the keys in params, slicing out the master table data that matches within the error 1E-8. COULD SPECIFY ERROR IF WE WANTED
                matchData = matchData.loc[abs(matchData.loc[:,key]-params[key])<1E-8,:]
            return matchData
        else: #Returns empty dataframe if no match
            return matchData
    def addDataToMaster(self,writeMaster=False):
        #Save data into master table of results. Should include all parameters
        newData = self.data
        n=len(newData.index) 
        #iterate through the keys in the parameters, appending a matrix of the values repeated for each entry
        for key in self.param:
            newData[key] = pd.Series(np.ones(n)*self.param[key], index=newData.index)
        newMaster = pd.concat([newData,self.masterTable],ignore_index=True)
        self.masterTable = newMaster
        if writeMaster:
            newMaster.to_csv(self.templFile[:-4]+'.csv',index=False)
    def plotSpeciation(self,titleStr='',solidTag="m_Fhy"):
        f2 = plt.figure(2)
        f2.clf()
        ax2 = f2.add_subplot(111)
        totRa = x.param['totRa']
        cmap = sns.color_palette("hls",n_colors=16)
        color = itertools.cycle(cmap)
        color2 = itertools.cycle(cmap)
        for colName in self.data.columns:
            if colName.startswith(solidTag): #m_Fhy is a marker for solid species
                ax2.plot(self.data.ix[:,'pH'].values,self.data.ix[:,colName].values/totRa,'--',label=colName,color=next(color))
            elif colName.startswith("m_Ra"): #m_Ra is a marker for solution species
                ax2.plot(self.data.ix[:,'pH'].values,self.data.ix[:,colName].values/totRa,'-',label=colName,color=next(color2))
            else: #Do not plot things that are not part of the speciation
                continue
        #ax2.plot(self.data.ix[:,'pH'],self.data.ix[:,'fSorb'],'-k',label='Fraction Sorbed')
        plt.legend(loc=0)
        plt.title(titleStr+" Speciation")
        plt.xlabel("pH")
        plt.ylabel("Fractional Abundance")
        plt.ylim([-0.01,1.0])
    def getData(self):
        return self.data
    def getMaster(self):
        return self.masterTable
    def getParam(self):
        return self.param
def extractData(path):
    #Function retrieves data from an excel spreadsheet
    fileLoc = path
    data = pd.read_excel(fileLoc)
    return data
             
totRa = 5.979e-11
k1 = 6.66
k2 = -5.67
totalSites = 5.98E-5 #Total expected number of sites given 2 sites/nm^2 on FHY

db = "C:\Program Files (x86)\USGS\Phreeqc Interactive 3.1.4-8929\database\sit.dat" #Database for lab computer
#db="D:\Junction\Program Files (x86)\USGS\Phreeqc Interactive\database\sit.dat" #Database for home computer
tmp = "Montmorillonite 1 site CEC Model\Montmorillonite 1 site CEC model LiteratureSites.txt"
titleString = "Single site model with Cation Exchange, DDL, Na Mont. STX-1"
#x = simulation({'totRa':totRa,'k1':k1,'k2':k2},[2,10],tmp,db)
#x.generateData()
sns.set_palette("deep",n_colors = 6)

#Find experimental data to use
expData = extractData('..\..\Sorption Experiments\Sorption Experiment Master Table.xlsx')
expData = expData.ix[expData.ix[:,'Include?']==True,:] #Select only data that's been vetted
expData = expData.ix[expData.ix[:,'Mineral']=="Sodium Montmorillonite"]

f1 = plt.figure(num=1,figsize=(10,8))
f1.clf()
ax = f1.add_subplot(111)


labelStr = "Cation exchange 1 site, Kint: {Kint}, Ks: {Ks}"

#siteSVal = np.array([1.4E-6]) #mol
#siteWVal = np.array([5.6E-5])
#Ksval = np.arange(0,10.1,0.1)
#Ksval = np.array([7])
#Kwval = np.array([-5.67])
#Kval = np.arange(5.5,6.1,0.1)
#ncol = np.size(siteSVal)*np.size(Ksval)*np.size(Kwval)*np.size(siteWVal)

#siteVal = np.array([1.92E-6])
#Kval = np.array([5.6])
#Kval = np.arange(0,10.1,0.1)
#ncol = np.size(Kval)*np.size(siteVal)

#Clay Paramters
#KsVal = np.arange(-10,11,1)
KintVal = np.array([0.15])
#siteiVal = np.logspace(-8,0,num=9,endpoint=True)
#siteiVal = np.array([2.53E-5]) #Clay value
KsVal = np.arange(6,7.1,0.1)
#KintVal = np.array([-3])
ncol = np.size(KintVal)*np.size(KsVal)

cmap = sns.cubehelix_palette(n_colors=ncol,dark=0.3,rot=0.4,light=0.8,gamma=1.3)
palette = itertools.cycle(cmap)
#
#for K in Kval:
#    for site in siteVal:
#        x = simulation({'totRa':totRa,'k1':K,'sites':site},tmp,db)
#        x.generateData()
#        x.addDataToMaster(writeMaster=True)
#        simRes = x.getData()
#        ax.plot(simRes.ix[:,'pH'],simRes.ix[:,'fSorb'],'-',label=labelStr.format(k=K,site=site),color=next(palette))
pos = 0.0
for Kint in KintVal:
    for Ks in KsVal:
        x = simulation({'totRa':totRa,'K_int':Kint,'Ks':Ks},tmp,db)
        x.generateData()
        x.addDataToMaster(writeMaster=True)
        simRes = x.getData()
        ax.plot(simRes.ix[:,'pH'],simRes.ix[:,'fSorb'],'-',label=labelStr.format(Kint=Kint,Ks=Ks),color=next(palette))
        pos = pos+1
        per = pos/ncol
        print '{:.2%}'.format(per)

        
#Plot all of the data without differentiation
#expPlot = ax.errorbar(expData.ix[:,'pH'],expData.ix[:,'fSorb'],xerr=expData.ix[:,'spH'],yerr=expData.ix[:,'sfSorb'],fmt='o',label='Experimental Data')

exp5 = expData.ix[expData['Total Activity']==5,:]
exp10 = expData.ix[expData['Total Activity']==10,:]
exp50 = expData.ix[expData['Total Activity']==50,:]
exp100 = expData.ix[expData['Total Activity']==100,:]
exp500 = expData.ix[expData['Total Activity']==500,:]

if not exp5.empty:
    exp5Plot = ax.errorbar(exp5.ix[:,'pH'].values,exp5.ix[:,'fSorb'].values,xerr=exp5.ix[:,'spH'].values,yerr=exp5.ix[:,'sfSorb'].values,fmt='o',label='Experimental Data 5 Bq Total')
if not exp10.empty:
    exp10Plot = ax.errorbar(exp10.ix[:,'pH'].values,exp10.ix[:,'fSorb'].values,xerr=exp10.ix[:,'spH'].values,yerr=exp10.ix[:,'sfSorb'].values,fmt='o',label='Experimental Data 10 Bq Total')
if not exp50.empty:
    exp50Plot = ax.errorbar(exp50.ix[:,'pH'].values,exp50.ix[:,'fSorb'].values,xerr=exp50.ix[:,'spH'].values,yerr=exp50.ix[:,'sfSorb'].values,fmt='o',label='Experimental Data 50 Bq Total')
if not exp100.empty:
    exp100Plot = ax.errorbar(exp100.ix[:,'pH'].values,exp100.ix[:,'fSorb'].values,xerr=exp100.ix[:,'spH'].values,yerr=exp100.ix[:,'sfSorb'].values,fmt='o',label='Experimental Data 100 Bq Total')
if not exp500.empty:
    exp500Plot = ax.errorbar(exp500.ix[:,'pH'].values,exp500.ix[:,'fSorb'].values,xerr=exp500.ix[:,'spH'].values,yerr=exp500.ix[:,'sfSorb'].values,fmt='o',label='Experimental Data 500 Bq Total')


ax.legend(loc=0)
ax.set_title(titleString)
ax.set_xlabel('pH')
ax.set_ylabel('Fraction Sorbed')
ax.set_ylim([-0.01,1.0])
plt.show()
