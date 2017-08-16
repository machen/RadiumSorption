# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 17:44:02 2016

@author: Michael
"""

#Script to plot and generate surface complexation results. Will read in master table, but also will duplicate results...

import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
from win32com.client import Dispatch
from string import Template
import itertools, os, copy

#New script will allow user to run a series of PHREEQC calculation using a template file, allowing for iteration over multiple values

class simulation:
    def __init__(self,parameters,templateFile,database,pHrange=[2,10],masterTable=None,check=False):
        self.param = parameters #Dictionary of parameters, which includes everything you want to input to the template
        self.pHrange = pHrange #Range of pHs each simulation should run over
        with open(templateFile,'r') as tempFile:
            self.templStr = Template(tempFile.read()) #Template allows for easy pythonic substitution of the model parameters into a given run of the model (template files formatted assuming this is used)
        self.templFile = templateFile #Location of the template, which is the key linked to the master table of data that have already been run
        self.db = database #Thermodynamic database to use for simulations
        if masterTable is not None:
            self.masterTable = masterTable #Can preload the master table, and keep it updated during execution instead of having to read/write each time
        else:
            self.masterTable = self.loadMaster() #Loads master table into memory
        self.data = pd.DataFrame() #Should only contian data that are specified by the parameters and pH range
        self.dataFromMaster = check

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
                fSorb = (inputParam['totRa']-subRes.loc[:,'Ra(mol/kgw)'].values)/inputParam['totRa']
                Rd = (inputParam['totRa']-subRes.loc[:,'Ra(mol/kgw)'].values)/subRes.loc[:,'Ra(mol/kgw)'].values
                subRes['fSorb'] = pd.Series(fSorb,index=subRes.index)
                subRes['Rd'] = pd.Series(Rd,index=subRes.index)
            else:
                subRes = chk #If the data is there, append the result that would have come out of running PHRREQC instead. Note that I assume that a given simulation instance may have both already run and unrun data
                self.dataFromMaster=True
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
                matchData = matchData.loc[abs(matchData.loc[:,key]-params[key])<1E-16,:]
            return matchData
        else: #Returns empty dataframe if no match
            return matchData
    def addDataToMaster(self,writeMaster=False):
        #Write master is a USER choice whether or not the simulation should attempt to save its results, dataFromMaster prevents the simulation from "double writing" data when it's drawing the data from the masterTable (see self.checkMaster())
        if  self.dataFromMaster:
            return
        #Save data into master table of results. Should include all parameters
        newData = self.data
        n=len(newData.index) 
        #iterate through the keys in the parameters, appending a matrix of the values repeated for each entry
        for key in self.param:
            newData[key] = pd.Series(np.ones(n)*self.param[key], index=newData.index)
        newMaster = pd.concat([newData,self.masterTable],ignore_index=True)
        self.masterTable = newMaster
        if writeMaster:
            newMaster.drop_duplicates(keep='first',subset=self.param.keys().append('pH')).to_csv(self.templFile[:-4]+'.csv',index=False) #Save this to a file for future usage, uses PD drop_duplicates with the params and pH to remove duplicate entries
    def plotSpeciation(self,titleStr='',solidTag="m_Fhy"):
        f2 = plt.figure(2)
        f2.clf()
        ax2 = f2.add_subplot(111)
        totRa = self.param['totRa']
        cmap = sns.color_palette("hls",n_colors=16)
        color = itertools.cycle(cmap)
        color2 = itertools.cycle(cmap)
        for colName in self.data.columns:
            if colName.startswith(solidTag) and "Ra" in colName: #m_Fhy is a marker for solid species
                ax2.plot(self.data.loc[:,'pH'].values,self.data.loc[:,colName].values/totRa,'--',label=colName,color=next(color))
            elif colName.startswith("m_Ra"): #m_Ra is a marker for solution species
                ax2.plot(self.data.loc[:,'pH'].values,self.data.loc[:,colName].values/totRa,'-',label=colName,color=next(color2))
            else: #Do not plot things that are not part of the speciation
                continue
        #ax2.plot(self.data.loc[:,'pH'],self.data.loc[:,'fSorb'],'-k',label='Fraction Sorbed')
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
    
    
#Plotting
sns.set_context('poster')
sns.set_style("ticks",rc={"font.size":48})
mpl.rcParams["lines.markeredgewidth"] = 2
mpl.rcParams["markers.fillstyle"] = "none"
mpl.rcParams["errorbar.capsize"] = 5
mpl.rcParams["lines.linewidth"] = 1
mpl.rcParams["lines.markersize"] = 20
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["figure.figsize"] = [16,14]       
       
specAct = 6.02E23*np.log(2)/(1600*365*24*60*60) #Gets Bq/mol       
totRa = 100/specAct/0.1 #Mol/L Ra-226

db = "C:\Program Files (x86)\USGS\Phreeqc Interactive 3.1.4-8929\database\sit.dat" #Database for lab computer
#db="F:\Programs\USGS\Phreeqc Interactive 3.3.12-12704\database\sit.dat" #Database for home computer
tmp = "Montmorillonite 2 Site CEC Model\Montmorillonite BaeyensBradbury RealSA.txt" #Location of template file to use as the input file for PHREEQC, changing this means changing the x.simulation inputs, as well as the figure legend plot strings
titleString = "Sodium Montmorillonite by Baeyens and Bradbury"
sns.set_palette("deep",n_colors = 6)

#Find experimental data to use
expData = extractData('..\..\Sorption Experiments\Sorption Experiment Master Table.xlsx')
expData = expData.loc[expData.loc[:,'Include?']==True,:] #Select only data that's been vetted
expData = expData.loc[expData.loc[:,'Mineral']=="Sodium Montmorillonite"]
expData = expData.loc[expData.loc[:,'Salt']=='NaCl'] #Want to fit isotherm data at first
expData = expData.loc[expData.loc[:,"Ionic Strength (meq/L)"]==10] #Want to fit isotherm data first
expData = expData.loc[abs(expData.loc[:,"MinMass (g)"].values-0.030)<0.01,:]

resData = extractData('..\..\Sorption Experiments\Isotherm Results.xlsx')
resData = resData.loc[resData.loc[:,'mineral']=='montmorillonite']

#MAIN SCRIPT PLOTTING

f1 = plt.figure(num=1,figsize=(10,8))
f1.clf()
ax = f1.add_subplot(111)

f3 = plt.figure(num=3,figsize=(10,8))
f3.clf()
ax3 = f3.add_subplot(111)


pos = 0.0

K1Val = np.array([6.6])
#K1Val = np.arange(5.0,7.1,0.1)

K2Val =np.array([0.5])
#K2Val = np.arange(0.0,2.1,0.1)

K3Val = np.array([0.2])
#K3Val = np.arange(0.1,0.21,0.01)

#siteVal = np.array([1.40E-6]) 
siteVal = np.array([6E-8])
#siteVal = np.arange(1.35E-6,1.44E-6,1E-8)
#siteVal = np.logspace(-10,-2,num=9,endpoint=True)
siteWVal = np.array([1.2E-6])
#siteWVal = np.logspace(-10,-2,num=9,endpoint=True)
siteWbVal = np.array([1.2E-6])
ncol = np.size(K1Val)*np.size(K2Val)*np.size(siteVal)*np.size(K3Val)*np.size(siteWVal)*np.size(siteWbVal)

cmap = sns.cubehelix_palette(n_colors=ncol,dark=0.3,rot=0.4,light=0.8,gamma=1.3,start=1.5)
palette = itertools.cycle(cmap)
labelStr = "3 sites 2 rxn, Strong Site: {siteS} K1: {K1}, Site W: {siteW} K2: {K2}, Site wB: {siteWb}, Ki: {Ki}"
#
#for K in Kval:
#    for site in siteVal:
#        x = simulation({'totRa':totRa,'k1':K,'sites':site},tmp,db)
#        x.generateData()
#        x.addDataToMaster(writeMaster=True)
#        simRes = x.getData()
#        ax.plot(simRes.loc[:,'pH'],simRes.loc[:,'fSorb'],'-',label=labelStr.format(k1=K,sites=site),color=next(palette))
#        pos = pos+1
#        per = pos/ncol
#        print '{:.2%}'.format(per)
for K1 in K1Val:
    for K2 in K2Val:
        for K3 in K3Val:
            for sites in siteVal:   
                for siteW in siteWVal:
                    for siteWb in siteWbVal:
                            x = simulation({'totRa':totRa,'Ks':K1,'siteS':sites,'Kw':K2,'siteW':siteW,'siteWb':siteWb,'Ki':K3},tmp,db)
                            x.generateData()
                            x.addDataToMaster(writeMaster=True)
                            simRes = x.getData()
                            curCol = next(palette)
                            ax.plot(simRes.loc[:,'pH'],simRes.loc[:,'fSorb'],'-',label=labelStr.format(K1=K1,siteS=sites,K2=K2,siteW=siteW,siteWb=siteWb,Ki=K3),color=curCol)
                            ax3.plot(simRes.loc[:,'pH'],simRes.loc[:,'Rd'],'-',label=labelStr.format(K1=K1,siteS=sites,K2=K2,siteW=siteW,siteWb=siteWb,Ki=K3),color=curCol)
                            pos = pos+1
                            per = pos/ncol
                            print '{:.2%}'.format(per)
     
 
#Plot all of the data without differentiation
expPlot = ax.errorbar(expData.loc[:,'pH'],expData.loc[:,'fSorb'],xerr=expData.loc[:,'spH'],yerr=expData.loc[:,'sfSorb'],fmt='o',label='Experimental Data')
resPlot = ax3.errorbar(resData.loc[:,'pH'],resData.loc[:,'Kd (mL/g)'],xerr=resData.loc[:,'spH'],yerr=resData.loc[:,'sKd (mL/g)'],fmt='.',label='Fitted Kds')
additionalPlot = ax3.plot(7,1443.032,label="Sajih",ls='None',marker='o')

handles, labels = ax.get_legend_handles_labels()
newHandles = []
for h in handles:
    try:
        newHandles.append(h[0])
    except TypeError:
        newHandles.append(h)
        continue
ax.legend(newHandles,labels,loc=0,numpoints=1)

#ax.legend(loc=0)
ax.set_title(titleString)
ax.set_xlabel('pH')
ax.set_ylabel('Fraction Sorbed')
ax.set_ylim([-0.1,1.1])
sns.despine()
plt.show()

ax3.set_title(titleString)
ax3.set_xlabel('pH')
ax3.set_ylabel('Rd (mL/g)')
ax3.set_yscale('linear')
ax3.set_ylim(-100,500000)
ax3.legend(loc=0)

x.plotSpeciation(solidTag="m_Clay_")