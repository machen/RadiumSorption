# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:28:21 2017

@author: Michael
"""
#PHREEQC Script tuned to work specifically on the multiSalinity testing. Scans through K values over different salinity conditions specified by an excel spreadsheet, and finds the minimal RMSE assocated

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
            newMaster.drop_duplicates(keep='first',subset=list(self.param.keys()).append('pH')).to_csv(self.templFile[:-4]+'.csv',index=False) #Save this to a file for future usage, uses PD drop_duplicates with the params and pH to remove duplicate entries
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

def stripErrorBarLegends(axis):
    handles, labels = axis.get_legend_handles_labels()
    newHandles = []
    for h in handles:
        try:
            newHandles.append(h[0])
        except TypeError:
            newHandles.append(h)
            continue
    axis.legend(newHandles,labels,loc=0,numpoints=1)

# Plotting Parameters
sns.set_context('poster')
sns.set_style("ticks",rc={"font.size":48})
mpl.rcParams["lines.markeredgewidth"] = 5
mpl.rcParams["markers.fillstyle"] = "full"
mpl.rcParams["errorbar.capsize"] = 10
mpl.rcParams["lines.linewidth"] = 5
mpl.rcParams["lines.markersize"] = 35
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["figure.figsize"] = [16,14]

#Database Selection

db = "C:\Program Files (x86)\\USGS\\Phreeqc Interactive 3.1.4-8929\\database\\sit.dat" #Database for lab computer
#db="F:\Programs\USGS\Phreeqc Interactive 3.3.12-12704\database\sit.dat" #Database for home computer

#Import the solution conditions to use as a table

expCond = pd.read_excel("SalinityConditions.xlsx",header=0,index_col=0,sheetname='Sheet1')

#Set K values to check or sweep through
K1Val = np.array([-2.5])
#K1Val = np.arange(5.5, 6.1, 0.1)
K2Val = np.array([-6.6])
#K2Val = np.arange(-11.5, -10.9, 0.1)
K3Val = np.array([-9.4])
#K3Val = np.arange(-10.0, -9.1, 0.1)
K4Val = np.array([8.9])
#K4Val = np.arange(-9.2,-8.5,0.1)
templateFile = 'GOE Dzombak\GOE MathurDzombak MultiSalinity.txt'

mineral = 'Goethite'

ncol = len(expCond.index)*len(K1Val)*len(K2Val)*len(K3Val)*len(K4Val)
print(ncol)

#Create data to compare against
expData = extractData('..\..\..\Sorption Experiments\Sorption Experiment Master Table.xlsx')
expData = expData.loc[expData.loc[:,'Include?']==True,:] #Select only data that's been vetted
expData = expData.loc[expData.loc[:,'Mineral']==mineral,:]
expData = expData.loc[abs(expData.loc[:,"TotAct"]-70.0)<30.0,:] #Select all data that is near the total activity of the mixed results
expData = expData.loc[abs(expData.loc[:,"pH"]-7.0)<0.2,:] #Further select down the data to only include data with similar pH
expData = expData.loc[abs(expData.loc[:,"MinMass (g)"]-0.03)<0.02,:] #Only use 30 mg experiments to compare salinity tests

pos = 0.0
start =0

plt.close("all") #Clear out current figures

fTestK, axTestK = plt.subplots()

fResK, axResK = plt.subplots()

errors = pd.Panel(items=expCond.index,major_axis=np.arange(pos+1,(ncol/len(expCond.index))+1,1.0),minor_axis=['K1','K2','err2','fSorb'])

for cond in expCond.index:
    cmap = sns.cubehelix_palette(n_colors=ncol/len(expCond.index),dark=0.6,rot=0.05,light=0.4,gamma=1.0,start=start) #Set each experimental condition to have its own color in the cubehelix palette, with the variations in fitted parameters forming a slow gradient in color
    palette = itertools.cycle(cmap)
    start+=3.0/len(expCond.index)
    simNum = 1.0 #Number for tracking the number of simulations in which the K values are varied
    for K1 in K1Val:
        for K2 in K2Val:
            for K3 in K3Val:
                for K4 in K4Val:
                    params = expCond.loc[cond,:].to_dict()
                    params['k1'] = K1
                    params['k2'] = K2
                    #params['k3'] = K3
                    #params['k4'] = K4
                    params['totRa'] = 8E-11
                    x = simulation(params,templateFile,db,pHrange=[7.0,7.0])
                    x.generateData()
                    x.addDataToMaster(writeMaster=True)
                    simRes = x.getData()
                    curCol = next(palette)
                    #Ploting for checking the efficacy of established fits
                    axResK.errorbar(simRes.loc[:,'fSorb'],expData.loc[expData.loc[:,'Salt']==cond,'fSorb'].values,yerr=expData.loc[expData.loc[:,'Salt']==cond,'sfSorb'].values,color=curCol,marker='.',ls='none',label=cond)
                    err = float(simRes.loc[:,'fSorb'].values-expData.loc[expData.loc[:,'Salt']==cond,'fSorb'])
                    pos +=1
                    per = pos/ncol
                    errors.loc[cond,simNum,:] = pd.Series(data = [K1,K2,err**2,float(simRes.loc[:,'fSorb'].values)],index=errors.minor_axis)
                    simNum+=1
                    print('{:.2%}'.format(per))

axResK.plot([0,1.0],[0,1.0],color='k',ls='-') #Plot 1 to 1 line of theoretical vs fitted
stripErrorBarLegends(axResK)
axResK.set_xlabel('Fraction sorbed (simulation)')
axResK.set_ylabel('Fraction sorbed (experimental)')
# Test fit against the best fit produced by simulation so far
bestErrPath = templateFile[:-4]+'BestErrors.csv'
if os.path.isfile(bestErrPath):
    bestErr = pd.read_csv(bestErrPath,header=0,index_col=0)
    minErr = np.sqrt(np.sum(bestErr.loc['err2',:])/len(bestErr.columns))
else:
    bestErr = pd.DataFrame(index=errors.minor_axis,columns=errors.items)
    minErr = 1E10

# Calculate the best fit and plot it
for simIndex in errors.major_axis:
    testRMSE = np.sqrt(np.sum(errors.loc[:,simIndex,'err2'].values)/len(errors.items))
    if testRMSE <= minErr:
        bestErr.loc[:,:] = errors.loc[:,simIndex,:]
        minErr = testRMSE
    else:
        continue
test = 0
mapc = sns.hls_palette(n_colors=8,l=0.5)
colors = itertools.cycle(mapc)
for cond in bestErr.columns:
    curCol = next(colors)
    axTestK.errorbar(bestErr.loc['fSorb',cond],expData.loc[expData.loc[:,'Salt']==cond,'fSorb'].values,yerr=expData.loc[expData.loc[:,'Salt']==cond,'sfSorb'].values,color=curCol,marker='.',ls='none',label=cond)
axTestK.plot([0,1.0],[0,1.0],color='k',ls='-') #Plot 1 to 1 line of theoretical vs fitted    

bestErr.to_csv(templateFile[:-4]+'BestErrors.csv')
stripErrorBarLegends(axTestK)
axTestK.set_xlabel('Fraction sorbed (simulation)')
axTestK.set_ylabel('Fraction sorbed (experimental)')
axTestK.set_title('Best fitted error over all conditions')
axTestK.annotate('RMSE: {R} \nK1: {K1}, K2:{K2}'.format(K1=bestErr.loc['K1','NaCl'],K2=bestErr.loc['K2','NaCl'],R=minErr),xy=[0.7,0.1])

# fTestK.savefig(templateFile[:-4]+' Fitted.png', dpi=600)
# fTestK.savefig('Example.svg', dpi=2400)
plt.show()
