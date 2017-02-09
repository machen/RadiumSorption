# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 13:53:02 2017

@author: mache
"""

import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns

def importData(filePath): #Import function to split simulation and experimental data
    fullData = pd.read_excel(filePath)
    simData = fullData.ix[:,0:2]
    expData = fullData.ix[:,2:].dropna()
    return simData, expData

#Step 0: Set plotting behavior
mpl.rcParams["lines.markeredgewidth"] = 1.5
mpl.rcParams["markers.fillstyle"] = "full"
mpl.rcParams["errorbar.capsize"] = 2
mpl.rcParams["lines.linewidth"] = 1
mpl.rcParams["lines.markersize"] = 6
mpl.rcParams["svg.fonttype"] = "none"

mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12            
mpl.rcParams["xtick.major.pad"] =5 
mpl.rcParams["ytick.major.pad"] = 5
mpl.rcParams["axes.titlesize"] = 14
            
mpl.rcParams["font.size"] = 12
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.labelpad"] = 5
mpl.rcParams["figure.autolayout"] = True


#Step 1: Import the data you want to plot

plt.close("all")

fhySim, fhyExp = importData("Figure5a-FHYTetradentateData.xlsx")
goeSim, goeExp = importData("Figure5b-GOE Tetradentate Data.xlsx")

f1, ax1 = plt.subplots(2,1,sharex='col',figsize=(3.33,4.5))

ax1[0].plot(fhySim.ix[:,"pH"].values,fhySim.ix[:,"fSorb"].values,ls="-",color = 'k',label = "Simulation")
ax1[0].errorbar(fhyExp.ix[:,"pH (data)"].values,fhyExp.ix[:,"fSorb (data)"],xerr = fhyExp.ix[:,"spH (data)"].values,yerr=fhyExp.ix[:,"sfSorb (data)"],color='k',marker = ".",ls="none",label = "Experiment")
ax1[0].set_xlim([2,10])
ax1[0].set_ylim([-0.1,1.1])
ax1[0].legend(loc=0)
ax1[0].set_title("Ferrihydrite")

ax1[1].plot(goeSim.ix[:,"pH"].values,goeSim.ix[:,"fSorb"].values,ls="-",color = 'k',label = "Simulation")
ax1[1].errorbar(goeExp.ix[:,"pH (data)"].values,goeExp.ix[:,"fSorb (data)"],xerr = goeExp.ix[:,"spH (data)"].values,yerr=goeExp.ix[:,"sfSorb (data)"],color='k',marker = ".",ls="none",label = "Experiment")
ax1[1].set_xlim([2,10])
ax1[1].set_ylim([-0.1,1.1])
ax1[1].legend(loc=0)
ax1[1].set_title("Goethite")
ax1[1].set_xlabel("pH")
ax1[0].set_ylabel("Fraction Sorbed (.)")

for i in [0,1]:
        handles, labels = ax1[i].get_legend_handles_labels()
        handles = [handles[0],handles[1][0]]
        ax1[i].legend(handles,labels,loc=0)
        
plt.savefig("Figure2-FHYGoeSCM.svg",dpi=1000)