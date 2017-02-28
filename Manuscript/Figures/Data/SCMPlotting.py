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
sns.set_style("white")
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
montSim, montExp = importData("Figure6b-Mont 2 site CEC Model Data.xlsx")
pyrSim, pyrExp = importData("Figure7-Pyrite 1 site DDL Deprotonated Site Data.xlsx")

#Step 2: Plot it all, notably, simulation data and error bar data

#F1: Ferrihydrite and Gothite

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

#F2: Ferrihydrite and Sodium montmorillonite

f2, ax2 = plt.subplots(2,1,sharex='col',figsize=(3.33,4.5))

ax2[0].plot(fhySim.ix[:,"pH"].values,fhySim.ix[:,"fSorb"].values,ls="-",color = 'k',label = "Simulation")
ax2[0].errorbar(fhyExp.ix[:,"pH (data)"].values,fhyExp.ix[:,"fSorb (data)"],xerr = fhyExp.ix[:,"spH (data)"].values,yerr=fhyExp.ix[:,"sfSorb (data)"],color='k',marker = ".",ls="none",label = "Experiment")
ax2[0].set_xlim([2,10])
ax2[0].set_ylim([-0.1,1.1])
ax2[0].legend(loc=0)
ax2[0].set_title("Ferrihydrite")

ax2[1].plot(montSim.ix[:,"pH"].values,montSim.ix[:,"fSorb"].values,ls="-",color = 'k',label = "Simulation")
ax2[1].errorbar(montExp.ix[:,"pH (data)"].values,montExp.ix[:,"fSorb (data)"],xerr = montExp.ix[:,"spH (data)"].values,yerr=montExp.ix[:,"sfSorb (data)"],color='k',marker = ".",ls="none",label = "Experiment")
ax2[1].set_xlim([2,10])
ax2[1].set_ylim([-0.1,1.1])
ax2[1].legend(loc=0)
ax2[1].set_title("Sodium Montmorillonite")
ax2[1].set_xlabel("pH")
ax2[0].set_ylabel("Fraction Sorbed (.)")

#F3: ALL OF THE FIGURES

f3, ax3 = plt.subplots(2,2,sharex='col',sharey='row',figsize=(7,4.5))

blue = sns.color_palette("Blues",1)[0]
red = sns.color_palette("Reds",1)[0]
purple = sns.color_palette("Purples",1)[0]
green = sns.color_palette("Greens",1)[0]

ax3[0,0].plot(fhySim.ix[:,"pH"].values,fhySim.ix[:,"fSorb"].values,ls="-",color = blue,label = "Simulation")
ax3[0,0].errorbar(fhyExp.ix[:,"pH (data)"].values,fhyExp.ix[:,"fSorb (data)"],xerr = fhyExp.ix[:,"spH (data)"].values,yerr=fhyExp.ix[:,"sfSorb (data)"],color=blue,marker = ".",ls="none",label = "Experiment")
ax3[0,0].set_xlim([2,10])
ax3[0,0].set_ylim([-0.1,1.1])
ax3[0,0].legend(loc=0)
ax3[0,0].set_title("Ferrihydrite")
ax3[0,0].set_ylabel("fSorb")

ax3[0,1].plot(goeSim.ix[:,"pH"].values,goeSim.ix[:,"fSorb"].values,ls="-",color = red,label = "Simulation")
ax3[0,1].errorbar(goeExp.ix[:,"pH (data)"].values,goeExp.ix[:,"fSorb (data)"],xerr = goeExp.ix[:,"spH (data)"].values,yerr=goeExp.ix[:,"sfSorb (data)"],color=red,marker = ".",ls="none",label = "Experiment")
ax3[0,1].set_xlim([2,10])
ax3[0,1].set_ylim([-0.1,1.1])
ax3[0,1].legend(loc=0)
ax3[0,1].set_title("Goethite")

ax3[1,0].plot(montSim.ix[:,"pH"].values,montSim.ix[:,"fSorb"].values,ls="-",color = green,label = "Simulation")
ax3[1,0].errorbar(montExp.ix[:,"pH (data)"].values,montExp.ix[:,"fSorb (data)"],xerr = montExp.ix[:,"spH (data)"].values,yerr=montExp.ix[:,"sfSorb (data)"],color=green,marker = ".",ls="none",label = "Experiment")
ax3[1,0].set_xlim([2,10])
ax3[1,0].set_ylim([-0.1,1.1])
ax3[1,0].legend(loc=0)
ax3[1,0].set_title("Sodium Montmorillonite")

ax3[1,1].plot(pyrSim.ix[:,"pH"].values,pyrSim.ix[:,"fSorb"].values,ls="-",color = purple,label = "Simulation")
ax3[1,1].errorbar(pyrExp.ix[:,"pH (data)"].values,pyrExp.ix[:,"fSorb (data)"],xerr = pyrExp.ix[:,"spH (data)"].values,yerr=pyrExp.ix[:,"sfSorb (data)"],color=purple,marker = ".",ls="none",label = "Experiment")
ax3[1,1].set_xlim([2,10])
ax3[1,1].set_ylim([-0.1,1.1])
ax3[1,1].legend(loc=0)
ax3[1,1].set_title("Pyrite")
ax3[1,1].set_xlabel("pH")

for i in [0,1]:
    handles, labels = ax1[i].get_legend_handles_labels()
    handles = [handles[0],handles[1][0]]
    ax1[i].legend(handles,labels,loc=0)
    handles, labels = ax2[i].get_legend_handles_labels()
    handles = [handles[0],handles[1][0]]
    ax2[i].legend(handles,labels,loc=0)
    for j in [0,1]:
        handles, labels = ax3[i,j].get_legend_handles_labels()
        handles = [handles[0],handles[1][0]]
        ax3[i,j].legend(handles,labels,loc=0)
        
#plt.savefig("Figure2-FHYGoeSCM.svg",dpi=1000)
f2.savefig("Figure2Alt-FHYNaMontSCM.svg",dpi=1000)
f3.savefig("AllSCM.svg",dpi=1000)