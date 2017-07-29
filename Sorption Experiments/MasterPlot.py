# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 16:47:14 2016

@author: Michael
"""

"""SECTION 1: IMPORT MODULES AND DATA, SETUP DATA FRAMES"""

import pandas as pd, numpy as np, matplotlib as mpl
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import linregress
sns.set_context('poster')
sns.set_style("ticks",rc={"font.size":48})
data = pd.read_excel("Sorption Experiment Master Table.xlsx",header=0)
data = data.ix[data.ix[:,"Include?"]==True,:] #Toggle if you only want to plot data flagged to include, sets up to have all data
              
#Need to also now sort by solution
dataIsotherm = data.ix[data.ix[:,"Salt"]=="NaCl",:] #Selecting the data for the isotherms (only in NaCl)

dataMultiSalinity = data.ix[abs(data.ix[:,"TotAct"]-64.0)<20.0,:] #Select all data that is near the total activity of the mixed results
dataMultiSalinity = dataMultiSalinity.ix[abs(dataMultiSalinity.ix[:,"pH"]-7.0)<0.2,:] #Further select down the data to only include data with similar pH
dataMultiSalinity = dataMultiSalinity.sort_values(by="Salt") #sort the table by salt to make it easier to select on                                                
                                                 
"""SECTION 2: ISOTHERM PLOTS"""
              
#Make Mineral Specific dataframes for each isotherm

FHYdata = dataIsotherm.ix[dataIsotherm.ix[:,'Mineral']=="Ferrihydrite",:]
montData = dataIsotherm.ix[dataIsotherm.ix[:,'Mineral']=='Sodium Montmorillonite']
goeData = dataIsotherm.ix[dataIsotherm.ix[:,'Mineral']=='Goethite']
pyrData = dataIsotherm.ix[dataIsotherm.ix[:,'Mineral']=='Pyrite']
glassData = dataIsotherm.ix[dataIsotherm.ix[:,'Mineral']=='None']

SA = {"fhy":382.9,"goe":146.46,"namont":50.162,"pyr":0.0685}

#QUICKLY NEED TO CONVERT GRAPHS TO pCi/L / pCi/g
#FHYdata.ix[:,2:6] = FHYdata.ix[:,2:6]*27.027*1000
#montData.ix[:,2:6] = montData.ix[:,2:6]*27.027*1000
#goeData.ix[:,2:6] = goeData.ix[:,2:6]*27.027*1000
#pyrData.ix[:,2:6] = pyrData.ix[:,2:6]*27.027*1000


plt.close("all") #Close all open figures

#Set plotting behavior here
mpl.rcParams["figure.figsize"] = [6.66 ,5]
mpl.rcParams["lines.markeredgewidth"] = 1.5
mpl.rcParams["markers.fillstyle"] = "none"
mpl.rcParams["errorbar.capsize"] = 4
mpl.rcParams["lines.linewidth"] = 1.5
mpl.rcParams["lines.markersize"] = 12
mpl.rcParams["svg.fonttype"] = "none"

mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12            
mpl.rcParams["axes.titlesize"] = 14
            
mpl.rcParams["font.size"] = 12
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.labelpad"] = 5
mpl.rcParams["figure.autolayout"] = True



#Plot of Isotherms separated by pH, along with isotherm fits

f3, ax3 = plt.subplots(2,2,sharex='col',figsize=(7.3,4.5))
f4 = plt.figure(4,figsize=(3.33,5))
ax4 = f4.add_subplot(111) #pH 7, all minerals
f5 = plt.figure(5) #Ferrihydrite, all pH values
ax5 = f5.add_subplot(111)
f6 = plt.figure(6) #Mont, all pH values
ax6 = f6.add_subplot(111)
f7 = plt.figure(7) #Geothite, all pH values
ax7 = f7.add_subplot(111)
f8 = plt.figure(8)
ax8 = f8.add_subplot(111)
f9 = plt.figure(9)
ax9 = f9.add_subplot(111)
pHvals  = [3,5,7,9]
markerStyles = ['o','^','s','p']
lineStyles = ["-","--","-.",":"]
fhyPal = sns.color_palette("Reds_d",4)
montPal = sns.color_palette("Greens_d",4)
goePal = sns.color_palette("Oranges_d",4)
pyrPal = sns.color_palette("Purples_d",4)
fhyPal.reverse()
montPal.reverse()
goePal.reverse()
pyrPal.reverse()


for i in range(len(pHvals)):
    pH = pHvals[i]
    pHs = str(pH)
    fhySub = FHYdata.ix[abs(FHYdata.ix[:,'pH']-pH)<0.2,:]
    montSub = montData.ix[abs(montData.ix[:,'pH']-pH)<0.2,:]
    goeSub = goeData.ix[abs(goeData.ix[:,'pH']-pH)<0.2,:]
    pyrSub = pyrData.ix[abs(pyrData.ix[:,'pH']-pH)<0.2,:]
    glassSub = glassData.ix[abs(glassData.ix[:,'pH']-pH)<0.2,:]
    xlim = [-0.5,4.0]
    ylim = [-100,14000]
    if not fhySub.empty:
        Cw = fhySub.ix[:,'Cw (Bq/mL)'].values
        Cs = fhySub.ix[:,'Cs (Bq/g)'].values
        sCw = fhySub.ix[:,'sCw (Bq/mL)'].values
        sCs = fhySub.ix[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)
        ax3[0,0].plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color=fhyPal[i])
        ax3[0,0].errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],label='pH: {}'.format(pH),ls='None',color=fhyPal[i])
        ax3[0,0].legend(loc=0)
        ax3[0,0].set_title('Ferrihydrite')
        ax3[0,0].set_xlim(xlim)
        ax3[0,0].set_ylim(ylim)
        ax3[0,0].set_ylabel('Cs (Bq/g)')
        if pH == 7:
            ax4.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[0],label=None,color='black')
            ax4.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[0],label='Ferrihydrite Kd: {:.2f} R2: {:.2f}'.format(slope,rval**2),ls='None',color='black')
        ax5.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color='black')
        ax5.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],label='pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),ls='None',color='black')
        ax5.set_xlim(xlim)
        ax5.set_ylim(ylim)
    if not montSub.empty:
        Cw = montSub.ix[:,'Cw (Bq/mL)'].values
        Cs = montSub.ix[:,'Cs (Bq/g)'].values
        sCw = montSub.ix[:,'sCw (Bq/mL)'].values
        sCs = montSub.ix[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)
        ax3[1,0].plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color=montPal[i])
        ax3[1,0].errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],ls="none",label='pH: {}'.format(pH),color=montPal[i],elinewidth=1.5)
        ax3[1,0].legend(loc=0)
        ax3[1,0].set_title('Sodium Montmorillonite')
        ax3[1,0].set_xlim(xlim)
        ax3[1,0].set_ylim(ylim)
        ax3[1,0].set_xlabel('Cw (Bq/mL)')
        ax3[1,0].set_ylabel('Cs (Bq/g)')
        if pH == 7:
            ax4.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[1],label=None,color='black')
            ax4.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[1],label='Sodium Montmorillonite Kd: {:.2f} R2: {:.2f}'.format(slope,rval**2),ls='None',color='black')
        ax6.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color='black')
        ax6.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],label='pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),ls='None',color='black')
        ax6.set_xlim(xlim)
        ax6.set_ylim(ylim)
    if not goeSub.empty:
        Cw = goeSub.ix[:,'Cw (Bq/mL)'].values
        Cs = goeSub.ix[:,'Cs (Bq/g)'].values
        sCw = goeSub.ix[:,'sCw (Bq/mL)'].values
        sCs = goeSub.ix[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)
        ax3[0,1].plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color=goePal[i])
        ax3[0,1].errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],ls='None',label='pH: {}'.format(pH),color=goePal[i],elinewidth=1.5)
        ax3[0,1].set_title('Goethite')
        ax3[0,1].legend(loc=0)
        ax3[0,1].set_xlim(xlim)
        ax3[0,1].set_ylim(ylim)
        if pH == 7:
            ax4.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[2],label=None,color='black')
            ax4.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[2],label='Goethite Kd: {:.2f} R2: {:.2f}'.format(slope,rval**2),ls='None',color='black')
        ax7.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color='black')
        ax7.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],label='pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),ls='None',color='black')
        ax7.set_xlim(xlim)
        ax7.set_ylim(ylim)
    if not pyrSub.empty:
        Cw = pyrSub.ix[:,'Cw (Bq/mL)'].values
        Cs = pyrSub.ix[:,'Cs (Bq/g)'].values
        sCw = pyrSub.ix[:,'sCw (Bq/mL)'].values
        sCs = pyrSub.ix[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)
        ax3[1,1].plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color=pyrPal[i])
        ax3[1,1].errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],ls='None',label='pH: {}'.format(pH),color=pyrPal[i],elinewidth=1.5)
        ax3[1,1].set_title('Pyrite')
        ax3[1,1].legend(loc=0)
        ax3[1,1].set_xlim(xlim)
        ax3[1,1].set_ylim(ylim)
        ax3[1,1].set_xlabel('Cw (Bq/mL)')
        if pH == 7:
            ax4.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[3],label=None,color='black')
            ax4.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[3],label='Pyrite Kd: {:.2f} R2: {:.2f}'.format(slope,rval**2),ls='None',color='black')
        ax8.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color='black')
        ax8.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],label='pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),ls='None',color='black')
        ax8.set_xlim(xlim)
        ax8.set_ylim(ylim)
    if not glassSub.empty:
        Cw =glassSub.ix[:,'Cw (Bq/mL)'].values
        fSorb = glassSub.ix[:,'fSorb'].values
        sCw = glassSub.ix[:,'sCw (Bq/mL)'].values
        sfSorb = glassSub.ix[:,'sfSorb'].values
        ax9.errorbar(Cw,fSorb,xerr=sCw,yerr=sfSorb,marker=markerStyles[i],label='pH: {}'.format(pH),ls='None',color='black')
                           

#Strip out the errorbars, despine

for i in [0,1]:
    for j in [0,1]:
        handles, labels = ax3[i,j].get_legend_handles_labels()
        handles = [h[0] for h in handles]
        ax3[i,j].legend(handles,labels,loc=0)

handles, labels = ax4.get_legend_handles_labels()
handles = [h[0] for h in handles]
labels = ['Ferrihydrite','Na Montmorillonite','Goethite','Pyrite']
ax4.legend(handles,labels,loc=0,numpoints=1)            
ax4.set_xlabel('Cw (Bq/mL)')
ax4.set_ylabel('Cs (Bq/g)')

handles, labels = ax5.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax5.legend(handles,labels,loc=0,numpoints=1)
ax5.set_xlabel('Cw (Bq/mL)')
ax5.set_ylabel('Cs (Bq/g)')    

handles, labels = ax6.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax6.legend(handles,labels,loc=0,numpoints=1)
ax6.set_xlabel('Cw (Bq/mL)')
ax6.set_ylabel('Cs (Bq/g)')   

handles, labels = ax7.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax7.legend(handles,labels,loc=0,numpoints=1)
ax7.set_xlabel('Cw (Bq/mL)')
ax7.set_ylabel('Cs (Bq/g)')    

handles, labels = ax8.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax8.legend(handles,labels,loc=0,numpoints=1)
ax8.set_xlabel('Cw (Bq/mL)')
ax8.set_ylabel('Cs (Bq/g)')   

handles, labels = ax9.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax9.legend(handles,labels,loc=0,numpoints=1)
ax9.set_xlabel('Cw (Bq/mL)')
ax9.set_ylabel('Fraction sorbed (.)')
#plt.xlabel('Cw (Bq/mL)')
#plt.ylabel('Cs (Bq/g)')
#plt.title('Sorption Isotherms')
#plt.legend(loc=0)
ax5.set_title("Ferrihydrite Isotherms")
ax6.set_title("Montmorillonite Isotherms")
ax7.set_title("Goethite Isotherms")
sns.despine(f4)
sns.despine(f5)
sns.despine(f6)
sns.despine(f7)
sns.despine(f8)
ax8.set_title("Pyrite Isotherms")

ax9.set_title("Glass Sorption Control")
sns.despine(f9)

"""SECTION 3: SALINITY PLOTS"""


#Data for each mineral

valsFHY = dataMultiSalinity.ix[dataMultiSalinity.ix[:,'Mineral']=='Ferrihydrite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)
valsGOE = dataMultiSalinity.ix[dataMultiSalinity.ix[:,'Mineral']=='Goethite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)
valsMont = dataMultiSalinity.ix[dataMultiSalinity.ix[:,'Mineral']=='Sodium Montmorillonite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)
valsPYR = dataMultiSalinity.ix[dataMultiSalinity.ix[:,'Mineral']=='Pyrite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)


##PLOT 1: Plot, by mineral, "Kd" (Cs/Cw) vs. Ionic Strength

KdIFig, KdIAx = plt.subplots()
I = valsFHY.ix[:,'Ionic Strength (meq/L)'].values
K = valsFHY.ix[:,'Cs (Bq/g)'].values/valsFHY.ix[:,'Cw (Bq/mL)'].values #Units of mL/g
sK = K*np.sqrt((valsFHY.ix[:,'sCs (Bq/g)']/valsFHY.ix[:,'Cs (Bq/g)'])**2+(valsFHY.ix[:,'sCw (Bq/mL)']/valsFHY.ix[:,'Cw (Bq/mL)'])**2)    #units of mL/g
KdIAx.errorbar(I,K,yerr=sK,marker='.',label='Ferrihydrite',ls='None',color=fhyPal[1])

I = valsGOE.ix[:,'Ionic Strength (meq/L)'].values
K = valsGOE.ix[:,'Cs (Bq/g)'].values/valsGOE.ix[:,'Cw (Bq/mL)'].values #Units of mL/g
sK = K*np.sqrt((valsGOE.ix[:,'sCs (Bq/g)']/valsGOE.ix[:,'Cs (Bq/g)'])**2+(valsGOE.ix[:,'sCw (Bq/mL)']/valsGOE.ix[:,'Cw (Bq/mL)'])**2)    #units of mL/g
KdIAx.errorbar(I,K,yerr=sK,marker='.',label='Goethite',ls='None',color=goePal[1])

I = valsMont.ix[:,'Ionic Strength (meq/L)'].values
K = valsMont.ix[:,'Cs (Bq/g)'].values/valsMont.ix[:,'Cw (Bq/mL)'].values #Units of mL/g
sK = K*np.sqrt((valsMont.ix[:,'sCs (Bq/g)']/valsMont.ix[:,'Cs (Bq/g)'])**2+(valsMont.ix[:,'sCw (Bq/mL)']/valsMont.ix[:,'Cw (Bq/mL)'])**2)    #units of mL/g
KdIAx.errorbar(I,K,yerr=sK,marker='.',label='Sodium Montmorillonite',ls='None',color=montPal[1])

I = valsPYR.ix[:,'Ionic Strength (meq/L)'].values
K = valsPYR.ix[:,'Cs (Bq/g)'].values/valsPYR.ix[:,'Cw (Bq/mL)'].values #Units of mL/g
sK = K*np.sqrt((valsPYR.ix[:,'sCs (Bq/g)']/valsPYR.ix[:,'Cs (Bq/g)'])**2+(valsPYR.ix[:,'sCw (Bq/mL)']/valsPYR.ix[:,'Cw (Bq/mL)'])**2)    #units of mL/g
KdIAx.errorbar(I,K,yerr=sK,marker='.',label='Pyrite',ls='None',color=pyrPal[1])

KdIAx.set_yscale('log')
KdIAx.set_xscale('log')
KdIAx.legend(loc=0)
KdIAx.set_title("Ionic Strength vs K")

##PLOT 2: Bars. Requires to have the same number of points for each data value.


saltLabels =valsFHY.ix[:,'Salt'].unique()
groups = len(saltLabels)

fBar, axBar = plt.subplots()
index = np.arange(groups)
bar_width =0.1

fhyBar = plt.bar(index-1.5*bar_width,valsFHY.ix[:,'fSorb'],bar_width,yerr = valsFHY.ix[:,'sfSorb'],alpha = 1, color = fhyPal[1],label = 'Ferrihydrite')
goeBar = plt.bar(index-0.5*bar_width,valsGOE.ix[:,'fSorb'],bar_width,yerr = valsGOE.ix[:,'sfSorb'],alpha = 1, color = goePal[1],label = 'Goethite')
montBar = plt.bar(index+0.5*bar_width,valsMont.ix[:,'fSorb'],bar_width,yerr = valsMont.ix[:,'sfSorb'],alpha = 1, color = montPal[1],label = 'Sodium Montmorillonite')
#pyrBar = plt.bar(index+1.5*bar_width,valsPYR.ix[:,'fSorb'],bar_width,yerr = valsPYR.ix[:,'sfSorb'],alpha = 1, color = pyrPal[1],label = 'Pyrite') #need more data points

axBar.set_ylabel('Fraction Sorbed')
axBar.set_title("Salinity testing")
axBar.set_xticks(index)
axBar.set_xticklabels(saltLabels)

axBar.legend(loc=0)

#PLOT 3: IONIC STRENGTH VS FSORB

sorbIFig, sorbIAx = plt.subplots()
I = valsFHY.ix[:,'Ionic Strength (meq/L)'].values
K = valsFHY.ix[:,'fSorb'].values #Dimensionless units
sK = valsFHY.ix[:,'sfSorb'].values #Dimensionless units
sorbIAx.errorbar(I,K,yerr=sK,marker='.',label='Ferrihydrite',ls='None',color=fhyPal[1])

I = valsGOE.ix[:,'Ionic Strength (meq/L)'].values
K = valsGOE.ix[:,'fSorb'].values #Dimensionless units
sK = valsGOE.ix[:,'sfSorb'].values #Dimensionless units
sorbIAx.errorbar(I,K,yerr=sK,marker='.',label='Goethite',ls='None',color=goePal[1])

I = valsMont.ix[:,'Ionic Strength (meq/L)'].values
K = valsMont.ix[:,'fSorb'].values #Dimensionless units
sK =valsMont.ix[:,'sfSorb'].values #Dimensionless units
sorbIAx.errorbar(I,K,yerr=sK,marker='.',label='Sodium Montmorillonite',ls='None',color=montPal[1])

I = valsPYR.ix[:,'Ionic Strength (meq/L)'].values
K = valsPYR.ix[:,'fSorb'].values #Dimensionless units
sK =valsPYR.ix[:,'sfSorb'].values #Dimensionless units
sorbIAx.errorbar(I,K,yerr=sK,marker='.',label='Pyrite',ls='None',color=pyrPal[1])

sorbIAx.set_xscale('log')
sorbIAx.legend(loc=0)
sorbIAx.set_title("Ionic Strength vs fSorb")



"""SECTION 4: SAVING EVERYTHING"""

plt.show()
#f3.savefig('..\\Manuscript\\Figures\\Sorption Isotherms.svg',dpi=1000)
f3.savefig('MasterTablePlots\\Sorption Isotherms All Valid Data.svg', dpi=1000)
#f4.savefig('..\\Manuscript\\Figures\\Figure1-pH7Isotherms.svg',dpi=1000)
f5.savefig('MasterTablePlots\\IsothermsFHY.svg',dpi=1000)
f6.savefig('MasterTablePlots\\IsothermsNaMont.svg',dpi=1000)
f7.savefig('MasterTablePlots\\IsothermsGOE.svg',dpi=1000)
f8.savefig('MasterTablePlots\\IsothermsPYR.svg',dpi=1000)
#f9.savefig('MasterTablePlots\\KineticsNaMont.svg',dpi=1000)
