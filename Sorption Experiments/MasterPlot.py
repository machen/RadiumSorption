# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 16:47:14 2016

@author: Michael"""
"""SECTION 1: IMPORT MODULES AND DATA, SETUP DATA FRAMES"""

import pandas as pd, numpy as np, matplotlib as mpl
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import linregress

sns.set_context('poster')
sns.set_style("ticks",rc={"font.size":48})
data = pd.read_excel("Sorption Experiment Master Table.xlsx",header=0)
data.loc[:,"Rd (mL/g)"] = data.loc[:,"Cs (Bq/g)"].values/data.loc[:,'Cw (Bq/mL)'].values #Calculate an Rd value to use in plotting
data.loc[:,"dRd (mL/g)"] = data.loc[:,"Rd (mL/g)"].values*np.sqrt((data.loc[:,"sCs (Bq/g)"].values/data.loc[:,"Cs (Bq/g)"])**2+(data.loc[:,"sCw (Bq/mL)"]/data.loc[:,'Cw (Bq/mL)'])**2)
data = data.loc[data.loc[:,"Include?"]==True,:] #Toggle if you only want to plot data flagged to include, sets up to have all data
              
#Need to also now sort by solution
dataIsotherm = data.loc[data.loc[:,"Salt"]=="NaCl",:] #Selecting the data for the isotherms (only in NaCl), should also select for minMass range

dataMultiSalinity = data.loc[abs(data.loc[:,"TotAct"]-70.0)<30.0,:] #Select all data that is near the total activity of the mixed results
dataMultiSalinity = dataMultiSalinity.loc[abs(dataMultiSalinity.loc[:,"pH"]-7.0)<0.2,:] #Further select down the data to only include data with similar pH
dataMultiSalinity = dataMultiSalinity.loc[abs(dataMultiSalinity.loc[:,"MinMass (g)"]-0.03)<0.02,:] #Only use 30 mg experiments to compare salinity tests
dataMultiSalinity = dataMultiSalinity.sort_values(by="Salt") #sort the table by salt to make it easier to select on        

dataMass = data.loc[data.loc[:,"Salt"].isin(["ABW","ASW","AGW"]),:]
dataMass = dataMass.loc[dataMass.loc[:,"Mineral"]=="Ferrihydrite",:]

#Include some external data for comparison
extData = pd.read_excel("SorptionKdComparison.xlsx",header=1)                                                 
"""SECTION 2: ISOTHERM PLOTS"""
              
#Make Mineral Specific dataframes for each isotherm

FHYdata = dataIsotherm.loc[dataIsotherm.loc[:,'Mineral']=="Ferrihydrite",:]
montData = dataIsotherm.loc[dataIsotherm.loc[:,'Mineral']=='Sodium Montmorillonite']
goeData = dataIsotherm.loc[dataIsotherm.loc[:,'Mineral']=='Goethite']
pyrData = dataIsotherm.loc[dataIsotherm.loc[:,'Mineral']=='Pyrite']
glassData = dataIsotherm.loc[dataIsotherm.loc[:,'Mineral']=='None']

#QUICKLY NEED TO CONVERT GRAPHS TO pCi/L / pCi/g
#FHYdata.loc[:,2:6] = FHYdata.loc[:,2:6]*27.027*1000
#montData.loc[:,2:6] = montData.loc[:,2:6]*27.027*1000
#goeData.loc[:,2:6] = goeData.loc[:,2:6]*27.027*1000
#pyrData.loc[:,2:6] = pyrData.loc[:,2:6]*27.027*1000


plt.close("all") #Close all open figures

#Set plotting behavior here
mpl.rcParams["figure.figsize"] = [6.66 ,5]
mpl.rcParams["figure.dpi"] = 100.0
#mpl.rcParams["figure.dpi"] = 235.0 #USE FOR QHD SCREEN ONLY
mpl.rcParams["lines.markeredgewidth"] = 0
mpl.rcParams["markers.fillstyle"] = "full" #"none"
mpl.rcParams["errorbar.capsize"] = 5
mpl.rcParams["lines.linewidth"] = 1.0
mpl.rcParams["lines.markersize"] = 7
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
minVals = ['fhy','goe','mont','pyr']
newIndex = []

for pH in pHvals:
    for mineral in minVals:
        newIndex.append(mineral+str(pH))

resultsIsotherm = pd.DataFrame(index=newIndex,columns=['Kd (mL/g)','sKd (mL/g)','pH','spH','mineral'])

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

#ISOTHERM PLOTTING. Iterates through pH values to fit lines to get Kd, as well as produces both individual charts, and a chart with 4 subplots with all data

for i in range(len(pHvals)):
    pH = pHvals[i]
    pHs = str(pH)
    fhySub = FHYdata.loc[abs(FHYdata.loc[:,'pH']-pH)<0.2,:].sort_values(by='Cw (Bq/mL)')
    montSub = montData.loc[abs(montData.loc[:,'pH']-pH)<0.2,:].sort_values(by='Cw (Bq/mL)')
    goeSub = goeData.loc[abs(goeData.loc[:,'pH']-pH)<0.2,:].sort_values(by='Cw (Bq/mL)')
    pyrSub = pyrData.loc[abs(pyrData.loc[:,'pH']-pH)<0.2,:].sort_values(by='Cw (Bq/mL)')
    glassSub = glassData.loc[abs(glassData.loc[:,'pH']-pH)<0.2,:].sort_values(by='Cw (Bq/mL)')
    xlim = [-0.5,4.0]
    ylim = [-500,14000]
    if not fhySub.empty:
        Cw = fhySub.loc[:,'Cw (Bq/mL)'].values
        Cs = fhySub.loc[:,'Cs (Bq/g)'].values
        sCw = fhySub.loc[:,'sCw (Bq/mL)'].values
        sCs = fhySub.loc[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)
        fitCs = np.polyval([slope,inter],Cw)
        ax3[0,0].plot(Cw,fitCs,ls=lineStyles[i],label=None,color=fhyPal[i])
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
        resultsIsotherm.loc['fhy'+str(pH),:] = [slope,stdErr,np.mean(fhySub.loc[:,'pH']),np.std(fhySub.loc[:,'pH']),'ferrihydrite']#Write results of slope fitting to new dataframe
        plotRes = pd.DataFrame([],columns=['Cw','Cs','sCw','sCs','fitCs'])
        plotRes.loc[:,'Cw'] = Cw
        plotRes.loc[:,'Cs'] = Cs
        plotRes.loc[:,'sCw'] = sCw
        plotRes.loc[:,'sCs'] = sCs
        plotRes.loc[:,'fitCs'] = fitCs
        plotRes.to_csv('MasterPlotData\\FHYpH{}.csv'.format(pH))
        del plotRes, Cw, Cs, sCw, sCs, fitCs
    if not montSub.empty:
        Cw = montSub.loc[:,'Cw (Bq/mL)'].values
        Cs = montSub.loc[:,'Cs (Bq/g)'].values
        sCw = montSub.loc[:,'sCw (Bq/mL)'].values
        sCs = montSub.loc[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)
        fitCs = np.polyval([slope,inter],Cw)
        ax3[1,0].plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color=montPal[i])
        ax3[1,0].errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],ls="none",label='pH: {}'.format(pH),color=montPal[i],elinewidth=1.5)
        ax3[1,0].legend(loc=0)
        ax3[1,0].set_title('Sodium Montmorillonite')
        ax3[1,0].set_xlim(xlim)
        ax3[1,0].set_ylim(ylim)
        ax3[1,0].set_xlabel('Cw (Bq/mL)')
        ax3[1,0].set_ylabel('Cs (Bq/g)')
        plotRes = pd.DataFrame([],columns=['Cw','Cs','sCw','sCs','fitCs'])
        plotRes.loc[:,'Cw'] = Cw
        plotRes.loc[:,'Cs'] = Cs
        plotRes.loc[:,'sCw'] = sCw
        plotRes.loc[:,'sCs'] = sCs
        plotRes.loc[:,'fitCs'] = fitCs
        plotRes.to_csv('MasterPlotData\\MontpH{}.csv'.format(pH))
        if pH == 7:
            ax4.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[1],label=None,color='black')
            ax4.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[1],label='Sodium Montmorillonite Kd: {:.2f} R2: {:.2f}'.format(slope,rval**2),ls='None',color='black')
        ax6.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color='black')
        ax6.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],label='pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),ls='None',color='black')
        ax6.set_xlim(xlim)
        ax6.set_ylim(ylim)
        resultsIsotherm.loc['mont'+str(pH),:] = [slope,stdErr,np.mean(montSub.loc[:,'pH']),np.std(montSub.loc[:,'pH']),'montmorillonite']
    if not goeSub.empty:
        Cw = goeSub.loc[:,'Cw (Bq/mL)'].values
        Cs = goeSub.loc[:,'Cs (Bq/g)'].values
        sCw = goeSub.loc[:,'sCw (Bq/mL)'].values
        sCs = goeSub.loc[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)
        fitCs = np.polyval([slope,inter],Cw)
        ax3[0,1].plot(Cw,fitCs,ls=lineStyles[i],label=None,color=goePal[i])
        ax3[0,1].errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],ls='None',label='pH: {}'.format(pH),color=goePal[i],elinewidth=1.5)
        ax3[0,1].set_title('Goethite')
        ax3[0,1].legend(loc=0)
        ax3[0,1].set_xlim(xlim)
        ax3[0,1].set_ylim(ylim)
        plotRes = pd.DataFrame([],columns=['Cw','Cs','sCw','sCs','fitCs'])
        plotRes.loc[:,'Cw'] = Cw
        plotRes.loc[:,'Cs'] = Cs
        plotRes.loc[:,'sCw'] = sCw
        plotRes.loc[:,'sCs'] = sCs
        plotRes.loc[:,'fitCs'] = fitCs
        plotRes.to_csv('MasterPlotData\\GOEpH{}.csv'.format(pH))
        if pH == 7:
            ax4.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[2],label=None,color='black')
            ax4.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[2],label='Goethite Kd: {:.2f} R2: {:.2f}'.format(slope,rval**2),ls='None',color='black')
        ax7.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color='black')
        ax7.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],label='pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),ls='None',color='black')
        ax7.set_xlim(xlim)
        ax7.set_ylim(ylim)
        resultsIsotherm.loc['goe'+str(pH),:] = [slope,stdErr,np.mean(goeSub.loc[:,'pH']),np.std(goeSub.loc[:,'pH']),'goethite']
    if not pyrSub.empty:
        Cw = pyrSub.loc[:,'Cw (Bq/mL)'].values
        Cs = pyrSub.loc[:,'Cs (Bq/g)'].values
        sCw = pyrSub.loc[:,'sCw (Bq/mL)'].values
        sCs = pyrSub.loc[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)
        fitCs = np.polyval([slope,inter],Cw)
        ax3[1,1].plot(Cw,fitCs,ls=lineStyles[i],label=None,color=pyrPal[i])
        ax3[1,1].errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],ls='None',label='pH: {}'.format(pH),color=pyrPal[i],elinewidth=1.5)
        ax3[1,1].set_title('Pyrite')
        ax3[1,1].legend(loc=0)
        ax3[1,1].set_xlim(xlim)
        ax3[1,1].set_ylim(ylim)
        ax3[1,1].set_xlabel('Cw (Bq/mL)')
        plotRes = pd.DataFrame([],columns=['Cw','Cs','sCw','sCs','fitCs'])
        plotRes.loc[:,'Cw'] = Cw
        plotRes.loc[:,'Cs'] = Cs
        plotRes.loc[:,'sCw'] = sCw
        plotRes.loc[:,'sCs'] = sCs
        plotRes.loc[:,'fitCs'] = fitCs
        plotRes.to_csv('MasterPlotData\\PYRpH{}.csv'.format(pH))
        if pH == 7:
            ax4.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[3],label=None,color='black')
            ax4.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[3],label='Pyrite Kd: {:.2f} R2: {:.2f}'.format(slope,rval**2),ls='None',color='black')
        ax8.plot(Cw,np.polyval([slope,inter],Cw),ls=lineStyles[i],label=None,color='black')
        ax8.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,marker=markerStyles[i],label='pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),ls='None',color='black')
        ax8.set_xlim(xlim)
        ax8.set_ylim(ylim)
        resultsIsotherm.loc['pyr'+str(pH),:] = [slope,stdErr,np.mean(pyrSub.loc[:,'pH']),np.std(pyrSub.loc[:,'pH']),'pyrite']
    if not glassSub.empty:
        Cw =glassSub.loc[:,'Cw (Bq/mL)'].values
        fSorb = glassSub.loc[:,'fSorb'].values
        sCw = glassSub.loc[:,'sCw (Bq/mL)'].values
        sfSorb = glassSub.loc[:,'sfSorb'].values
        ax9.errorbar(Cw,fSorb,xerr=sCw,yerr=sfSorb,marker=markerStyles[i],label='pH: {}'.format(pH),ls='None',color='black')
    


#Strip out the errorbars from the legend, despine

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


#Try plotting pH vs fitted Kd

fRd, axRd = plt.subplots()
fRdSA, axRdSA = plt.subplots()
palMin = {'ferrihydrite':fhyPal[1],'goethite':goePal[1],'montmorillonite':montPal[1],'pyrite':pyrPal[1]}
mstyleMin = {'ferrihydrite':'o','goethite':'^','montmorillonite':'s','pyrite':'p'}
SA = {"ferrihydrite":382.9,"goethite":146.46,"montmorillonite":50.162,"pyrite":0.0685}
for mineral in resultsIsotherm.loc[:,'mineral'].unique():
    minSub = resultsIsotherm.loc[resultsIsotherm.loc[:,'mineral']==mineral,:]
    x = minSub.loc[:,'pH'].values
    y=minSub.loc[:,'Kd (mL/g)'].values
    sx = minSub.loc[:,'spH'].values
    sy = minSub.loc[:,'sKd (mL/g)'].values
    axRd.errorbar(x,y,xerr=sx,yerr=sy,ls='None',marker=mstyleMin[mineral],color=palMin[mineral],label=mineral)
    axRdSA.errorbar(x,y/SA[mineral],xerr=sx,yerr=sy/SA[mineral],ls='None',marker=mstyleMin[mineral],color=palMin[mineral],label=mineral) #plot SA
    ex = extData.loc[extData.loc[:,'Mineral']==mineral,'pH'].values
    ey = extData.loc[extData.loc[:,'Mineral']==mineral,'Kd (mL/g)'].values
    #axRd.errorbar(ex,ey,marker=mstyleMin[mineral],color=palMin[mineral],fillstyle='none',label='Other studies '+mineral,ls='none')
    ey = extData.loc[extData.loc[:,'Mineral']==mineral,'Ksa (mL/m2)'].values
    #axRdSA.errorbar(ex,ey,marker=mstyleMin[mineral],color=palMin[mineral],fillstyle='none',label='Other studies '+mineral,ls='none')
    resultsIsotherm.loc[resultsIsotherm.loc[:,'mineral']==mineral,'Ksa (mL/m2)'] = y/SA[mineral]
    resultsIsotherm.loc[resultsIsotherm.loc[:,'mineral']==mineral,'sKsa (mL/m2)'] = sy/SA[mineral]
axRd.set_ylabel('Kd (mL/g)')
axRd.set_xlabel('pH')
axRd.set_title('pH vs Fitted Kd')
axRd.set_yscale('log')
axRdSA.set_title('pH vs Fitted Ksa')
axRdSA.set_xlabel('pH')
axRdSA.set_ylabel('Ksa (mL/m^2)')
plt.yscale('log')

handles, labels = axRd.get_legend_handles_labels()
handles = [h[0] for h in handles]
axRd.legend(handles,labels,loc=0,numpoints=1)

handles, labels = axRdSA.get_legend_handles_labels()
handles = [h[0] for h in handles]
axRdSA.legend(handles,labels,loc=0,numpoints=1)
sns.despine()
"""SECTION 3: SALINITY PLOTS"""


#Data for each mineral, only mixed ion solutions

dataArtificial = dataMultiSalinity.loc[dataMultiSalinity.loc[:,'Salt'].isin(["AGW","ABW","ASW"]),:]

valsFHY = dataArtificial.loc[dataArtificial.loc[:,'Mineral']=='Ferrihydrite',:]
valsGOE = dataArtificial.loc[dataArtificial.loc[:,'Mineral']=='Goethite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)
valsMont = dataArtificial.loc[dataArtificial.loc[:,'Mineral']=='Sodium Montmorillonite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)
valsPYR = dataArtificial.loc[dataArtificial.loc[:,'Mineral']=='Pyrite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)


##PLOT 1: Plot, by mineral, "Kd" (Cs/Cw) vs. Ionic Strength, using only artificial waters
                                 
KdIFig, KdIAx = plt.subplots()#2,2,sharex="all",sharey="all")
I = valsFHY.loc[:,'Ionic Strength (meq/L)'].values
K = valsFHY.loc[:,'Rd (mL/g)'] #Units of mL/g
sK = valsFHY.loc[:,'dRd (mL/g)']
KdIAx.errorbar(I,K,yerr=sK,marker='o',label='Ferrihydrite',ls='None',color=fhyPal[1])



I = valsGOE.loc[:,'Ionic Strength (meq/L)'].values
K = valsGOE.loc[:,'Rd (mL/g)'] #Units of mL/g
sK = valsGOE.loc[:,'dRd (mL/g)']
KdIAx.errorbar(I,K,yerr=sK,marker='o',label='Goethite',ls='None',color=goePal[1])


I = valsMont.loc[:,'Ionic Strength (meq/L)'].values
K = valsMont.loc[:,'Rd (mL/g)']
sK = valsMont.loc[:,'dRd (mL/g)']  #units of mL/g
KdIAx.errorbar(I,K,yerr=sK,marker='o',label='Sodium Montmorillonite',ls='None',color=montPal[1])

I = valsPYR.loc[:,'Ionic Strength (meq/L)'].values
K = valsPYR.loc[:,'Rd (mL/g)']
sK = valsPYR.loc[:,'dRd (mL/g)']
KdIAx.errorbar(I,K,yerr=sK,marker='o',label='Pyrite',ls='None',color=pyrPal[1]) #need data first



KdIAx.set_yscale('log')
KdIAx.set_xscale('linear')
KdIAx.set_xlabel("Ionic Strength (meq/L)")
KdIAx.set_ylabel("Rd (mL/g)")
KdIAx.legend(loc=0)
KdIAx.set_title("Ionic Strength vs K")
sns.despine()

#PLOT 2: IONIC STRENGTH VS FSORB

sorbIFig, sorbIAx = plt.subplots()
I = valsFHY.loc[:,'Ionic Strength (meq/L)'].values
K = valsFHY.loc[:,'fSorb'].values #Dimensionless units
sK = valsFHY.loc[:,'sfSorb'].values #Dimensionless units
sorbIAx.errorbar(I,K,yerr=sK,marker='o',label='Ferrihydrite',ls='None',color=fhyPal[1])

I = valsGOE.loc[:,'Ionic Strength (meq/L)'].values
K = valsGOE.loc[:,'fSorb'].values #Dimensionless units
sK = valsGOE.loc[:,'sfSorb'].values #Dimensionless units
sorbIAx.errorbar(I,K,yerr=sK,marker='o',label='Goethite',ls='None',color=goePal[1])

I = valsMont.loc[:,'Ionic Strength (meq/L)'].values
K = valsMont.loc[:,'fSorb'].values #Dimensionless units
sK =valsMont.loc[:,'sfSorb'].values #Dimensionless units
sorbIAx.errorbar(I,K,yerr=sK,marker='o',label='Sodium Montmorillonite',ls='None',color=montPal[1])

I = valsPYR.loc[:,'Ionic Strength (meq/L)'].values
K = valsPYR.loc[:,'fSorb'].values #Dimensionless units
sK =valsPYR.loc[:,'sfSorb'].values #Dimensionless units
sorbIAx.errorbar(I,K,yerr=sK,marker='o',label='Pyrite',ls='None',color=pyrPal[1])

#sorbIAx.set_xscale('log')
sorbIAx.set_ylim([0,0.6])
sorbIAx.set_xlim([0,850])
sorbIAx.legend(loc=0)
plt.title("Ionic Strength vs fSorb")
sns.despine()

##PLOT 3: Bars. Requires to have the same number of points for each data value. Plotted by salt, with each mineral appearing

dataLowI = dataMultiSalinity.loc[dataMultiSalinity.loc[:,'Ionic Strength (meq/L)']<50,:]

valsFHY = dataLowI.loc[dataLowI.loc[:,'Mineral']=='Ferrihydrite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)
valsGOE = dataLowI.loc[dataLowI.loc[:,'Mineral']=='Goethite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)
valsMont = dataLowI.loc[dataLowI.loc[:,'Mineral']=='Sodium Montmorillonite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)
valsPYR = dataLowI.loc[dataLowI.loc[:,'Mineral']=='Pyrite',:].sort_values(by='Ionic Strength (meq/L)',ascending=True)

saltLabels =valsFHY.loc[:,'Salt'].unique()
groups = len(saltLabels)

fBar, axBar = plt.subplots()
index = np.arange(groups)
bar_width =1.0/(groups+1)

fhyBar = plt.bar(index-1.5*bar_width,valsFHY.loc[:,'Rd (mL/g)'],bar_width,yerr = valsFHY.loc[:,'dRd (mL/g)'],alpha = 1, color = fhyPal[1],label = 'Ferrihydrite',ecolor='k')
goeBar = plt.bar(index-0.5*bar_width,valsGOE.loc[:,'Rd (mL/g)'],bar_width,yerr = valsGOE.loc[:,'dRd (mL/g)'],alpha = 1, color = goePal[1],label = 'Goethite',ecolor='k')
montBar = plt.bar(index+0.5*bar_width,valsMont.loc[:,'Rd (mL/g)'],bar_width,yerr = valsMont.loc[:,'dRd (mL/g)'],alpha = 1, color = montPal[1],label = 'Sodium Montmorillonite',ecolor='k')
pyrBar = plt.bar(index+1.5*bar_width,valsPYR.loc[:,'Rd (mL/g)'],bar_width,yerr = valsPYR.loc[:,'dRd (mL/g)'],alpha = 1, color = pyrPal[1],label = 'Pyrite',ecolor='k') #need more data points

axBar.set_ylabel('Rd (mL/g)')
axBar.set_yscale('log')
axBar.set_title("Impact from ions at similar ionic strength")
axBar.set_xticks(index)
axBar.set_xticklabels(saltLabels)

axBar.legend(loc=0)
sns.despine()

#PLOT 4: BARS BY FSORB BECAUSE WHAT DOES RD EVEN MEAN

f2Bar, axBar2 = plt.subplots()
index = np.arange(groups)
bar_width =1.0/(groups+1.0)

fhyBar = axBar2.bar(index-1.5*bar_width,valsFHY.loc[:,'fSorb'],bar_width,yerr = valsFHY.loc[:,'sfSorb'],alpha = 1, color = fhyPal[1],label = 'Ferrihydrite',ecolor='k')
goeBar = axBar2.bar(index-0.5*bar_width,valsGOE.loc[:,'fSorb'],bar_width,yerr = valsGOE.loc[:,'sfSorb'],alpha = 1, color = goePal[1],label = 'Goethite',ecolor='k')
montBar = axBar2.bar(index+0.5*bar_width,valsMont.loc[:,'fSorb'],bar_width,yerr = valsMont.loc[:,'sfSorb'],alpha = 1, color = montPal[1],label = 'Sodium Montmorillonite',ecolor='k')
pyrBar = axBar2.bar(index+1.5*bar_width,valsPYR.loc[:,'fSorb'],bar_width,yerr = valsPYR.loc[:,'sfSorb'],alpha = 1, color = pyrPal[1],label = 'Pyrite',ecolor='k') #need more data points

axBar2.set_ylabel('Fraction sorbed (.)')
axBar2.set_title("Impact from ions at similar ionic strength")
axBar2.set_xticks(index)
axBar2.set_xticklabels(saltLabels)

axBar2.legend(loc=0)
sns.despine()

#PLOT 5: Bars plotted by mineral, also, example of how to bar plot with an unknown amount of bars

f3Bar, axBar3 = plt.subplots()
minLabels = ['Ferrihydrite','Goethite','Pyrite','Sodium Montmorillonite']
saltLabels = ['NaCl','KCl','MgCl2','CaCl2','SrCl2','AGW']
groups = len(minLabels)
index = np.arange(groups)
nBars = len(saltLabels)
bar_width = 1.0/(nBars+1) #Spacing off of "center point" is defined by this step size
offset = iter(np.linspace(-0.5+bar_width,0.5-bar_width,num=nBars))
dataMinSort = dataLowI.loc[dataLowI.loc[:,'Mineral'].isin(minLabels),:].sort_values(by='Mineral',ascending=True)
palette = iter(sns.color_palette('deep',n_colors=nBars))

for label in saltLabels:
    fSalt = dataMinSort.loc[dataMinSort.loc[:,'Salt']==label,'fSorb']
    sfSalt = dataMinSort.loc[dataMinSort.loc[:,'Salt']==label,'sfSorb']
    axBar3.bar(index+offset.next(),fSalt,bar_width,yerr=sfSalt,label=label,color=next(palette),alpha=1,ecolor='k')

axBar3.legend(loc=0)
axBar3.set_xticks(index)
axBar3.set_xticklabels(minLabels)
axBar3.set_ylabel("Fraction Sorbed (.)")
sns.despine()

#PLOT 6: TRYING TO DO AWAY WITH BARS TO CREATE SOMETHING... NEW

saltKeys = saltLabels
minKeys = dataLowI.loc[:,'Mineral'].unique()
saltIndex = np.arange(len(saltKeys))
minIndex = np.arange(len(minKeys))

saltMap = dict(zip(saltKeys,saltIndex))
minMap = dict(zip(minKeys,minIndex))

dataLowI.loc[:,"SaltKey"] = dataLowI.loc[:,"Salt"].map(saltMap)
dataLowI.loc[:,"MinKey"] = dataLowI.loc[:,"Mineral"].map(minMap)

f4Bar, ax4Bar = plt.subplots()

colors = sns.cubehelix_palette(start=0,rot=1.5,as_cmap=True,gamma=1)
colors = sns.dark_palette("green",as_cmap=True,reverse=True)
spacePlot = ax4Bar.scatter(dataLowI.loc[:,"SaltKey"],dataLowI.loc[:,"MinKey"],cmap=colors,c=dataLowI.loc[:,'fSorb'],s=dataLowI.loc[:,'fSorb']*750)
f4Bar.colorbar(spacePlot)
ax4Bar.set_xticks(saltIndex)
ax4Bar.set_yticks(minIndex)
ax4Bar.set_xticklabels(saltKeys)
ax4Bar.set_yticklabels(minKeys)
ax4Bar.set_title("Radium Fraction Sorbed")
sns.despine()

"""SECTION 4: IMPACT OF MASS ON SORPTION"""

agwData = dataMass.loc[dataMass.loc[:,'Salt']=="AGW",:]
abwData = dataMass.loc[dataMass.loc[:,'Salt']=="ABW",:]
aswData = dataMass.loc[dataMass.loc[:,'Salt']=="ASW",:]

figMass, axMass = plt.subplots()
agwPlot = axMass.plot(agwData.loc[:,"MinMass (g)"].values,agwData.loc[:,"Cs (Bq/g)"].values/agwData.loc[:,"Cw (Bq/mL)"].values,color="b",label="Aritificial groundwater",ls="None",marker="o")
abwPlot = axMass.plot(abwData.loc[:,"MinMass (g)"].values,abwData.loc[:,"Cs (Bq/g)"].values/abwData.loc[:,"Cw (Bq/mL)"].values,color="g",label="Artificial Brackish Water",ls="None",marker="o")
aswPlot = axMass.plot(aswData.loc[:,"MinMass (g)"].values,aswData.loc[:,"Cs (Bq/g)"].values/aswData.loc[:,"Cw (Bq/mL)"].values,color="r",label="Artificial Seawater",ls="None",marker="o")


axMass.set_xlabel("Mineral mass (g)")
axMass.set_ylabel("Rd (mL/g)")
axMass.legend(loc=0)
axMass.set_title("Mineral mass vs K")
sns.despine()
"""SECTION 5: SAVING EVERYTHING"""

plt.show()
#f3.savefig('..\\Manuscript\\Figures\\Sorption Isotherms.svg',dpi=1000)
#f3.savefig('MasterTablePlots\\Sorption Isotherms All Valid Data.svg', dpi=1000)
##f4.savefig('..\\Manuscript\\Figures\\Figure1-pH7Isotherms.svg',dpi=1000)
#f5.savefig('MasterTablePlots\\IsothermsFHY.svg',dpi=1000)
#f6.savefig('MasterTablePlots\\IsothermsNaMont.svg',dpi=1000)
#f7.savefig('MasterTablePlots\\IsothermsGOE.svg',dpi=1000)
#f8.savefig('MasterTablePlots\\IsothermsPYR.svg',dpi=1000)
##f9.savefig('MasterTablePlots\\KineticsNaMont.svg',dpi=1000)

#resultsIsotherm.to_csv('..\\Manuscript\\Second Submission EST\\Figures\\IsothermResults.csv')
#dataLowIExport = dataLowI.loc[:,['SampleID','Ionic Strength (meq/L)','Salt','Mineral','fSorb','sfSorb']]
#dataLowIExport.to_csv('..\\Manuscript\\Second Submission EST\\Figures\\LowSalinityData.csv')
#dataArtificialExport = dataArtificial.loc[:,['SampleID','Ionic Strength (meq/L)','Mineral','fSorb','sfSorb']]
#dataArtificialExport.to_csv('..\\Manuscript\\Second Submission EST\\Figures\\ArtificialWaters.csv')
#
#fRd.savefig('..\\Manuscript\\Second Submission EST\\Figures\\Figure1a-pHKd.svg',dpi=1000)
#fRdSA.savefig('..\\Manuscript\\Second Submission EST\\Figures\\Figure1a-pHKdSA.svg',dpi=1000)
#sorbIFig.savefig('..\Manuscript\\Second Submission EST\\Figures\\Figure2a-IvsfSorb.svg',dpi=1000)
#f3Bar.savefig('..\Manuscript\\Second Submission EST\\Figures\\Figure2b-LowIBars.svg',dpi=1000)