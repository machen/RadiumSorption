# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 16:47:14 2016

@author: Michael
"""

import pandas as pd, numpy as np, matplotlib as mpl
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import linregress
sns.set_context('poster')
sns.set_style("ticks",rc={"font.size":48})
data = pd.read_excel("Sorption Experiment Master Table.xlsx",header=0)
data = data.ix[data.ix[:,"Include?"]==True,:] #Toggle if you only want to plot data flagged to include

#Create dataframes containing only the sub data (sort by minerals)

FHYdata = data.ix[data.ix[:,'Mineral']=="Ferrihydrite",:]
montData = data.ix[data.ix[:,'Mineral']=='Sodium Montmorillonite']
goeData = data.ix[data.ix[:,'Mineral']=='Goethite']
pyrData = data.ix[data.ix[:,'Mineral']=='Pyrite']

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


#Metaplot of all data: Sorption Envelope

f1, ax = plt.subplots(1,1,figsize=(20,10))
fhyP = ax.errorbar(FHYdata.ix[:,'pH'].values,FHYdata.ix[:,'fSorb'].values,xerr=FHYdata.ix[:,'spH'].values,yerr=FHYdata.ix[:,'sfSorb'].values,fmt='o',label='Ferrihydrite')
montP = ax.errorbar(montData.ix[:,'pH'].values,montData.ix[:,'fSorb'].values,xerr=montData.ix[:,'spH'].values,yerr=montData.ix[:,'sfSorb'].values,fmt='o',label='Na Montmorillonite')
goeP = ax.errorbar(goeData.ix[:,'pH'].values,goeData.ix[:,'fSorb'].values,xerr=goeData.ix[:,'spH'].values,yerr=goeData.ix[:,'sfSorb'].values,fmt='o',label='Goethite')
pyrP = ax.errorbar(pyrData.ix[:,'pH'].values,pyrData.ix[:,'fSorb'].values,xerr=pyrData.ix[:,'spH'].values,yerr=pyrData.ix[:,'sfSorb'].values,fmt='o',label='Pyrite')
plt.xlabel('pH')
plt.ylabel('Fraction Sorbed')
plt.title('Sorption Envelopes')
plt.ylim([0,1])
plt.legend(loc=0)
plt.show()
sns.despine()
#Metaplot of all data: Sorption isotherm

f2, ax2 = plt.subplots(1,1,figsize=(20,10))
fhyP = ax2.errorbar(FHYdata.ix[:,'Cw (Bq/mL)'].values,FHYdata.ix[:,'Cs (Bq/g)'].values,xerr=FHYdata.ix[:,'sCw (Bq/mL)'].values,yerr=FHYdata.ix[:,'sCs (Bq/g)'].values,fmt='s',label='Ferrihydrite',mfc=
'None',mec=sns.color_palette()[0])
montP = ax2.errorbar(montData.ix[:,'Cw (Bq/mL)'].values,montData.ix[:,'Cs (Bq/g)'].values,xerr=montData.ix[:,'sCw (Bq/mL)'].values,yerr=montData.ix[:,'sCs (Bq/g)'].values,fmt='o',label='Na Montmorillonite',mfc=
'None',mec=sns.color_palette()[1])
goeP = ax2.errorbar(goeData.ix[:,'Cw (Bq/mL)'].values,goeData.ix[:,'Cs (Bq/g)'].values,xerr=goeData.ix[:,'sCw (Bq/mL)'].values,yerr=goeData.ix[:,'sCs (Bq/g)'].values,fmt='^',label='Goethite',mfc=
'None',mec=sns.color_palette()[2])
pyrP = ax2.errorbar(pyrData.ix[:,'Cw (Bq/mL)'].values,pyrData.ix[:,'Cs (Bq/g)'].values,xerr=pyrData.ix[:,'sCw (Bq/mL)'].values,yerr=pyrData.ix[:,'sCs (Bq/g)'].values,fmt='p',label='Pyrite',mfc=
'None',mec=sns.color_palette()[3])

#Remove errorbars from legend
handles, labels = ax2.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax2.legend(handles,labels,loc=0,numpoints=1)

plt.xlabel('Cw (Bq/mL)')
plt.ylabel('Cs (Bq/g)')
plt.title('Sorption Isotherms')
plt.xlim([-0.1,2.9])
plt.ylim([-100,8700])
#plt.legend(loc=0)
plt.show()
sns.despine()

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

fhyOutput = pd.DataFrame(index = np.arange(0,5))
goeOutput = pd.DataFrame(index = np.arange(0,5))
montOutput = pd.DataFrame(index = np.arange(0,5))
pyrOutput = pd.DataFrame(index = np.arange(0,5))

for i in range(len(pHvals)):
    pH = pHvals[i]
    pHs = str(pH)
    fhySub = FHYdata.ix[abs(FHYdata.ix[:,'pH']-pH)<0.2,:]
    montSub = montData.ix[abs(montData.ix[:,'pH']-pH)<0.2,:]
    goeSub = goeData.ix[abs(goeData.ix[:,'pH']-pH)<0.2,:]
    pyrSub = pyrData.ix[abs(pyrData.ix[:,'pH']-pH)<0.2,:]
    xlim = [-0.5,3.0]
    ylim = [-100,20000]
    if not fhySub.empty:
        Cw = fhySub.ix[:,'Cw (Bq/mL)'].values
        Cs = fhySub.ix[:,'Cs (Bq/g)'].values
        sCw = fhySub.ix[:,'sCw (Bq/mL)'].values
        sCs = fhySub.ix[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)
        out = np.array([Cw,Cs,sCw,sCs,np.polyval([slope,inter],Cw)])
        out = pd.DataFrame(np.transpose(out), columns = [pHs+' Cw',pHs+' Cs',pHs+' sCw',pHs+' sCs',pHs+' Fit'])
        fhyOutput = fhyOutput.merge(out,how='outer',left_index=True,right_index=True)
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
        out = np.array([Cw,Cs,sCw,sCs,np.polyval([slope,inter],Cw)])
        out = pd.DataFrame(np.transpose(out), columns = [pHs+' Cw',pHs+' Cs',pHs+' sCw',pHs+' sCs',pHs+' Fit'])
        montOutput = montOutput.merge(out,how='outer',left_index=True,right_index=True)
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
        out = np.array([Cw,Cs,sCw,sCs,np.polyval([slope,inter],Cw)])
        out = pd.DataFrame(np.transpose(out), columns = [pHs+' Cw',pHs+' Cs',pHs+' sCw',pHs+' sCs',pHs+' Fit'])
        goeOutput = goeOutput.merge(out,how='outer',left_index=True,right_index=True)
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
        out = np.array([Cw,Cs,sCw,sCs,np.polyval([slope,inter],Cw)])
        out = pd.DataFrame(np.transpose(out), columns = [pHs+' Cw',pHs+' Cs',pHs+' sCw',pHs+' sCs',pHs+' Fit'])
        pyrOutput = pyrOutput.merge(out,how='outer',left_index=True,right_index=True)
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
      
#Plot of Montmorillonite Kinetic Data      
times = np.array([175,374,1440,15547,15530])
montKineticCw = np.array([0.492803,0.473911,0.487378,0.541394,0.638563])
montKineticsCw = np.array([0.007277,0.018693,0.028428,0.101393,0.054642])
f9 = plt.figure(9)
ax9 = f9.add_subplot(111)
ax9.errorbar(times,montKineticCw,yerr=montKineticsCw,marker=markerStyles[0],color='k')
ax9.set_ylim([0,1.0])
ax9.set_xlabel('Experiment Length (sec)')
ax9.set_ylabel('Cw (Bq/mL)')

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
#plt.xlabel('Cw (Bq/mL)')
#plt.ylabel('Cs (Bq/g)')
#plt.title('Sorption Isotherms')
#plt.legend(loc=0)
sns.despine(f4)
sns.despine(f5)
sns.despine(f6)
sns.despine(f7)
sns.despine(f8)
sns.despine(f9)
plt.show()
#f1.savefig('MasterTablePlots\\Sorption Envelope.svg',dpi=1000)
f3.savefig('..\\Manuscript\\Figures\\Sorption Isotherms.svg',dpi=1000)
#f4.savefig('..\\Manuscript\\Figures\\Figure1-pH7Isotherms.svg',dpi=1000)
#f5.savefig('MasterTablePlots\\IsothermsFHY.svg',dpi=1000)
#f6.savefig('MasterTablePlots\\IsothermsNaMont.svg',dpi=1000)
#f7.savefig('MasterTablePlots\\IsothermsGOE.svg',dpi=1000)
#f8.savefig('MasterTablePlots\\IsothermsPYR.svg',dpi=1000)
#f9.savefig('MasterTablePlots\\KineticsNaMont.svg',dpi=1000)
#
fhyOutput.to_csv('MasterTablePlots\\FhyIsothermData.csv')
goeOutput.to_csv('MasterTablePlots\\GoeIsothermData.csv')
montOutput.to_csv('MasterTablePlots\\MontIsothermData.csv')
pyrOutput.to_csv('MasterTablePlots\\PyrIsothermData.csv')