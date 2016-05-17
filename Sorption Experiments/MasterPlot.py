# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 16:47:14 2016

@author: Michael
"""

import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import linregress
sns.set_context('talk')
data = pd.read_excel("Sorption Experiment Master Table.xlsx",header=0)
data = data.ix[data.ix[:,"Include?"]==True,:] #Toggle if you only want to plot data flagged to include
FHYdata = data.ix[data.ix[:,'Mineral']=="Ferrihydrite",:]
montData = data.ix[data.ix[:,'Mineral']=='Sodium Montmorillonite']
goeData = data.ix[data.ix[:,'Mineral']=='Goethite']
pyrData = data.ix[data.ix[:,'Mineral']=='Pyrite']



#Metaplot of all data: Sorption Envelope

f1 = plt.figure(1)
f1.clf()
ax = f1.add_subplot(111)
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

#Metaplot of all data: Sorption isotherm

f2 = plt.figure(2)
f2.clf()
ax2 = f2.add_subplot(111)
fhyP = ax2.errorbar(FHYdata.ix[:,'Cw (Bq/mL)'].values,FHYdata.ix[:,'Cs (Bq/g)'].values,xerr=FHYdata.ix[:,'sCw (Bq/mL)'].values,yerr=FHYdata.ix[:,'sCs (Bq/g)'].values,fmt='o',label='Ferrihydrite')
montP = ax2.errorbar(montData.ix[:,'Cw (Bq/mL)'].values,montData.ix[:,'Cs (Bq/g)'].values,xerr=montData.ix[:,'sCw (Bq/mL)'].values,yerr=montData.ix[:,'sCs (Bq/g)'].values,fmt='o',label='Na Montmorillonite')
goeP = ax2.errorbar(goeData.ix[:,'Cw (Bq/mL)'].values,goeData.ix[:,'Cs (Bq/g)'].values,xerr=goeData.ix[:,'sCw (Bq/mL)'].values,yerr=goeData.ix[:,'sCs (Bq/g)'].values,fmt='o',label='Goethite')
pyrP = ax2.errorbar(pyrData.ix[:,'Cw (Bq/mL)'].values,pyrData.ix[:,'Cs (Bq/g)'].values,xerr=pyrData.ix[:,'sCw (Bq/mL)'].values,yerr=pyrData.ix[:,'sCs (Bq/g)'].values,fmt='o',label='Pyrite')
plt.xlabel('Cw (Bq/mL)')
plt.ylabel('Cs (Bq/g)')
plt.title('Sorption Isotherms')
plt.legend(loc=0)
plt.show()

#Plot of Isotherms separated by pH, along with isotherm fits

f3 = plt.figure(3)
f3.clf()
ax3 = f3.add_subplot(111)
pHvals  = [3,5,7,9]
fhyPal = sns.cubehelix_palette(n_colors=4,dark=0.3,rot=0.2,light=0.8,gamma=1.3)
montPal = sns.cubehelix_palette(n_colors=4,dark=0.3,rot=-0.2,light=0.8,gamma=1.3)
goePal = sns.cubehelix_palette(n_colors=4,dark=0.3,rot=0,light=0.8,gamma=1.3)
pyrPal = sns.cubehelix_palette(n_colors=4,dark=0.3,rot=0.4,light=0.8,gamma=1.3)

for i in range(4):
    pH = pHvals[i]
    fhySub = FHYdata.ix[abs(FHYdata.ix[:,'pH']-pH)<0.1,:]
    montSub = montData.ix[abs(montData.ix[:,'pH']-pH)<0.1,:]
    goeSub = goeData.ix[abs(goeData.ix[:,'pH']-pH)<0.1,:]
    pyrSub = pyrData.ix[abs(pyrData.ix[:,'pH']-pH)<0.1,:]
    if not fhySub.empty:
        Cw = fhySub.ix[:,'Cw (Bq/mL)'].values
        Cs = fhySub.ix[:,'Cs (Bq/g)'].values
        sCw = fhySub.ix[:,'sCw (Bq/mL)'].values
        sCs = fhySub.ix[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)  
        ax3.plot(Cw,np.polyval([slope,inter],Cw),ls='-',label=None,color=fhyPal[i])
        ax3.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,fmt='o',label='FHY pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),color=fhyPal[i])
    if not montSub.empty:
        Cw = montSub.ix[:,'Cw (Bq/mL)'].values
        Cs = montSub.ix[:,'Cs (Bq/g)'].values
        sCw = montSub.ix[:,'sCw (Bq/mL)'].values
        sCs = montSub.ix[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)  
        ax3.plot(Cw,np.polyval([slope,inter],Cw),ls='-',label=None,color=montPal[i])
        ax3.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,fmt='o',label='Montmorillonite pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),color=montPal[i])
    if not goeSub.empty:
        Cw = goeSub.ix[:,'Cw (Bq/mL)'].values
        Cs = goeSub.ix[:,'Cs (Bq/g)'].values
        sCw = goeSub.ix[:,'sCw (Bq/mL)'].values
        sCs = goeSub.ix[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)  
        ax3.plot(Cw,np.polyval([slope,inter],Cw),ls='-',label=None,color=goePal[i])
        ax3.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,fmt='o',label='Goethite pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),color=goePal[i])
    if not pyrSub.empty:
        Cw = pyrSub.ix[:,'Cw (Bq/mL)'].values
        Cs = pyrSub.ix[:,'Cs (Bq/g)'].values
        sCw = pyrSub.ix[:,'sCw (Bq/mL)'].values
        sCs = pyrSub.ix[:,'sCs (Bq/g)'].values
        [slope,inter,rval,pval,stdErr] = linregress(Cw,Cs)  
        ax3.plot(Cw,np.polyval([slope,inter],Cw),ls='-',label=None,color=pyrPal[i])
        ax3.errorbar(Cw,Cs,xerr=sCw,yerr=sCs,fmt='o',label='Pyrite pH: {} Kd: {:.2f} R2: {:.2f}'.format(pH,slope,rval**2),color=pyrPal[i])
plt.xlabel('Cw (Bq/mL)')
plt.ylabel('Cs (Bq/g)')
plt.title('Sorption Isotherms')
plt.legend(loc=0)
plt.show()

#f1.savefig('MasterTablePlots\\Sorption Envelope.png',DPI=900)
#f3.savefig('MasterTablePlots\\Sorption Isotherms.png',DPI=900)