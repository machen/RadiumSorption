# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 16:47:14 2016

@author: Michael
"""

import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
sns.set_context('talk')
data = pd.read_excel("Sorption Experiment Master Table.xlsx",header=0)
data = data.ix[data.ix[:,"Include?"]==True,:] #Toggle if you only want to plot data flagged to include
FHYdata = data.ix[data.ix[:,'Mineral']=="Ferrihydrite",:]
montData = data.ix[data.ix[:,'Mineral']=='Sodium Montmorillonite']
goeData = data.ix[data.ix[:,'Mineral']=='Goethite']



#Metaplot of all data: Sorption Envelope

f1 = plt.figure(1)
f1.clf()
ax = f1.add_subplot(111)
fhyP = ax.errorbar(FHYdata.ix[:,'pH'],FHYdata.ix[:,'fSorb'],xerr=FHYdata.ix[:,'spH'],yerr=FHYdata.ix[:,'sfSorb'],fmt='o',label='Ferrihydrite')
montP = ax.errorbar(montData.ix[:,'pH'],montData.ix[:,'fSorb'],xerr=montData.ix[:,'spH'],yerr=montData.ix[:,'sfSorb'],fmt='o',label='Na Montmorillonite')
goeP = ax.errorbar(goeData.ix[:,'pH'],goeData.ix[:,'fSorb'],xerr=goeData.ix[:,'spH'],yerr=goeData.ix[:,'sfSorb'],fmt='o',label='Goethite')
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
fhyP = ax2.errorbar(FHYdata.ix[:,'Cw (Bq/mL)'],FHYdata.ix[:,'Cs (Bq/g)'],xerr=FHYdata.ix[:,'sCw (Bq/mL)'],yerr=FHYdata.ix[:,'sCs (Bq/g)'],fmt='o',label='Ferrihydrite')
montP = ax2.errorbar(montData.ix[:,'Cw (Bq/mL)'],montData.ix[:,'Cs (Bq/g)'],xerr=montData.ix[:,'sCw (Bq/mL)'],yerr=montData.ix[:,'sCs (Bq/g)'],fmt='o',label='Na Montmorillonite')
goeP = ax2.errorbar(goeData.ix[:,'Cw (Bq/mL)'],goeData.ix[:,'Cs (Bq/g)'],xerr=goeData.ix[:,'sCw (Bq/mL)'],yerr=goeData.ix[:,'sCs (Bq/g)'],fmt='o',label='Goethite')
plt.xlabel('Cw (Bq/mL)')
plt.ylabel('Cs (Bq/g)')
plt.title('Sorption Isotherms')
plt.legend(loc=0)
plt.show()

f3 = plt.figure(3)
f3.clf()
ax3 = f3.add_subplot(111)
pHvals  = [3,5,7,9]
fhyPal = sns.cubehelix_palette(n_colors=4,dark=0.3,rot=0.4,light=0.8,gamma=1.3)
montPal = sns.cubehelix_palette(n_colors=4,dark=0.3,rot=-0.4,light=0.8,gamma=1.3)
goePal = sns.cubehelix_palette(n_colors=4,dark=0.3,rot=0,light=0.8,gamma=1.3)

for i in range(4):
    pH = pHvals[i]
    fhySub = FHYdata.ix[abs(FHYdata.ix[:,'pH']-pH)<0.1,:]
    montSub = montData.ix[abs(montData.ix[:,'pH']-pH)<0.1,:]
    goeSub = goeData.ix[abs(goeData.ix[:,'pH']-pH)<0.1,:]
    if not fhySub.empty:
        ax3.errorbar(fhySub.ix[:,'Cw (Bq/mL)'].values,fhySub.ix[:,'Cs (Bq/g)'].values,xerr=fhySub.ix[:,'sCw (Bq/mL)'].values,yerr=fhySub.ix[:,'sCs (Bq/g)'].values,fmt='o',label='FHY pH: {0}'.format(str(pH)),color=fhyPal[i])
    if not montSub.empty:
        ax3.errorbar(montSub.ix[:,'Cw (Bq/mL)'].values,montSub.ix[:,'Cs (Bq/g)'].values,xerr=montSub.ix[:,'sCw (Bq/mL)'].values,yerr=montSub.ix[:,'sCs (Bq/g)'].values,fmt='o',label='Na Mont. pH: {0}'.format(str(pH)),color=montPal[i])
    if not goeSub.empty:
        ax3.errorbar(goeSub.ix[:,'Cw (Bq/mL)'].values,goeSub.ix[:,'Cs (Bq/g)'].values,xerr=goeSub.ix[:,'sCw (Bq/mL)'].values,yerr=goeSub.ix[:,'sCs (Bq/g)'].values,fmt='o',label='GOE pH: {0}'.format(str(pH)),color=goePal[i])
plt.xlabel('Cw (Bq/mL)')
plt.ylabel('Cs (Bq/g)')
plt.title('Sorption Isotherms')
plt.legend(loc=0)
plt.show()

f1.savefig('MasterTablePlots\\Sorption Envelope.png',DPI=900)
f3.savefig('MasterTablePlots\\Sorption Isotherms.png',DPI=900)