# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 09:46:33 2014

@author: Michael
"""

#Mucking about with problem set, curve fitting via minimization of sum of squared error
import numpy as np, matplotlib.pyplot as plt, pandas as pd
    
    
#py.sign_in('machen27','41mjojsg9f')    
directory = ""
name = "AGU_2014_FHY_Ra"
filename = name+".csv"
raw = pd.read_csv(directory+filename)
Aw = raw['Cw']
As = raw['Cs'] #Values are in CPM, taken straight from the scintillation counter
countEff = 0.43 #Counting efficiency of the cocktail
Vwater = 0.0100 #L
Msolid = 13.255*0.001 #g (Slurry concentration*1 mL)
Aw=Aw*60/countEff/Vwater #Convert to Becquerels of actual 226-Ra per L
As=As*60/countEff/Msolid #Convert to Becquerels 226-Ra per g solid

Cw = Aw/3.7E10/226 #Convert to Molar
Cs = As/3.7E10/226 #Convert to moles/g solid


try:
    dAw = raw['dCw'] #Fractional uncertainties
    dAs = raw['dCs'] #Fractional unceratinties
    dCw = Cw*dAw
    dCs = Cs*dAs
    dAw = Aw*dAw #Convert fractional uncertainties to straight uncertainties
    dAs = As*dAs #Convert fractional uncertainties to straight uncertainties
except KeyError:
    pass

kdFitC = np.polyfit(Cw,Cs,1,full=True)
kdFitA = np.polyfit(Aw,As,1)

CwM = np.array([0.7*min(Cw),1.1*max(Cw)])
AwM = np.array([0.7*min(Aw),1.1*max(Aw)])

f1 = plt.figure(1)
plt.clf()
ax1 = f1.add_subplot(111)
perr = ax1.errorbar(Cw,Cs,yerr = dCs,xerr = dCw,fmt=None)
p1, = ax1.plot(Cw,Cs,"xb")
p4, = ax1.plot(CwM,np.polyval(kdFitC[0],CwM),'-k')
ax1.ticklabel_format(axis='y',style = 'sci',scilimits=(-2,2))
ax1.ticklabel_format(axis='x',style='sci',scilimits=(-2,2))
ax1.set_ylim([0,1.2*max(Cs)])
plt.legend((p1,p4),('Experimental Data','Linear fit,  '+r'$Kd=0.39 g mol^{-1}$'),loc=0)
f1.set_size_inches(12.5,6.5)
#plt.savefig(name+".pdf",dpi=600)

#plot_url = py.plot_mpl(f1)
plt.show()
