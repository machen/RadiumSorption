# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:33:31 2015

@author: Michael
"""

import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns

CO2data = pd.read_excel('RadiumSolutionComplexationCO2.xlsx',parse_cols = 'B,M:S',skiprows=[1])
noCO2data = pd.read_excel('RadiumSolutionComplexationNoCO2.xlsx',parse_cols = 'B,M:S',skiprows=[1])

sns.set_palette('hls',n_colors=len(CO2data.columns)-1)
sns.color_palette()
sns.set_context('talk')

CO2f = plt.figure(1)
ax = CO2f.add_subplot(111)
pA1, = ax.plot(CO2data.ix[:,'pH'],CO2data.ix[:,'Ra+2'],label = r'$\mathsf{Ra^{+2}}$',marker='None') 
pA2, = ax.plot(CO2data.ix[:,'pH'],CO2data.ix[:,'Ra(OH)+'],label=r'$\mathsf{RaOH^{+}}$',marker='None')
pA3, = ax.plot(CO2data.ix[:,'pH'],CO2data.ix[:,'Ra(OH)2 (aq)'],label=r'$\mathsf{Ra(OH)_{2}\,(aq)}$')
pA4, = ax.plot(CO2data.ix[:,'pH'],CO2data.ix[:,'RaHCO3+'],label=r'$\mathsf{RaHCO_{3}^{+}}$')
pA5, = ax.plot(CO2data.ix[:,'pH'],CO2data.ix[:,'RaCO3 (aq)'],label=r'$\mathsf{RaCO_{3}\,(aq)}$')
pA6, = ax.plot(CO2data.ix[:,'pH'],CO2data.ix[:,'RaCl+'],label=r'$\mathsf{RaCl^{+}}$')
pA7, = ax.plot(CO2data.ix[:,'pH'],CO2data.ix[:,'RaCl2 (aq)'],label=r'$\mathsf{RaCl_{2}\,(aq)}$')
ax.set_xlabel('pH')
ax.set_ylabel('Fraction of total Radium')
ax.set_title(r'Radium Complexation with Carbon Dioxide')
plt.legend(loc=6)
plt.show()
CO2f.savefig('RadiumComplexeswCO2.png',dpi=600)

noCO2f = plt.figure(1)
ax = noCO2f.add_subplot(111)
pA1, = ax.plot(noCO2data.ix[:,'pH'],noCO2data.ix[:,'Ra+2'],label = r'$\mathsf{Ra^{+2}}$',marker='None') 
pA2, = ax.plot(noCO2data.ix[:,'pH'],noCO2data.ix[:,'Ra(OH)+'],label=r'$\mathsf{RaOH^{+}}$',marker='None')
pA3, = ax.plot(noCO2data.ix[:,'pH'],noCO2data.ix[:,'Ra(OH)2 (aq)'],label=r'$\mathsf{Ra(OH)_{2}\,(aq)}$')
pA4, = ax.plot(noCO2data.ix[:,'pH'],noCO2data.ix[:,'RaHCO3+'],label=r'$\mathsf{RaHCO_{3}^{+}}$')
pA5, = ax.plot(noCO2data.ix[:,'pH'],noCO2data.ix[:,'RaCO3 (aq)'],label=r'$\mathsf{RaCO_{3}\,(aq)}$')
pA6, = ax.plot(noCO2data.ix[:,'pH'],noCO2data.ix[:,'RaCl+'],label=r'$\mathsf{RaCl^{+}}$')
pA7, = ax.plot(noCO2data.ix[:,'pH'],noCO2data.ix[:,'RaCl2 (aq)'],label=r'$\mathsf{RaCl_{2}\,(aq)}$')
ax.set_xlabel('pH')
ax.set_ylabel('Fraction of total Radium')
ax.set_title(r'Radium Complexation without Carbon Dioxide')
plt.legend(loc=6)
plt.show()
noCO2f.savefig('RadiumComplexeswoCO2.png',dpi=600)