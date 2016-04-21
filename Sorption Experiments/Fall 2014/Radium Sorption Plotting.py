# -*- coding: utf-8 -*-
"""
Created on Thu Nov 06 17:31:18 2014

@author: Michael
"""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
Ra = np.array([363.0,463.0,564.00,596.00,561.50,593.00,599.50,639.50,584.00,675.50,648.00,671.50,661.00])-63.5 #Counts per minute divided by the baseline
Ra_error = np.array([7.42,6.57,5.95,5.79,5.97,5.81,5.78,5.59,5.85,5.44,5.56,5.46,5.50]) #Reported as percentages
FHY = np.array([100.0,10.0,1,0.1,0.01,0.001,10**-4,10**-5,10**-6,10**-7,10**-8,10**-9,10**-15]) #Simply a dilution factor from a given point

f1 = plt.figure(1)
p1 = plt.errorbar(FHY,Ra,yerr=Ra*Ra_error/100.0)
plt.xscale('log')
plt.xlim((10**-16,1000))
plt.ylabel('CPM')
plt.xlabel('FHY Stock Solution Dilution')
plt.title('Range Testing Sorption Test')
plt.savefig('RangeTesting_2014_11_3.pdf')

