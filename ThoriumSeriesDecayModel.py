# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 17:14:36 2014

@author: machen
"""

#Model of Thorium series decay through Ra 224 given a certain initial concentration

#Next step: Include Sorption

import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

thalf = np.array([1.405E10,5.75,6.25/24/365,1.91,3.63/365])
kdecay = np.log(2)/thalf

#Initial condition controlled by pref
c_eq = np.array([1,4.07E-10,5.12E-14,1.36E-10,7.07E-13]) #Represents the activity of the constituents at secular equillibrium given a 232Th concentration of 1
Kdgw = 1000.0 #L/kg
Kdsw = 1.0 #L/kg
Kdmix = 100.0 #L/kg
#Assume a volume of 1 L and a solid mass of 1 kg, no site availability limitations
pref = 1/(1+1.0/Kdgw)+1/(1+Kdsw) #Prefactor for initial condition. Two beakers at secular eq and sorption eq, one with seawater, one with gw (same solid). Replace gw in one beaker with sw, allow to equillibrate
c_initial = np.array([10**5,4.07E-5*pref,5.12E-9,1.36E-5,7.07E-8*pref]).transpose() #Objects in c are: 232Th, 228Ra, 228Ac, 228Th, 224Ra
dt = 0.0002 #years
t = np.arange(0,30,dt)
c = np.zeros([np.shape(c_initial)[0],np.shape(t)[0]])
c[:,0] = c_initial

#Time loop to find amounts of radioactive compounds over time

for i in xrange(1,np.shape(t)[0]):    
    
    #if t[i]%0.2 <dt: #Every 0.2 years, flush brackish water with seawater because I can
        #c[1,i-1] = c[1,i-1]/(1+1/Kdmix)+c_eq[1]/(1+1/Kdsw)
        #c[4,i-1] = c[4,i-1]/(1+1/Kdmix)+c_eq[4]/(1+1/Kdsw)

    
    f = c[:-1,i-1]*kdecay[:-1]-c[1:,i-1]*kdecay[1:]
    f = np.insert(f,0,0) #Keep thorium 232 at a constant value
    c[:,i] = f*dt+c[:,i-1]
    
Th232 = c[0,:]
Ra228 = c[1,:]
Ac228 = c[2,:]
Th228 = c[3,:]
Ra224 = c[4,:]

f1 =plt.figure(1)
plt.clf()
p1, = plt.plot(t,Ra228,'-r')
p2, = plt.plot(t,Ac228,'-c')
p3, = plt.plot(t,Th228,'-g')
p4, = plt.plot(t,Ra224,'-b')
plt.xlabel('Time elapsed (years)')
plt.ylabel('Amount of compound (moles)')
plt.yscale('log')
plt.legend((p1,p2,p3,p4),(r'$^{228}Ra$',r'$^{228}Ac$',r'$^{228}Th$',r'$^{224}Ra$'),loc=0)
plt.show()

f2 = plt.figure(2)
plt.clf()
p21, = plt.plot(t,Ra228,'-r')
p24, = plt.plot(t,Ra224,'-b')
plt.xlabel('Time elapsed (years)')
plt.ylabel('Amount of compound (moles)')
plt.yscale('log')
plt.legend((p21,p24),(r'$^{228}Ra$',r'$^{224}Ra$'),loc=0)
plt.show()

Th232_n = c[0,:]/c_initial[0]
Ra228_n = c[1,:]/c_initial[1]
Ac228_n = c[2,:]/c_initial[2]
Th228_n = c[3,:]/c_initial[3]
Ra224_n = c[4,:]/c_initial[4]
