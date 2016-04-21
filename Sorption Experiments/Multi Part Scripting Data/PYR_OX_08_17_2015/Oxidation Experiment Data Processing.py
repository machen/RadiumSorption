# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:44:05 2015

Oxidation experiment processing script

@author: Michael
"""
import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns

sns.set_context("talk")
colorcycle=sns.color_palette()
data = pd.read_excel("./OrganizedOxidizedData.xlsx",header=0)
fpost = plt.figure()
axpost = fpost.add_subplot(111)
for i in (2,4,6,8):
    clr = colorcycle[i/2-1]
    oxErr = axpost.errorbar(data.ix[i,['pHpre','pHpost']].values,data.ix[i,['fSorbpre','fSorbpost']].values,yerr=data.ix[i,['dfsorbpre','dfsorbpost']].values,marker='',color=clr,linestyle='-')
    #normErr = axpost.errorbar(data.ix[i+1,['pHpre','pHpost']].values,data.ix[i+1,['fSorbpre','fSorbpost']].values,yerr=data.ix[i+1,['dfsorbpre','dfsorbpost']].values,marker='',color=clr,linestyle='--')
    prePt = axpost.plot(data.ix[i,'pHpre'],data.ix[i,'fSorbpre'],'o',mfc=clr)
    postPt = axpost.plot(data.ix[i,'pHpost'],data.ix[i,'fSorbpost'],'o',mfc="None",mec=clr,mew=2)
axpost.set_xlabel('pH')
axpost.set_ylabel('Fraction Sorbed')

fpre = plt.figure()
axpre = fpre.add_subplot(111)
for i in xrange(2,9,2):
    clr = colorcycle[i/2-1]
    prePt = axpre.errorbar(data.ix[i,'pHpre'],data.ix[i,'fSorbpre'],yerr=data.ix[i,'dfsorbpre'],marker='o',color=clr)
axpre.set_ylim(axpost.get_ylim())
axpre.set_xlim(axpost.get_xlim())
axpre.set_xlabel('pH')
axpre.set_ylabel('Fraction Sorbed')


fpost.savefig('PostOxidation.png',dpi=1500)
fpre.savefig('PreOxidation.png',dpi=1500)