# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 17:44:02 2016

@author: Michael
"""

#Script to plot surface complexation results

import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
import datetime

#Strings for the log K values used in the testing

weakK = str(-5.67)
strongK = str(7.2)

def extractData(path):
    """Function retrieves data from an excel spreadsheet, also includes the error"""
    fileLoc = path
    data = pd.read_excel(fileLoc)
    pH = data.ix[:,'pH'].values
    fSorb = data.ix[:,'fSorb'].values
    spH = data.ix[:,'spH'].values
    sfSorb = data.ix[:,'sfSorb'].values
    try:
        #totAct = data.ix[:,'Total Activity'].values
        fWeak = data.ix[:,'fWeak']
        fStrong = data.ix[:,'fStrong']
    except KeyError:
        res = {'pH':pH,'spH':spH,'fSorb':fSorb,'sfSorb':sfSorb,'data':data}
        return res
    else:
        res = {'pH':pH,'spH':spH,'fSorb':fSorb,'sfSorb':sfSorb,'data':data,'fWeak':fWeak,'fStrong':fStrong}
        return res

tetraMod = extractData('RaFHY_Sajih4SiteModel.xlsx')
SCM = extractData('RaFHY_SajihSCM.xlsx')
expData = extractData('RaFHY Experimental Data.xlsx')
#testMod = extractData('RaFHY_SajihSCM_TEST.xlsx')
exp5 = expData['data'].ix[expData['data']['Total Activity']==5,:]
exp10 = expData['data'].ix[expData['data']['Total Activity']==10,:]
exp50 = expData['data'].ix[expData['data']['Total Activity']==50,:]
exp100 = expData['data'].ix[expData['data']['Total Activity']==100,:]
exp500 = expData['data'].ix[expData['data']['Total Activity']==500,:]

f1 = plt.figure(num=1,figsize=(10,8))
plt.clf()
ax = f1.add_subplot(111)
tetplot = ax.plot(tetraMod['pH'],tetraMod['fSorb'],'-',label='Tetradentate Model, Sajih')
SCMplot = ax.plot(SCM['pH'],SCM['fSorb'],'-',label='Simple Complexation Model, Sajih')
#testplot = ax.plot(testMod['pH'],testMod['fSorb'],'-',label='Simple Complexation Model, Weak log K: '+weakK+', Strong log K: '+strongK)
#weakTest = ax.plot(testMod['pH'],testMod['fWeak'],'--',label='Weak sites')
#strTest = ax.plot(testMod['pH'],testMod['fStrong'],'--',label='Strong Sites')
exp5Plot = ax.errorbar(exp5['pH'],exp5['fSorb'],xerr=exp5['spH'],yerr=exp5['sfSorb'],fmt='o',label='Experimental Data 5 Bq Total') #Need to fix this coloring scheme
exp10Plot = ax.errorbar(exp10['pH'],exp10['fSorb'],xerr=exp10['spH'],yerr=exp10['sfSorb'],fmt='o',label='Experimental Data 10 Bq Total')
exp50Plot = ax.errorbar(exp50['pH'],exp50['fSorb'],xerr=exp50['spH'],yerr=exp50['sfSorb'],fmt='o',label='Experimental Data 50 Bq Total')
exp100Plot = ax.errorbar(exp100['pH'],exp100['fSorb'],xerr=exp100['spH'],yerr=exp100['sfSorb'],fmt='o',label='Experimental Data 100 Bq Total')
exp500Plot = ax.errorbar(exp500['pH'],exp500['fSorb'],xerr=exp500['spH'],yerr=exp500['sfSorb'],fmt='o',label='Experimental Data 500 Bq Total')
ax.legend(loc=0)
ax.set_title(str(datetime.date.today()))
ax.set_xlabel('pH')
ax.set_ylabel('Fraction Sorbed')
ax.set_ylim([-0.01,1.0])
plt.show()
#f1.savefig('Radium Complexation All.pdf',dpi=900)