# -*- coding: utf-8 -*-
"""
Spyder Editor

Script is designed to interact with the master experimental table in python
"""

import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import linregress

sns.set_context("talk")

saveData = False

data = pd.read_excel("Sorption Experiment Master Table.xlsx")
data = data.ix[data.ix[:,'Include']==True]
Cw = data.ix[:,'Cw (Bq/mL)']/60
sCw = data.ix[:,'sCw (Bq/mL)']/60
Cs = data.ix[:,'Cs (Bq/mg)']/60*1000
sCs = data.ix[:,'sCs (Bq/mg)']/60*1000
pH = data.ix[:,'Final pH']
spH = data.ix[:,'spH']
totAct = data.ix[:,'Total Activity (Bq)']/60
minlMass = data.ix[:,'Mineral Amount (mg)']/1000
solVol = data.ix[:,'Solution Volume (mL)']

subData = pd.DataFrame({'Cw':Cw,'sCw':sCw,'Cs':Cs,'sCs':sCs,'pH':pH,'spH':spH,'totAct':totAct,'minlMass':minlMass,'solVol':solVol})
FHY = subData[data['Mineral']=='FHY']
PYR = subData[data['Mineral']=='PYR']

def selectIsotherm(lowpH,highpH,dataSource):
    """Selects data and plot an isotherm within a certain pH range. Is not mineral specific currently, but could be augmented to 
    fix that.
    VARIABLES
    lowpH: Number for lower pH value
    highpH: Number for upper pH value (Note if values are switched up, code will fix them automatically)
    dataSource: DataFrame containing the relevant data, requires pH, Cw, Cs, minlMass data
    RETURNS
    Result: Dictionary containing the following keys:
        f: Matplotlib figure of the isotherm plot
        ax: Axes for the isotherm plot
        fit: Dictionary of fit parameters, containing the values 'Slope', 'Intercept', 'R2', 'pVal', and 'stdErr' 
        pHstat: Array containing the average and standard deviation of the pH values
        data: DataFrame containing the data within the pH range specified"""
    if lowpH > highpH:
        ph1 = lowpH
        ph2 = highpH
        lowpH = ph2
        highpH = ph1
    pH = dataSource.ix[:,'pH']
    Cw= dataSource.ix[:,'Cw']
    Cs = dataSource.ix[:,'Cs']
    minlMass = dataSource.ix[:,'minlMass']
    data = dataSource[pH>lowpH][pH<highpH][Cw>0][Cs>0][minlMass>0.001]#Select data in pH range and ignore any concentrations that are negative
    avgpH = np.mean(data.ix[:,'pH'].values)
    stdpH = np.std(data.ix[:,'pH'].values)
    fit = linregress(data.ix[:,'Cw'],data.ix[:,'Cs'])
    fit = {'Slope':fit[0],'Intercept':fit[1],'R2':fit[2]**2,'pVal':fit[3],'stdErr':fit[4]}
    f = plt.figure()
    plt.clf()
    ax = f.add_subplot(111)
    err = ax.errorbar(data.ix[:,'Cw'],data.ix[:,'Cs'], yerr = data.ix[:,'sCs'].values,xerr = data.ix[:,'sCw'].values, fmt = None)
    val, = ax.plot(data.ix[:,'Cw'],data.ix[:,'Cs'],"ob")
    fitplt, = plt.plot(data.ix[:,'Cw'],np.polyval((fit['Slope'],fit['Intercept']),data.ix[:,'Cw']),'-r')
    plt.xlabel('Cw (DPM/mL)')
    plt.ylabel('Cs (DPM/mg)')
    plt.legend((val,fitplt),('Data Points','Linear fit'),loc=0)
    result = {'figure':f,'axes':ax,'fit':fit,'pHstat':[avgpH,stdpH],'data':data}
    return result
    
def selectSweep(lowAct,highAct,dataSource):
    if lowAct > highAct:
        act1 = lowAct
        act2 = highAct
        lowAct = act2
        highAct = act1
    Cw= dataSource.ix[:,'Cw']
    Cs = dataSource.ix[:,'Cs']
    totAct = dataSource.ix[:,'totAct']
    minlMass = dataSource.ix[:,'minlMass']
    data = dataSource[totAct>lowAct][totAct<highAct][Cw>0][Cs>0]
    fSorb = data.ix[:,'Cs']*data.ix[:,'minlMass']/data.ix[:,'totAct']
    data['fSorb'] = pd.Series(fSorb)
    data['sfSorb'] = pd.Series(data.ix[:,'sCs']/data.ix[:,'Cs']*data.ix[:,'fSorb'])
    f = plt.figure()
    plt.clf()
    ax = f.add_subplot(111)
    err = ax.errorbar(data.ix[:,'pH'],data.ix[:,'fSorb'], yerr = data.ix[:,'sfSorb'],xerr = data.ix[:,'spH'],fmt='none')
    val, = ax.plot(data.ix[:,'pH'],data.ix[:,'fSorb'],"ob",label="Data")
    plt.xlabel('pH')
    plt.ylabel('Fraction Sorbed (Bq basis)')
    plt.legend(loc=0)
    result = {'figure':f,'axes':ax,'data':data}
    return result

def isothermOverlay(isotherms,minerals):
    f = plt.figure()
    plt.clf()
    ax = f.add_subplot(111)
    i = 0
    colorcycle = sns.color_palette()
    for isotherm in isotherms:
        data = isotherm['data']
        mineral = minerals[i]
        fit = isotherm['fit']
        colr = colorcycle[i]
        err = ax.errorbar(data.ix[:,'Cw'].values,data.ix[:,'Cs'].values, yerr = data.ix[:,'sCs'].values,xerr = data.ix[:,'sCw'].values,marker='o',linestyle='none', label = mineral+': pH '+'%0.1f, Kd: %0.1f'%(isotherm['pHstat'][0],isotherm['fit']['Slope']),color=colr)
        fitplt, = plt.plot(data.ix[:,'Cw'],np.polyval((fit['Slope'],fit['Intercept']),data.ix[:,'Cw']),linestyle='-',color=colr,label ='none')# mineral+' Linear Fit: Kd = '+'%0.2f'%isotherm['fit']['Slope'],)
        i+=1
    plt.xlabel('Cw (Bq/mL)')
    plt.ylabel('Cs (Bq/g)')
    plt.legend(loc=0)
    return {'figure':f,'ax':ax}

def sweepOverlay(sweeps,minerals):
    f = plt.figure()
    plt.clf()
    ax = f.add_subplot(111)
    i = 0
    colorcycle = sns.color_palette()
    for sweep in sweeps:
        data = sweep['data']
        mineral = minerals[i]
        colr = colorcycle[i]
        err = ax.errorbar(data.ix[:,'pH'],data.ix[:,'fSorb'], yerr = data.ix[:,'sfSorb'],xerr = data.ix[:,'spH'],marker='o',linestyle='none', label = mineral,color=colr)
        i+=1
    plt.ylabel('Fraction Sorbed')
    plt.xlabel('pH')
    plt.legend(loc=0)
    return {'figure':f,'ax':ax}

ph3FHY = selectIsotherm(2.5,3.5,FHY)
ph7FHY = selectIsotherm(6.5,7.5,FHY)
ph7PYR = selectIsotherm(6.5,7.5,PYR)
FHYSweep = selectSweep(0,20.0,FHY)
PYRSweep = selectSweep(0,100.0,PYR)
isotherms = isothermOverlay((ph7FHY,ph3FHY,ph7PYR),('Ferrihydrite','Ferrihydrite','Pyrite'))
sweeps = sweepOverlay((FHYSweep,PYRSweep),('Ferrihydrite','Pyrite'))
if saveData == True:
    isotherms['figure'].savefig('MasterTablePlots\Isotherm ACS 2015.png',dpi=1500,format='png')
    sweeps['figure'].savefig('MasterTablePlots\Sweeps ACS 2015.png',dpi=1500,format='png')