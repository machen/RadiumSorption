# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 11:31:22 2016

@author: Michael
"""

import numpy as np, pandas as pd

fileLocation = "RaPYR_pH7\\" #Working folder, which contains the input folder and output
dataFile = 'RaPYR_pH7_NoScript.xlsx' #Name of files
resultFile ='RaPYR_pH7 Equilibrated Data.xlsx'


def calcEquilSample(data):
    """Should feed in data for a specific sample (ie its evolution over time after collection)"""
    equilVals = []
    equilErrs = []
    if len(data) == 1:
        value = data['CPM'].values/60
        error = data['% Error'].values/100
        return value, error*value
    for index in reversed(data.index):
        newVal = data.loc[index,'CPM']/60
        newErr = data.loc[index,'% Error']/100
        if len(equilVals)==0:
            equilVals.append(newVal)
            equilErrs.append(newErr)
            continue
        if np.std(equilVals+newVal) > np.mean(equilErrs+newErr)*np.mean(equilVals+newVal):
            continue
        else:
            equilVals.append(newVal)
            equilErrs.append(newErr)
            value = np.mean(equilVals)
            error = np.mean(equilErrs)
            if error >0.1:
                return "Equilibrium Not Achieved"
    return value, value*error
    
scintRawData = pd.read_excel(fileLocation+dataFile,sheetname='Scintillation Counter Results',header=0)
equilData =  pd.DataFrame(columns=['Counts (cps)','Error (cps)'],index=scintRawData.loc[:,'Sample'].unique())

for ID in scintRawData.loc[:,'Sample'].unique():
    data = scintRawData.loc[scintRawData.loc[:,'Sample']==ID,:]
    [eqVal,eqErr] = calcEquilSample(data)
    equilData.loc[ID,'Counts (cps)'] = eqVal
    equilData.loc[ID,'Error (cps)'] = eqErr
    
writer = pd.ExcelWriter(fileLocation+resultFile)
equilData.to_excel(writer,sheet_name='Equilibrated Data')
writer.save()