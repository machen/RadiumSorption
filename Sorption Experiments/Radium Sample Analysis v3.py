# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:50:12 2015

@author: Michael
"""

#Script meant to supplant original radium analysis set. Works with template Excel file. Script plots results only for initial verification, use  master table script.

import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns, re
from scipy.stats import linregress

sns.set_context("poster")
saveData = True #Flag to control data overwrite
fileLocation = "RaFHY_pH9\\" #Working folder, which contains the input folder and output
dataFile = 'RaFHY_pH9.xlsx' #Name of files
resultFile ='RaFHY_pH9 Result.xlsx'

paramSource = pd.read_excel(fileLocation+dataFile,sheetname='Parameters',index_col=0,header=0)

class Parameter:
    def __init__(self,value,error=None,units=None):
        self.value = value
        self.error = error
        self.units = units
    def __repr__(self):
        return "Parameter()"
    def __str__(self):
        return " %0.2f +/- %0.2f %s"%(self.value,self.error,self.units)

def calcEquilSample(data):
    """Should feed in data for a specific sample (ie its evolution over time after collection)"""
    equilVals = []
    equilErrs = []
    for index in reversed(data.index):
        newVal = data.loc[index,'CPM']/60
        newErr = data.loc[index,'% Error']/100
        if len(equilVals)==0:
            equilVals.append(newVal)
            equilErrs.append(newErr)
            continue
        if np.std(equilVals+newVal) > np.mean(equilErrs+newErr)*np.mean(equilVals+newVal):
            break
        else:
            equilVals.append(newVal)
            equilErrs.append(newErr)
            value = np.mean(equilVals)
            error = np.mean(equilErrs)
            if error >0.1:
                return "Equilibrium Not Achieved"
    return value, value*error

def convertActivity(sampCt,calibrationData):
    """Function takes counts, and returns activities using units defined by calibrationData, along with error (assumes each sample is its own replicate)"""
    calx = calibrationData.ix[:,1]
    caly = calibrationData.ix[:,2]
    slope,incpt,r,pval,stderr = linregress(calx,caly)
    sampAct = (sampCt-incpt)/slope
    resErr = np.sum((caly-slope*calx-incpt)**2/(len(calx)-2))
    dsampAct = resErr/abs(slope)*np.sqrt(1/len(calx)+1+(sampCt-np.mean(caly))**2/(slope**2*np.sum((calx-np.mean(calx))**2)))
    return sampAct, dsampAct

#Load in sample data and data from scintillation counter

scintRawData = pd.read_excel(fileLocation+dataFile,sheetname='Scintillation Counter Results',header=0)
equilData = pd.read_excel(fileLocation+dataFile,sheetname='Equilibrated Data',index_col=0,header=0)
bottleData = pd.read_excel(fileLocation+dataFile,sheetname='Bottle Results',index_col=0,header=0)
calData = pd.read_excel(fileLocation+dataFile,sheetname = 'Calibration Info',header=0)

for ID in scintRawData.loc[:,'Sample'].unique():
    data = scintRawData.loc[scintRawData.loc[:,'Sample']==ID,:]
    [eqVal,eqErr] = calcEquilSample(data)
    equilData.loc[ID,'Scintillation volume (mL)']
    equilData.loc[ID,'Scintillation volume error (mL)']
    equilData.loc[ID,'Counts (cps)'] = eqVal
    equilData.loc[ID,'Error (cps)'] = eqErr

#Load parameters
sampVol = Parameter(paramSource.ix['Sample Volume','Value'],paramSource.ix['Sample Volume','Error'],paramSource.ix['Sample Volume','Units'])
stockAct = Parameter(paramSource.ix['Stock Activity','Value'],error=paramSource.ix['Stock Activity','Error'],units = paramSource.ix['Stock Activity','Units'])

if paramSource.loc['Independent Cs?','Value']:
    pat = re.compile('([0-9a-zA-Z]*\_[a-zA-Z])\_C([s,w])')
    sampleAct,dsampleAct = convertActivity(equilData.ix[:,'Counts (cps)'],calData) #Activity of sample (Bq/mL)
    for ID in equilData.index:        
        try:
            bottleID = re.search(pat,ID).groups()[0]
            vialType = re.search(pat,ID).groups()[1]
        except AttributeError:
            print 'Pattern Match Failure: '+ID       
            continue
        count = equilData.loc[ID,'Counts (cps)']
        dCount = equilData.loc[ID,'Error (cps)']
        vol = equilData.loc[ID,'Scintillation volume (mL)']
        dvol = equilData.loc[ID,'Scintillation volume error (mL)']
        sldConc = equilData.loc[ID,'Solid Concentration (g/L)']
        dsldConc = equilData.loc[ID,'Solid Concentration Error (g/L)']
        #Calculate Cs and Cw from scintillation counter/ferrozine results        
        if vialType == 'w':
            Cw = count/vol
            dCw = np.sqrt((dCount/count)**2+(dvol/vol)**2)*count/vol
        elif vialType == 's':
            Cs = count/vol/sldConc
            dCs = np.sqrt((dCount/count)**2+(dvol/vol)**2+(dsldConc/sldConc)**2)*count/vol/sldConc
        #Calculate total activity based on Cw and Cs
        minlMass = bottleData.loc[bottleID,'Mineral Mass (g)']
        dminlMass = bottleData.loc[bottleID,'Mineral Mass Error (g)']
        totAct = Cw*sampVol.value+Cs*minlMass        
        dtotAct =np.sqrt(((dCw/Cw)**2+(sampVol.error/sampVol.value)**2)*(Cw*sampVol.value)**2+((dCs/Cs)**2+(dminlMass/minlMass)**2)*(Cs*minlMass)**2)
        bottleData.loc[bottleID,'Cw (Bq/mL)'] = Cw
        bottleData.loc[bottleID, 'dCw (Bq/mL)'] = dCw
        bottleData.loc[bottleID,'Cs (Bq/g)'] = Cs
        bottleData.loc[bottleID,'dCs (Bq/g)'] =dCs
        bottleData.loc[bottleID,'Total Activity (Bq)'] = totAct
        bottleData.loc[bottleID,'dTotal Activity (Bq)'] = dtotAct
else:
    #Assumes that all reported samples values are only for the experiment supernatent
    sampleAct,dsampleAct = convertActivity(equilData.ix[:,'Counts (cps)'],calData) #Activity of sample (Bq/mL)
    scintVol = equilData.loc[:,'Scintillation volume (mL)'] #Volume mixed with scintillation fluid
    dscintVol = equilData.loc[:,'Scintillation volume error (mL)']
    Cw = sampleAct/scintVol #Since all samples are only the supernatent, no sorting is required
    dCw = np.sqrt((dscintVol/scintVol)**2+(dsampleAct/sampleAct)**2)*Cw
    #Calculate total activity by knowing the amount of stock that was added
    stockVol = bottleData.ix[:,'Stock Volume Added (mL)'].values 
    dstockVol = bottleData.ix[:,'Stock Vol Err (mL)'].values
    totAct = stockVol * stockAct.value
    dtotAct = np.sqrt((dstockVol/stockVol)**2+(stockAct.error/stockAct.value)**2)*totAct
    #Calculate Cs by mass balance
    minlMass = bottleData.loc[:,'Mineral Mass (g)']
    dminlMass = bottleData.loc[:,'Mineral Mass Error (g)']    
    Cs = (totAct-Cw*sampVol.value)/minlMass
    dCs = Cs*np.sqrt((np.sqrt(dtotAct**2+(np.sqrt((dCw/Cw)**2+(sampVol.error/sampVol.value)**2)*Cw*sampVol.value)**2)/(totAct-Cw*sampVol.value))**2+(dminlMass/minlMass)**2)

    #Write the bottle data into the bottle table    
    bottleData.loc[:,'Cw (Bq/mL)'] = Cw
    bottleData.loc[:,'dCw (Bq/mL)'] = dCw
    bottleData.loc[:,'Total Activity (Bq)'] = totAct
    bottleData.loc[:,'dTotal Activity (Bq)'] = dtotAct
    bottleData.loc[:,'Cs (Bq/g)'] = Cs
    bottleData.loc[:,'dCs (Bq/g)'] = dCs
    
#At this point, bottleData should be fully populated
    
#Average exisiting data by using the established pattern
pat = re.compile('([0-9a-zA-Z]*)\_([a-zA-Z])(\_[0-9a-zA-Z]*)?') #pattern searches for THINGS_LETTER_THINGS, essentially looking for samples. Pattern used for sample averaging.
nRows = np.size(equilData,0)
expIDs = [] #List of sample IDs with multiplicate notation removed

#Convert sample IDs from multiple notation to single notation (to combine multiplicates), adding them to the equilData table
for testID in equilData.index:
    IDmatches = re.search(pat,testID)
    if IDmatches == None or len(IDmatches.groups())==1:
        continue
    elif IDmatches.groups()[2]!=None:
        expID = IDmatches.group(1)+IDmatches.group(3)
    else:
        expID = IDmatches.group(1)
        expIDs.append(expID)
equilData['expID'] = pd.Series(expIDs,index=equilData.index)
#Use the generated multiplicate names as an index for averaging values

procData = pd.DataFrame(np.zeros((len(set(expIDs)),9)),index=set(expIDs),columns=['Cw','sCw','Cs','sCs','pH','spH','totAct','stotAct','nVals'])

for ID in procData.index:
    idPat = re.compile('('+ID+')\_([a-zA-Z])(\_[0-9a-zA-Z]*)?')
    pHvals = []
    Cwvals = []
    Csvals = []
    totActVals = []
    dtotActVals = []
    fSorbvals = []
    for sample in equilData.index:
        if bottleData.ix[sample,'Include'] == False:
            continue
        if idPat.match(sample):
            pHvals.append(bottleData.ix[sample,'Final pH'])
            Cwvals.append(bottleData.ix[sample,'Cw (Bq/mL)'])
            Csvals.append(bottleData.ix[sample,'Cs (Bq/g)'])
            totActVals.append(bottleData.ix[sample,'Total Activity (Bq)'])
            dtotActVals.append(bottleData.ix[sample,'dTotal Activity (Bq)'])
    procData.loc[ID,'pH'] = np.mean(pHvals)
    procData.loc[ID,'spH'] = np.std(pHvals)
    procData.loc[ID,'Cw'] = np.mean(Cwvals)
    procData.loc[ID,'sCw'] = np.std(Cwvals)
    procData.loc[ID,'Cs']= np.mean(Csvals)
    procData.loc[ID,'sCs'] = np.std(Csvals)
    procData.loc[ID,'totAct'] = np.mean(totActVals)
    procData.loc[ID,'stotAct'] = max([np.std(totActVals),np.mean(dtotActVals)]) #Either report the max of variation in the total activity calculated by addition or by error propogation
    procData.loc[ID,'nVals'] = len(pHvals)

f1 = plt.figure(1)
plt.clf()
ax1 = f1.add_subplot(111)
colors = sns.color_palette()
dataErr = ax1.errorbar(Cw,Cs,xerr=dCw,yerr=dCs,marker='.',linestyle = 'none',color=colors[0],label='Raw Data')
avgErr = ax1.errorbar(procData.loc[:,'Cw'],procData.loc[:,'Cs'],yerr=procData.loc[:,'sCs'],xerr=procData.loc[:,'sCw'],marker='.',linestyle='none',label='Averaged Data',color=colors[1])
plt.xlabel('Cw (Bq/mL)') 
plt.ylabel('Cs (Bq/mg)')
plt.legend(loc=0)


#Write averaged data to new file    
if saveData:
    writer = pd.ExcelWriter(fileLocation+resultFile)
    paramSource.to_excel(writer,sheet_name='Parameters')
    bottleData.to_excel(writer,sheet_name='Bottle Results')
    equilData.to_excel(writer,sheet_name='Equilibrated Data',index_label = 'Sample ID')
    procData.to_excel(writer,sheet_name='Result',index_label = 'Sample')
    writer.save()
    f1.savefig(fileLocation+'CwvCsPlot.png',dpi=1500,format='png')
