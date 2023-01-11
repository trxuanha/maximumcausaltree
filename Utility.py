from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
import mord as m
import numpy as np
from scipy.stats import *

def convertOrdinalCategory(dataset, covariates, treatment, nbrOfQuantile):
    bin_labels = []
    
    for i in range(1, nbrOfQuantile + 1):  
        bin_labels.append(i)
        
    results, bin_edges = pd.qcut(dataset[treatment],
                                q= nbrOfQuantile,
                                labels=bin_labels,
                                retbins=True)
    dataset[treatment] = results
    return dataset, bin_edges
    
def estimateOrdinalPropensity(dataset, covariates, treatment):
        
    ##estimate propensity score here
    X = dataset[covariates]
    y = dataset[treatment]
    c = m.LogisticAT()
    c.fit(X, y)
    result = c.predict_proba(X)
    unique_vals = sorted(y.unique())   
    icount = 0
    for ival in unique_vals:
        if(icount == 0):
            icount = icount + 1
            continue  # skip the first element
        dataset [treatment + '_' + str(ival)] = 1- np.sum(result[:,0:icount ], axis = 1)
        icount = icount + 1  
    return dataset, c
    
def estimateBinaryPropensity(dataset, covariates, treatment):
    X = dataset[covariates]
    y = dataset[treatment]
    logreg = LogisticRegression()
    logreg.fit(X, y)
    result = logreg.predict_proba(X)
    dataset[treatment + '_' + str(1)] = result[:,1]  
    return dataset
    
        
def binningAttribute(dataset, attributeName, outcomeName, diffSig):
    newDataset = dataset.copy()    
    newDataset.sort_values(by=[attributeName, outcomeName], ascending = [True, True ], inplace=True, axis=0)
    newDataset = newDataset.reset_index(drop=True) 
    minStep = int(0.002*newDataset.shape[0])
    if(minStep < 5):
        minStep = 5
    mergeList = []
    ## initiate 
    sumsize = 0  
    startIndex = 0
    while sumsize < newDataset.shape[0]:
        endIndex =  startIndex + minStep -1
        currentVal = 0
        if(endIndex >= newDataset.shape[0]):
            endIndex = newDataset.shape[0]  - 1
        else:
             currentVal = newDataset[attributeName].iloc[endIndex]     
        if(endIndex < (newDataset.shape[0]  - 1)):
            icount2 = endIndex + 1
            ## search for the same attributeName value
            while ((icount2 < newDataset.shape[0] ) and (newDataset[attributeName].iloc[icount2] == currentVal)):                
                endIndex = icount2
                icount2 = icount2 + 1 
        sumsize = sumsize + (endIndex - startIndex + 1)
        mergeList.append((startIndex, endIndex))
        startIndex = endIndex + 1
    change = True
    stepCount = 1
    while(change):
        change = False
        currentTscore = 9999999
        curentSGIndex= -1
        for currentIndex in range(0, len(mergeList) - 1):
            firstGroup = mergeList[currentIndex]
            a = newDataset[outcomeName].iloc[firstGroup[0]: firstGroup[1] + 1]
            secondGroup = mergeList[currentIndex + 1]
            b = newDataset[outcomeName].iloc[secondGroup[0]: secondGroup[1] + 1]
            if(len(b) < minStep):
                curentSGIndex = currentIndex
                break   
            tscore, pscore = stats.ttest_ind(a,b)
            if((np.isnan(tscore))): ## Merge since they are the same  (abs(tscore) < diffSig)    
                curentSGIndex = currentIndex
                break
            else:
                if((abs(tscore) <= diffSig) and (abs(tscore)  < currentTscore)):
                    currentTscore = abs(tscore) 
                    curentSGIndex = currentIndex
            
        if(curentSGIndex >= 0):
            firstGroup = mergeList[curentSGIndex]
            secondGroup = mergeList[curentSGIndex + 1]
            del mergeList[curentSGIndex + 1]
            mergeList[curentSGIndex] = (firstGroup[0], secondGroup[1])
            change = True   
        stepCount = stepCount + 1
    breakPoints = np.array([])
    ## convert to breakpoints
    for icount in range (1, len(mergeList)):
        eIndex = mergeList[icount] [0]
        breakPoints = np.append (breakPoints,  newDataset[attributeName].iloc[eIndex])
    breakPoints = np.sort(breakPoints)    
    result= dataset.copy() #attributeName
    result[attributeName] = result[attributeName].apply(lambda x: 1 
                                                if x < breakPoints[0] 
                                                else (len(breakPoints) + 1 if x >= breakPoints[-1] 
                                                else np.argmin(breakPoints <= x) + 1 ))
    return result[attributeName], breakPoints 
          
def convertToOrignalBreakPoint(index, breakPoints, realbreak):
    import math
    lowInt = math.ceil(index)
    if(realbreak):
        return breakPoints[lowInt - 2]
    else:
        return breakPoints[lowInt]
         

def getAllBinaryPropensityWithMinV2(dataset, covariateNames, treatmentName):
    newDataset = dataset.copy()
    newDataset.reset_index(drop=True, inplace=True)
    orUniqueTrVals = sorted(newDataset[treatmentName].unique())
    orUniqueTrVals = np.array(orUniqueTrVals)       
    modelList = []
    propensities = []
    uniqueTrVals = (orUniqueTrVals[1:] + orUniqueTrVals[:-1]) / 2
    for uval in uniqueTrVals:     
        promodel = estimateBinaryPropensityWithMin(newDataset, covariateNames, treatmentName, uval)   
        propen = promodel.predict_proba(newDataset[covariateNames])
        propensities.append(propen[:, 1])
        modelList.append(promodel)
    return modelList, uniqueTrVals
    
def getAllBinaryPropensityWithMin(dataset, covariateNames, treatmentName, maxBin):
    newDataset = dataset.copy()
    newDataset.reset_index(drop=True, inplace=True)
    orUniqueTrVals = sorted(newDataset[treatmentName].unique())
    orUniqueTrVals = np.array(orUniqueTrVals)
    modelList = []
    propensities = []
    if(len(orUniqueTrVals) > maxBin):
            tempres, orUniqueTrVals = pd.cut(newDataset[treatmentName], maxBin, retbins = True)
            orUniqueTrVals = orUniqueTrVals[1:]
            orUniqueTrVals = orUniqueTrVals[:-1]
    uniqueTrVals = (orUniqueTrVals[1:] + orUniqueTrVals[:-1]) / 2
    for uval in uniqueTrVals:    
        promodel = estimateBinaryPropensityWithMin(newDataset, covariateNames, treatmentName, uval)   
        propen = promodel.predict_proba(newDataset[covariateNames])
        propensities.append(propen[:, 1])
        modelList.append(promodel)   
    return modelList, uniqueTrVals

def estimateBinaryPropensityWithMin(dataset, covariates, treatment, minLevel):
    newDataset = dataset.copy()
    newDataset.reset_index(drop=True, inplace=True)
    X = newDataset[covariates]
    numpyArray = newDataset[treatment].to_numpy()
    index = (numpyArray >=  minLevel)
    numpyArray[index] = 1
    numpyArray[np.logical_not(index)] = 0  
    y = numpyArray
    logreg = LogisticRegression()
    logreg.fit(X, y)     
    return logreg

def getOrdinalProba(ordinalPredictor, treatmentThreshold, individual, covariateNames):
    try:
        propen_vec = ordinalPredictor.predict_proba(individual[covariateNames])
    except:
        return 1
    propen_vec = propen_vec.flatten()
    icount = 0
    totalProba = 0
    for elem in ordinalPredictor.classes_:
        if(elem >= treatmentThreshold):
            break
        else:
            totalProba = totalProba + propen_vec[icount]
        icount = icount + 1   
    return 1 - totalProba

def inverseTransformOutcome(treatmentEffect, propensityScore, Treatment):
    if(Treatment):
        Treatment = 1
    else:
        Treatment = 0
    potentialOutcome = treatmentEffect*propensityScore* (1- propensityScore)/(Treatment - propensityScore)
    return potentialOutcome

def populateTreatments(filePath):
    file1 = open(filePath, 'r')
    Lines = file1.readlines()
    treatmentNameList = []
    outcomeName = ''
    readOutcome = False
    outcomeMarker = '==Outcome=='
    for line in Lines:
        currentVal = line.strip()
        if(outcomeMarker in currentVal):
            readOutcome = True
        elif (readOutcome):
            outcomeName = currentVal
            break
        else:
            treatmentNameList.append(currentVal)   
    return treatmentNameList, outcomeName
    