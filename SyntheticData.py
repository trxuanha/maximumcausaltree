import numpy as np
from scipy.stats import *
import os
import pandas as pd
from scipy.special import expit, logit
from sklearn.preprocessing import PolynomialFeatures
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from EvaluationUtil import *

def genScenario1(fileNamePath):
       
    varNbr = 20
    varSample = 2000 
    mu, sigma = 0,1
    np.random.seed(20)
    ## covariates
    X = np.random.normal(mu, sigma, size=(varSample, varNbr)) 
    
    ## Causal factor F
    p = 0.3
    Cof1 = np.random.binomial(1, p, varNbr)
    preF1 = X*X*Cof1    
    Cof2 = np.random.normal(size= varNbr) + 0.35
    preF2 = np.dot(preF1, Cof2)    
    F = np.maximum(preF2, np.zeros((varSample)))
    F = F.reshape(varSample, 1)
    
    ## generate Y
    XF = np.hstack((X, F) )
    Cof3 = np.random.binomial(1, p, varNbr)        
    Cof3 = np.hstack((Cof3, 1) )
    preY1 = XF*Cof3
    
    Cof4 = np.random.normal(size=varNbr)
    Cof4 = np.hstack((Cof4, 9) )
    preY2 = np.dot(preY1, Cof4)
    Y = np.random.binomial(1, expit(preY2*preY2 -2))
    data = pd.DataFrame({})
    
    for icount in range(0, varNbr):    
        colName = 'X' + str(icount)
        data[colName] = X[:, icount]
        
    data['F'] = F
    data['Y'] = Y
    
    data.to_csv(fileNamePath, index=False)      
    
##################
def genScenario2(fileNamePath):
    
    varNbr = 20
    varSample = 2000
    mu, sigma = 0,1
    np.random.seed(20)
    ## covariates
    X = np.random.normal(mu, sigma, size=(varSample, varNbr)) 
    ## Causal factor F
    p = 0.3
    cof1 = np.random.binomial(1, p, varNbr)    
    preF1 = X*X*cof1    
    cof2 = np.random.normal(size= varNbr) + 0.35
    #########################    
    poly = PolynomialFeatures(degree = 2, interaction_only=True)
    InterMax = poly.fit_transform(X)    
    cof3 = np.random.binomial(1, p, InterMax.shape[1])
    
    preF2 = InterMax*cof3    
    cof4 = np.random.normal(size= InterMax.shape[1]) + 0.1
        
    #########################
    preTotal = np.hstack((preF1, preF2))
    cofTotal = np.hstack((cof2, cof4))    
    preF = np.dot(preTotal, cofTotal)
    F = np.maximum(preF, np.zeros((varSample)))    
    F = F.reshape(varSample, 1)    
    ## generate Y
    XF = np.hstack((X, F) )    
    cof5 = np.random.binomial(1, p, varNbr)            
    cof5 = np.hstack((cof5, 1) )    
    preY1 = XF*cof5    
    cof6 = np.random.normal(size=varNbr)
    cof6 = np.hstack((cof6, 9) )    
    preY1Total = np.dot(preY1, cof6)
    preY1Total = preY1Total*preY1Total -2
     
    ############################## Interaction terms
    poly = PolynomialFeatures(degree = 2, interaction_only=True)
    InterMax = poly.fit_transform(XF)        
    cof7 = np.random.binomial(1, p, InterMax.shape[1])    
    preY2 = InterMax*cof7    
    cof8 = np.random.normal(size= InterMax.shape[1]) 
    preY2Total = np.dot(preY2, cof8)    
    ##############################
    preY =  preY1Total + np.tan(preY2Total) - 1.5    
    Y = np.random.binomial(1, expit(preY))    
    data = pd.DataFrame({})    
    
    for icount in range(0, varNbr):    
        colName = 'X' + str(icount)
        data[colName] = X[:, icount]    
    
    data['F'] = F
    data['Y'] = Y
    data.to_csv(fileNamePath, index=False)   

###################    
def genScenario3(fileNamePath):
    
    varNbr = 50
    varSample = 2000
    mu, sigma = 0,1
    np.random.seed(20)
    
    ## covariates
    X = np.random.normal(mu, sigma, size=(varSample, varNbr)) 
    
    ## Causal factor F
    p = 0.3
    cof1 = np.random.binomial(1, p, varNbr)
    preF1 = X*X*cof1    
    cof2 = np.random.normal(size= varNbr) + 0.35
        
    #########################
    
    poly = PolynomialFeatures(degree = 2, interaction_only=True)
    InterMax = poly.fit_transform(X)
    cof3 = np.random.binomial(1, p, InterMax.shape[1])
    preF2 = InterMax*cof3
    cof4 = np.random.normal(size= InterMax.shape[1]) + 0.1
        
    #########################
    
    preTotal = np.hstack((preF1, preF2))
    cofTotal = np.hstack((cof2, cof4))
    preT = np.dot(preTotal, cofTotal)
    F = np.maximum(preT, np.zeros((varSample)))
    F = F.reshape(varSample, 1)
    ## generate Y
    XF = np.hstack((X, F) )
    
    cof5 = np.random.binomial(1, p, varNbr)        
    cof5 = np.hstack((cof5, 1) )
    preY1 = XF*cof5
    
    cof6 = np.random.normal(size=varNbr)
    cof6 = np.hstack((cof6, 9) )
    
    preY1Total = np.dot(preY1, cof6)
    preY1Total = preY1Total*preY1Total -2
     
    ############################## Interaction terms
    poly = PolynomialFeatures(degree = 2, interaction_only=True)

    InterMax = poly.fit_transform(XF)    
    
    cof7 = np.random.binomial(1, p, InterMax.shape[1])
    preY2 = InterMax*cof7
    cof8 = np.random.normal(size= InterMax.shape[1]) 
    preY2Total = np.dot(preY2, cof8)
    
    ##############################
    preY =  preY1Total - 5.5*((np.tan(preY2Total))**2) 
    
    Y = np.random.binomial(1, np.where(expit(preY) > 0.7, 0.7, expit(preY))   )
        
    data = pd.DataFrame({})
        
    for icount in range(0, varNbr):
        
        colName = 'X' + str(icount)
        data[colName] = X[:, icount]
        
    data['F'] = F    
    data['Y'] = Y
        
    data.to_csv(fileNamePath, index=False)   

###################################        
def genScenario4(fileNamePath):
    
    varNbr = 50
    varSample = 2000
    mu, sigma = 0,1
    np.random.seed(20)
    ## covariates
    X = np.random.normal(mu, sigma, size=(varSample, varNbr)) 
    
    ## Causal factor F
    p = 0.3
    cof1 = np.random.binomial(1, p, varNbr)    
    preF1 = X*X*cof1    
    cof2 = np.random.normal(size= varNbr) + 0.35
        
    #########################
    poly = PolynomialFeatures(degree = 2, interaction_only=True)
    InterMax = poly.fit_transform(X)
    cof3 = np.random.binomial(1, p, InterMax.shape[1])
    preF2 = InterMax*cof3
    cof4 = np.random.normal(size= InterMax.shape[1]) + 0.1
        
    #########################
    
    preTotal = np.hstack((preF1, preF2))
    cofTotal = np.hstack((cof2, cof4))
    
    preF = np.dot(preTotal, cofTotal)
        
    
    preF2= np.maximum(preF, np.zeros((varSample)))
    
    F = np.random.poisson(preF2) ## Mod this
    
    F = F.reshape(varSample, 1)
    
    ## generate Y
    XF = np.hstack((X, F) )
    
    cof5 = np.random.binomial(1, p, varNbr)        
    cof5 = np.hstack((cof5, 1) )
    preY1 = XF*cof5
    
    cof6 = np.random.normal(size=varNbr)
    cof6 = np.hstack((cof6, 9) )
    
    preY1Total = np.dot(preY1, cof6)
    preY1Total = preY1Total*preY1Total -2
     
    ############################## Interaction terms
    poly = PolynomialFeatures(degree = 2, interaction_only=True)

    InterMax = poly.fit_transform(XF)    
    
    cof7 = np.random.binomial(1, p, InterMax.shape[1])
    
    preY2 = InterMax*cof7
    
    cof8 = np.random.normal(size= InterMax.shape[1]) 

    preY2Total = np.dot(preY2, cof8)
    
    ##############################
    preY =  preY1Total - 5.5*((np.tan(preY2Total))**2) 
    Y = np.random.binomial(1, np.where(expit(preY) > 0.7, 0.7, expit(preY))   )
    data = pd.DataFrame({})
    
    for icount in range(0, varNbr):
        colName = 'X' + str(icount)
        data[colName] = X[:, icount]
        
    data['F'] = F
    data['Y'] = Y
    
    data.to_csv(fileNamePath, index=False)
    
def generatBinfromContinuousTreatmentData(inFilePath, outFolder, treatment, fromIndex,toIndex , nbrOfQuantile = 50, normalize=False):
    
    dataset = pd.read_csv(inFilePath,  encoding = "ISO-8859-1", engine='python')
    
    Mdataset = dataset.copy()
    
    Mdataset.sort_values(by=[treatment], inplace=True, axis=0)
    
    Mdataset = Mdataset.reset_index(drop=True)
    
    from10PerIndex = int(Mdataset.shape[0]*fromIndex)
    
    to90PerIndex = int(Mdataset.shape[0]*toIndex)
    
    Mdataset = Mdataset.iloc[from10PerIndex: to90PerIndex, :]    
    
    results, bin_edges = pd.cut(Mdataset[treatment],
                                    nbrOfQuantile-1,
                                    retbins=True)
    
    bsFileName = os.path.basename(inFilePath)
    bsFileName = os.path.splitext(bsFileName)[0]
    
    icount = 1
    
    for minLevel in bin_edges:
        
        Mdataset = dataset.copy()
        numpyArray = Mdataset[treatment].to_numpy().copy()
    
        
        dirPath = os.path.join(outFolder, str(icount))
        
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)       
        
        index = (numpyArray >=  minLevel)
        
        numpyArray[index] = 1
        numpyArray[np.logical_not(index)] = 0  
        
        Mdataset[treatment] = numpyArray
        
        icount = icount + 1
        
        
        fullFilePath = os.path.join(dirPath, bsFileName + '_bin.csv')
        
        if(normalize):
            Mdataset = (Mdataset - Mdataset .min())/(Mdataset .max()- Mdataset.min())
        
        
        Mdataset.to_csv(fullFilePath, index=False)
        
        splitData(Mdataset, 10, dirPath, bsFileName)
        