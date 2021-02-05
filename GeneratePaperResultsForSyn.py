
import os
import sys
import statistics 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl, matplotlib.pyplot as plt
import os.path
from os import path

from EvaluationUtil import *

AUC = True
diversification = True
BASE_DIR = os.getcwd()
baseInFolder = os.path.join(BASE_DIR, 'output')
syntheticResults = os.path.join(baseInFolder, 'PerformanceEval', 'synthetic')
outcomeName = ''
causeFile = os.path.join(BASE_DIR, 'output/Cause/synthetic/causes.txt')
treatmentNameList, outcomeName =  populateCausesOutcome(causeFile)

def summarizeResults(baseCaseFolder):
    
    imagePath = os.path.join(baseCaseFolder, 'Synimage')
    prefileName = ''
    postfileName = ''
    epsilonObList = [0.05, 0.1, 0.15]
    epsilonEffList = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    epsilonTreatList = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    resultLogF = pd.DataFrame({'AUC':[], 'Kendall':[], 'epsilonOb':[] , 'epsilonEff':[], 'epsilonTreat':[] })  
    for epsilonOb in epsilonObList:
        for epsilonEff in epsilonEffList:
            for epsilonTreat in epsilonTreatList:                
                folderName = 'ep1_' + str(epsilonOb) + '_ep2_' + str(epsilonEff) + '_ep3_' + str(epsilonTreat)                
                upliftPath = os.path.join(baseCaseFolder, folderName)
                if(not (path.exists(upliftPath) )):
                    continue
           
                dir = os.listdir(upliftPath)
                if len(dir) < 5: 
                    continue
                    
                if not os.path.exists(imagePath):        
                    try:
                        os.makedirs(imagePath)
                    except:
                        print("An exception occurred making dir: " + imagePath)
    
                prefileName = 'recommendation_' + folderName + '_disEmp_'    
                auc = getAUCNTopPopReSort(upliftPath, 5, prefileName, postfileName, outcomeName, folderName, imagePath, False)                
                kendal, variation, variance, spear = getDiversification(upliftPath, 5, prefileName, postfileName, outcomeName) 
                
                q0 = pd.DataFrame({'AUC': auc,'Kendall':kendal,'epsilonOb':epsilonOb, 'epsilonEff':epsilonEff,'epsilonTreat':epsilonTreat }, index =[0]) 
                
                resultLogF = pd.concat([q0, resultLogF]).reset_index(drop = True)      
    resultLogF.sort_values(by=['AUC'], ascending = False, inplace=True, axis=0)
    return resultLogF

def selectHyParameters(summResults):
    
    maxAUUC = summResults['AUC'].max()
    maxEffdiv = summResults['Kendall'].max()
    summResults['SCALE_AUC'] = summResults['AUC']/maxAUUC
    summResults['SCALE_Kendall'] = summResults['Kendall']*1.0/maxEffdiv
    summResults['COMBINEDMETRIC']  = summResults['SCALE_AUC']  + summResults['SCALE_Kendall']
    summResults.sort_values(by=['COMBINEDMETRIC'], ascending = False, inplace=True, axis=0)
    topRow = summResults.iloc[0:1, ]
    
    return topRow
    
    
def genGroupBarAUC(opiAuc, methodNames, filePaths, outPutPath, dataSetCount,ifigure, iaxis, subTitile, miny, maxy):
    results = pd.DataFrame({})
    AUCVal = []
    newMethodNames = []
    for fileCount in range(0, len(filePaths)):        
        tempRes = pd.read_csv(filePaths[fileCount],  encoding = "ISO-8859-1", engine='python')     
        tempRes.FolderNbr=tempRes.FolderNbr.astype(int)
        if(methodNames[fileCount] == 'CausalTree'):
            tempRes['model'] = 'CT'
            newMethodNames.append('CT')   
        if(methodNames[fileCount] == 'TOTree'):
            tempRes['model'] = 'TOT'
            newMethodNames.append('TOT') 
        if(methodNames[fileCount] == 'TStatisticTree'):
            tempRes['model'] = 'ST'
            newMethodNames.append('ST')   
        if(methodNames[fileCount] == 'FitBasedTree'):
            tempRes['model'] = 'FT'
            newMethodNames.append('FT')           
        results = pd.concat([tempRes, results]).reset_index(drop = True)
    newMethodNames.append('MCT')
    AUCVal.append(opiAuc)
    q0 = pd.DataFrame({'AUC': [opiAuc], 'FolderNbr': [1], 'model': ['MCT']}, index =[0])
    results = pd.concat([q0, results]).reset_index(drop = True)          
    my_pal = {"CT": "#3274a1", "TOT": "#e1812c", "ST":"#3a923a", "FT":"#c03d3e", "MCT":"#9372b2"}
    ax = sns.stripplot(data=results, x="model", y="AUC", palette=my_pal, 
                       ax= iaxis,
                       dodge=False, alpha=.8, zorder=1, order = ['MCT', 'CT', 'TOT', 'ST', 'FT'],
                       )
    groupedData = results.groupby(['model'])
    tempResult = pd.DataFrame({'AUC': [], 'model': []})
    for symbol, group in groupedData:    
        q0 = pd.DataFrame({'model': [symbol], 'AUC': [group['AUC'].mean()] })    
        tempResult = pd.concat([q0, tempResult]).reset_index(drop = True)
    
    my_pal = {"CT": "blue", "TOT": "blue", "ST":"blue", "FT":"blue", "MCT":"#9372b2"}
    ax =  sns.pointplot(x="model", y="AUC",
                  data=tempResult, dodge=False, join=False, 
                  palette=my_pal,
                  order = ['MCT', 'CT', 'TOT', 'ST', 'FT'],
                  markers="d", 
                  scale=3, ax= iaxis) 
    my_pal = {"CT": "#3274a1", "TOT": "#e1812c", "ST":"#3a923a", "FT":"#c03d3e", "MCT":"#9372b2"}          
    ax = sns.pointplot(data=tempResult, x="model", y="AUC", palette=my_pal, 
                         order = ['MCT', 'CT', 'TOT', 'ST', 'FT'],
                         join=False,
                         ax= iaxis,)
    
    ax.set_title(subTitile, fontsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.get_yaxis().tick_left()
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel('', fontsize=30)
    ax.set_ylabel('AUUC', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(False)
    
if(AUC):                
    dataSetCount = 1
    dataNames = []
    plt.clf()    
    figure, allAxis = plt.subplots(1, 4, figsize=(20,6),  sharey=False, dpi=300)    
    allAxis = allAxis.flatten()
    for dataSetCount in range(1, 5):
        caseFolderName = 'Case' + str(dataSetCount)        
        fullCaseFolderName = baseInFolder + '/MCT/synthetic/' + caseFolderName         
        summaryResultFile = fullCaseFolderName + '/' + 'AUUC.csv'
        if(not (path.exists(summaryResultFile) )):
            print('Summarize the results')
            summaryResults = summarizeResults(fullCaseFolderName)
        summaryResults = pd.read_csv(summaryResultFile,  encoding = "ISO-8859-1", engine='python')
        selectedParms = selectHyParameters(summaryResults)        
        epsilonOb = selectedParms.iloc[0].loc['epsilonOb']
        epsilonEff = selectedParms.iloc[0].loc['epsilonEff']
        epsilonTreat = selectedParms.iloc[0].loc['epsilonTreat'] 
        parmfolderName = 'ep1_' + str(epsilonOb) + '_ep2_' + str(epsilonEff) + '_ep3_' + str(epsilonTreat)
        prefileName = 'recommendation_' + parmfolderName + '_disEmp_'
        postfileName = ''        
        fullResultFolder = fullCaseFolderName + '/' + parmfolderName       
        opiAuc = getAUCNTopPop(fullResultFolder, 5, prefileName, postfileName, outcomeName, False)
        
        ################### baselines        
        methodNameList = ['CausalTree', 'TOTree', 'TStatisticTree', 'FitBasedTree']
        filePaths = []
        for methodName in methodNameList:
            filePath = baseInFolder + '/' + methodName + '/synthetic/' + 'Case' + str(dataSetCount)  + '/' +  'AUUC' + '.csv'
            filePaths.append(filePath)            
        subTitile = 'Scenario ' + str(dataSetCount)    
        genGroupBarAUC(opiAuc, methodNameList, filePaths, syntheticResults, dataSetCount, figure, allAxis[dataSetCount - 1], subTitile, 4, None) 
    figure.tight_layout(pad=3.0)
    plt.savefig(syntheticResults + '/'  + 'GroupAUUC.png', dpi=300)
                                   
if(diversification):
        resultLog = pd.DataFrame({'Dataset':[], 'Method':[], 'Kendall':[], 'Spearman':[]})        
        for dataSetCount in range(1, 5):            
            ## OPITree             
            caseFolderName = 'Case' + str(dataSetCount)
            fullCaseFolderName = baseInFolder + '/MCT/synthetic/' + caseFolderName 
            summaryResultFile = fullCaseFolderName + '/' + 'AUUC.csv'
            if(not (path.exists(summaryResultFile) )):
                print('Summarize the results')
                summaryResults = summarizeResults(fullCaseFolderName)
            summaryResults = pd.read_csv(summaryResultFile,  encoding = "ISO-8859-1", engine='python')
            selectedParms = selectHyParameters(summaryResults)
            epsilonOb = selectedParms.iloc[0].loc['epsilonOb']
            epsilonEff = selectedParms.iloc[0].loc['epsilonEff']
            epsilonTreat = selectedParms.iloc[0].loc['epsilonTreat']
            parmfolderName = 'ep1_' + str(epsilonOb) + '_ep2_' + str(epsilonEff) + '_ep3_' + str(epsilonTreat)
            prefileName = 'recommendation_' + parmfolderName + '_disEmp_'
            postfileName = ''          
            fullResultFolder = fullCaseFolderName + '/' + parmfolderName
            kendal, variation, variance, spear = getDiversification(fullResultFolder, 5, prefileName, postfileName, outcomeName)
            q0 = pd.DataFrame({'Dataset': dataSetCount, 'Method':'MCT', 'Kendall': kendal, 'Spearman': spear}, index =[0])
            resultLog = pd.concat([q0, resultLog]).reset_index(drop = True)                   
            ### baseline models                        
            methodNameList = ['CausalTree', 'TOTree', 'TStatisticTree', 'FitBasedTree']            
            for methodName in methodNameList:
                imprvFilePath = baseInFolder + '/' + methodName + '/synthetic/' + 'Case' + str(dataSetCount)  + '/' +  'AUUC' + '.csv'
                imprvResults = pd.read_csv(imprvFilePath,  encoding = "ISO-8859-1", engine='python')    
                ## get top 1
                topThree = imprvResults.iloc[0:1, ]    
                binDatasets = topThree['FolderNbr']
                kendals = []
                variations = []
                variances = []
                spears = []
                for binDataCount in binDatasets:    
                    folderLocation = baseInFolder + '/' + methodName + '/synthetic/' + 'Case' + str(dataSetCount)  + '/' + str(int(binDataCount))
                    folderName = 'Case' + str(dataSetCount)
                    prefileName = 'recommendation_syn_'
                    prefileName = 'Case' + str(dataSetCount)+'_test_'
                    
                    if(methodName == 'CausalTree'):
                        postfileName = '_CT'
                    if(methodName == 'TOTree'):
                        postfileName = '_TOT'                
                    if(methodName == 'TStatisticTree'):
                        postfileName = '_tstats'
                    if(methodName == 'FitBasedTree'):
                        postfileName = '_fit' 
                    kendal, variation, variance, spear = getDiversification(folderLocation, 5, prefileName, postfileName, outcomeName)
                    kendals.append(kendal)
                    variations.append(variation)
                    variances.append(variance)
                    spears.append(spear)                    
                if(methodName == 'CausalTree'):
                    methodName = 'CT'            
                if(methodName == 'TOTree'):
                    methodName = 'TOT'            
                if(methodName == 'TStatisticTree'):
                    methodName = 'ST'            
                if(methodName == 'FitBasedTree'):
                    methodName = 'FT'
                q0 = pd.DataFrame({'Dataset': dataSetCount, 'Method':methodName, 'Kendall':statistics.mean(kendals), 'Spearman': statistics.mean(spears) }, index =[0]) # Zero for binary treatment
                resultLog = pd.concat([q0, resultLog]).reset_index(drop = True)
                
                ## End of for binDatasets
            ## End of for methodNameList
        ## End for dataset
        resultLog.to_csv(syntheticResults + '/' + 'diversitication.csv',index=False)