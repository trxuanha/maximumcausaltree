import os
import sys
import pandas as pd

from EvaluationUtil import *

BASE_DIR = os.getcwd()
outcomeName = ''
causeFile = os.path.join(BASE_DIR, 'output/Cause/synthetic/causes.txt')
treatmentNameList, outcomeName =  populateCausesOutcome(causeFile)

def processBaselinesResults(caseName, methodFolder, postfileName):
    basePah = os.path.join(BASE_DIR, 'output', methodFolder,'synthetic', caseName)
    newImageFolder = os.path.join(BASE_DIR, 'output', methodFolder, 'synthetic',caseName, 'TopImage')
    resultLogF = pd.DataFrame({'AUC':[], 'FolderNbr':[]})
    prefileName = caseName + '_test_'
    totalTestCase = 50   
    for i in range (1, totalTestCase + 1):
        print(str(i))
        upliftPath = os.path.join(basePah, str(i))
        if not os.path.exists(newImageFolder):
            try:
                os.makedirs(newImageFolder)
            except:
                print("An exception occurred making dir: " + newImageFolder)
                
        auc = getAUCNTopPopReSort(upliftPath, 5, prefileName, postfileName, outcomeName, str(i), newImageFolder, False)
        q0 = pd.DataFrame({'AUC': auc, 'FolderNbr':i}, index =[0])
        resultLogF = pd.concat([q0, resultLogF]).reset_index(drop = True)        
    resultLogF.sort_values(by=['AUC'], ascending = False, inplace=True, axis=0)
    resultFile = os.path.join(basePah, 'AUUC.csv')  
    resultLogF.to_csv(resultFile, index=False)
    
# Process data from R
processBaselinesResults('Case1', 'CausalTree','_CT')
processBaselinesResults('Case2', 'CausalTree','_CT')
processBaselinesResults('Case3', 'CausalTree','_CT')
processBaselinesResults('Case4', 'CausalTree','_CT')

processBaselinesResults('Case1', 'TOTree','_TOT')
processBaselinesResults('Case2', 'TOTree','_TOT')
processBaselinesResults('Case3', 'TOTree','_TOT')
processBaselinesResults('Case4', 'TOTree','_TOT')

processBaselinesResults('Case1', 'TStatisticTree','_tstats')
processBaselinesResults('Case2', 'TStatisticTree','_tstats')
processBaselinesResults('Case3', 'TStatisticTree','_tstats')
processBaselinesResults('Case4', 'TStatisticTree','_tstats')

processBaselinesResults('Case1', 'FitBasedTree','_fit')
processBaselinesResults('Case2', 'FitBasedTree','_fit')
processBaselinesResults('Case3', 'FitBasedTree','_fit')
processBaselinesResults('Case4', 'FitBasedTree','_fit')

