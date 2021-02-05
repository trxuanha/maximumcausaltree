import os
import sys
from multiprocessing import Pool
import multiprocessing as mp
import pandas as pd
from EvaluationUtil import *
import warnings
warnings.filterwarnings("ignore")
maxTreeDepth = 5
minSize = 50
fold = 5
BASE_DIR = os.getcwd()
outputPath = os.path.join(BASE_DIR, 'output',  'MCT', 'synthetic','Case2')
MdataPath = os.path.join(BASE_DIR, 'input','synthetic' ,'Case2.csv')
outputName = ''
causeFile = os.path.join(BASE_DIR, 'output/Cause/synthetic/causes.txt')
treatmentNameList, outcomeName =  populateCausesOutcome(causeFile)
epsilonObList = [0.05, 0.1, 0.15]
epsilonEffList = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
epsilonTreatList = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
diffSigA = 1.5
diffSigM = 1.5

### Initial dataset
print('start process dataset')
Mdataset = pd.read_csv(MdataPath,  encoding = "ISO-8859-1", engine='python')
columsntoAproximate = list(set(Mdataset.columns) - {outcomeName, 'Index'})
for col in columsntoAproximate:
    Mdataset[col], breakpoints = binningAttributeV4(Mdataset, col, outcomeName, diffSigA)
    Mdataset[col] = Mdataset[col].astype(float)
print('end of process dataset')

def wrapperPara(x):
    print(' start of proc before build the tree ID' + str(os.getpid()))    
    epsilonOb = x[0]
    epsilonEff = x[1]
    epsilonTreat = x[2]    
    folderName = 'ep1_' + str(epsilonOb) + '_ep2_' + str(epsilonEff) + '_ep3_' + str(epsilonTreat)
    dirPath = os.path.join(outputPath, folderName)
    imagePath = os.path.join(outputPath, 'image')
    if not os.path.exists(dirPath):
        try:
            os.makedirs(dirPath)
        except:
            print("An exception occurred making dir: " + dirPath)
    if not os.path.exists(imagePath):
        try:
            os.makedirs(imagePath)
        except:
            print("An exception occurred making dir: " + imagePath)
    print('start build tree')
    outputName = 'ep1_' + str(epsilonOb) + '_ep2_' + str(epsilonEff) + '_ep3_' + str(epsilonTreat) +'_disEmp'
    epsilon = [epsilonOb, epsilonEff, epsilonTreat ]
    print(' start of proc ID' + str(os.getpid()))
    MdatasetforCross = Mdataset.copy()             
    doCrossValidationV2(MdatasetforCross, None, maxTreeDepth, minSize, 
                     dirPath, outputName, outcomeName, treatmentNameList, fold, imagePath, epsilon)                     
    print(' end of proc ID' + str(os.getpid()))
    return 1
    
inputParams = []
for epsilonOb in epsilonObList:
    for epsilonEff in epsilonEffList:
        for epsilonTreat in epsilonTreatList:
            inputParams.append((epsilonOb, epsilonEff, epsilonTreat)) 
            
if __name__ == '__main__':
    with Pool(16) as p:
        p.map(wrapperPara, inputParams)
        p.close()