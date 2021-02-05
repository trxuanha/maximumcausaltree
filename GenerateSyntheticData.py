from SyntheticData import *

generateCase1 = True
generateBinCase1 = True
generateCase2 = True
generateBinCase2 = True
generateCase3 = True
generateBinCase3 = True
generateCase4 = True
generateBinCase4 = True

BASE_DIR = os.getcwd()
basePath = os.path.join(BASE_DIR, 'input', 'synthetic')

if(generateCase1):
    fileName = 'Case1.csv'    
    fileNamePath = basePath + '/' + fileName   
    genScenario1(fileNamePath) 
    
if(generateBinCase1):
    fileName = 'Case1'
    inFilePath = basePath + '/' + fileName +'.csv' 
    outPath = basePath + '/' + fileName
    treatment = 'F'
    nbrOfQuantile = 50
    generatBinfromContinuousTreatmentData(inFilePath, outPath, treatment, 0.65, 0.9, nbrOfQuantile) ## over 60% of T has value of 0
  
if(generateCase2):
    fileName = 'Case2.csv'
    fileNamePath = basePath + '/' + fileName
    genScenario2(fileNamePath) 
    
if(generateBinCase2):
    fileName = 'Case2'
    inFilePath = basePath + '/' + fileName +'.csv' 
    outPath = basePath + '/' + fileName
    treatment = 'F'
    nbrOfQuantile = 50
    generatBinfromContinuousTreatmentData(inFilePath, outPath, treatment, 0.6, 0.9, nbrOfQuantile) 
    
if(generateCase3):
    fileName = 'Case3.csv'    
    fileNamePath = basePath + '/' + fileName
    genScenario3(fileNamePath) 
    
if(generateBinCase3):
    fileName = 'Case3'
    inFilePath = basePath + '/' + fileName +'.csv' 
    outPath = basePath + '/' + fileName
    treatment = 'F'
    nbrOfQuantile = 50
    generatBinfromContinuousTreatmentData(inFilePath, outPath, treatment, 0.4, 0.9, nbrOfQuantile) 
    
if(generateCase4):
    fileName = 'Case4.csv'    
    fileNamePath = basePath + '/' + fileName    
    genScenario4(fileNamePath) 
    
if(generateBinCase4):
    fileName = 'Case4'  
    inFilePath = basePath + '/' + fileName +'.csv' 
    outPath = basePath + '/' + fileName
    treatment = 'F'
    nbrOfQuantile = 50
    generatBinfromContinuousTreatmentData(inFilePath, outPath, treatment, 0.4, 0.9, nbrOfQuantile, True) 