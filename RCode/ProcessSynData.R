source("SynCausalTree.R")
source("LiftUtility.R")
baselinesEvaluation <- function(fileName, testType, causalFactor, outcomeName){
  
  folderName = ''
  splitRule = ''
  cvRule = ''
  
  
  if(testType == 'CT'){
    
    folderName = 'CausalTree'
    splitRule = 'CT'
    cvRule = 'CT'
    
  }
  
  if(testType == 'TOT'){
    folderName = 'TOTree'
    splitRule = 'TOT'
    cvRule = 'TOT'
    
  }
  
  
  if(testType == 'tstats'){
    folderName = 'TStatisticTree'
    splitRule = 'tstats'
    cvRule = 'fit'
    
  }
  
  
  if(testType == 'fit'){
    folderName = 'FitBasedTree'
    splitRule = 'fit'
    cvRule = 'fit'
    
  }
  
  inBasePath <- paste (dirname(getwd()), '/input/synthetic', sep='')
  inBasePath <- paste (inBasePath,  fileName ,sep='/')
  outbase <- paste (dirname(getwd()), '/output/', folderName, '/synthetic', sep='')
  outbase <- paste (outbase, fileName , sep='/')
  totalTestCase = 50
  outCome <- 'Y'
  
  for (fileCount in 1: totalTestCase){
    inputPath = paste (inBasePath, '/',fileCount, sep='')
    
    for(timecn in 1: 5){
      training_data_file <- paste (inputPath,'/',fileName, '_train_',timecn,'.csv', sep='') 
      trainingData <-read.csv(file = training_data_file)
      CDTmodel <- buildSynPoisCausalDTModel(trainingData, causalFactor, outcomeName, splitRule= splitRule, cvRule = cvRule)
      tesing_data_file <- paste (inputPath,'/',fileName, '_test_',timecn,'.csv', sep='') 
      subDir = fileCount
      dir.create(file.path(outbase), showWarnings = FALSE)
      dir.create(file.path(outbase, subDir), showWarnings = FALSE)
      outputFolder = paste (outbase, '/',subDir, sep='')
      estimateUpLiftScore(CDTmodel,outCome, testType, tesing_data_file, outputFolder)
      
    }
    
    
  }
  
    
}

###
outcomeName = ''
causalFactor <- '' 
readOutcome = FALSE
outcomeMarker = '==Outcome=='

causeFile <- paste (dirname(getwd()), '/output/Cause/synthetic/causes.txt', sep='') 
myLines<- readLines(causeFile)


for (line in myLines){
  
  currentVal <- trimws(line) 
  
  if(grepl(outcomeMarker, currentVal))
    readOutcome <- TRUE
  else if(readOutcome){
    outcomeName <- currentVal
    break
    
  }else{
    causalFactor <- currentVal
    
  }
      
}

baselinesEvaluation('Case1', 'CT', causalFactor, outcomeName)
baselinesEvaluation('Case2', 'CT', causalFactor, outcomeName)
baselinesEvaluation('Case3', 'CT', causalFactor, outcomeName)
baselinesEvaluation('Case4', 'CT', causalFactor, outcomeName)

baselinesEvaluation('Case1', 'TOT', causalFactor, outcomeName)
baselinesEvaluation('Case2', 'TOT', causalFactor, outcomeName)
baselinesEvaluation('Case3', 'TOT', causalFactor, outcomeName)
baselinesEvaluation('Case4', 'TOT', causalFactor, outcomeName)

baselinesEvaluation('Case1', 'tstats', causalFactor, outcomeName)
baselinesEvaluation('Case2', 'tstats', causalFactor, outcomeName)
baselinesEvaluation('Case3', 'tstats', causalFactor, outcomeName)
baselinesEvaluation('Case4', 'tstats', causalFactor, outcomeName)

baselinesEvaluation('Case1', 'fit', causalFactor, outcomeName)
baselinesEvaluation('Case2', 'fit', causalFactor, outcomeName)
baselinesEvaluation('Case3', 'fit', causalFactor, outcomeName)
baselinesEvaluation('Case4', 'fit', causalFactor, outcomeName)