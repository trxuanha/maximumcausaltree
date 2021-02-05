
source("SynCausalTree.R")
source("LiftUtility.R")


baselinesEvaluation <- function(fileName, testType){
  
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
      CDTmodel <- buildSynPoisCausalDTModel(trainingData, splitRule= splitRule, cvRule = cvRule)
      tesing_data_file <- paste (inputPath,'/',fileName, '_test_',timecn,'.csv', sep='') 
      subDir = fileCount
      dir.create(file.path(outbase), showWarnings = FALSE)
      dir.create(file.path(outbase, subDir), showWarnings = FALSE)
      outputFolder = paste (outbase, '/',subDir, sep='')
      estimateUpLiftScore(CDTmodel,outCome, testType, tesing_data_file, outputFolder)
      
    }
    
    
  }
  
    
}



baselinesEvaluation('Case1', 'CT')
baselinesEvaluation('Case2', 'CT')
baselinesEvaluation('Case3', 'CT')
baselinesEvaluation('Case4', 'CT')

baselinesEvaluation('Case1', 'TOT')
baselinesEvaluation('Case2', 'TOT')
baselinesEvaluation('Case3', 'TOT')
baselinesEvaluation('Case4', 'TOT')

baselinesEvaluation('Case1', 'tstats')
baselinesEvaluation('Case2', 'tstats')
baselinesEvaluation('Case3', 'tstats')
baselinesEvaluation('Case4', 'tstats')

baselinesEvaluation('Case1', 'fit')
baselinesEvaluation('Case2', 'fit')
baselinesEvaluation('Case3', 'fit')
baselinesEvaluation('Case4', 'fit')



