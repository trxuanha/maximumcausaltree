library(stringr)
library(ggplot2)
library(tidyr)
library(reshape)
library(ggthemes)

estimateUpLiftScore<- function(model, outComeColName, estimationType, infileName, outputFileFolder, manipulableAttribute=NULL, exceptAttrbute = c()){
  
  data <-read.csv(file = infileName)
  
  data['LIFT_SCORE'] = 0
  data['TREATMENT_NAME'] = ''
  data['UPLIFT'] = 0
  data ['Y_TREATED'] = 0
  data ['N_TREATED'] = 0
  data ['Y_UNTREATED'] = 0
  data ['N_UNTREATED'] = 0
  data ['FOLLOW_REC'] = 0
  
  for(row in 1: nrow(data)){
    
    inputRow = dplyr::select (data[row, ], -c('LIFT_SCORE', 'TREATMENT_NAME','UPLIFT',
                                              'Y_TREATED','N_TREATED','Y_UNTREATED','N_UNTREATED', outComeColName, exceptAttrbute ))
    
    val <- estimateCDTLiftScore(model, inputRow )
    val <- unlist(val)
    data[row,'LIFT_SCORE'] <- as.numeric(val[1])
    data[row,'TREATMENT_NAME'] <- val[2]
    
  }
  
  data$ABS_LIFT_SCORE <- abs(data$LIFT_SCORE)
  
  data <- data[order(-data$ABS_LIFT_SCORE),]
  
  y_treated <- 0
  n_treated <- 0
  y_untreated <- 0
  n_untreated <- 0
  
  for(row in 1: nrow(data)){
    
    TREATMENT_NAME = data[row,'TREATMENT_NAME']
    TREATMENT_NAME <- toString(TREATMENT_NAME)
    
    data[row,'N_TREATED']<- n_treated
    data[row,'Y_TREATED']<- y_treated
    data[row,'N_UNTREATED']<- n_untreated
    data[row,'Y_UNTREATED']<- y_untreated
    
    
    if((TREATMENT_NAME != 'NA')&&(
      
      ((data[row,TREATMENT_NAME] == 1) && (data[row,'LIFT_SCORE'] > 0))
      
      ||
      
      ((data[row,TREATMENT_NAME] == 0) && (data[row,'LIFT_SCORE'] <= 0))
      
      )
  
       ){
      
      n_treated <- n_treated + 1
      data[row,'N_TREATED']<- n_treated
      y_treated <- y_treated + data[row,outComeColName]
      data[row,'Y_TREATED']<- y_treated
      
    }else{
      n_untreated <- n_untreated + 1
      data[row,'N_UNTREATED']<- n_untreated
      y_untreated <- y_untreated + data[row,outComeColName]
      data[row,'Y_UNTREATED']<- y_untreated
    }
    
    if(n_treated == 0) {
      data[row,'UPLIFT'] = 0
      
    }else if(n_untreated == 0){
      
      data[row,'UPLIFT'] = 0
      
    }else{
      
      liftestimate = ((y_treated/n_treated) - (y_untreated/n_untreated) )*(n_treated + n_untreated)
      qiniestimate = ((y_treated) - (y_untreated*(n_treated/n_untreated) ))
      data[row,'UPLIFT'] <- liftestimate
      
    }
    
  }
  
  totalIncrease <- ((y_treated/n_treated) - (y_untreated/n_untreated) )
  
  for(row in 1: nrow(data)){
    
    n_treated <- data[row,'N_TREATED']
    y_treated <- data[row,'Y_TREATED']
    n_untreated <- data[row,'N_UNTREATED']
    y_untreated <- data[row,'Y_UNTREATED']
    liftestimate <- (((y_treated/n_treated) - (y_untreated/n_untreated) ))
    liftestimateWithBase <- (((y_treated/n_treated) - (y_untreated/n_untreated) ))/totalIncrease
    data[row,'UPLIFT'] <- liftestimate
  }
  
  #####Output
  fileName = basename(infileName)
  fileNameParts <- strsplit(fileName ,'\\.')
  fileNameParts <- unlist(fileNameParts)
  newFileName <- paste(c(fileNameParts[1],'_',estimationType,'.', fileNameParts[2]), collapse = "")
  fullPath <- paste(c(outputFileFolder,'/',newFileName ), collapse = "")
  write.csv(data,fullPath, row.names = FALSE)
  
}

estimateCDTLiftScore <-function(models, recordForEstimate){
    
  prevLift = -9999
  treatmentName = 'NA'
  result = 0
  
  for(i in 1: length(models)){
    treeModel = models[[i]]  
    result <- predict(treeModel$model, recordForEstimate)
    if(prevLift < result){
      prevLift <- result
      treatmentName <- treeModel$factor
    }
  }
  return (list(prevLift, treatmentName))  
}