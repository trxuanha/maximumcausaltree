library(causalTree)
library(dplyr)

buildSynPoisCausalDTModel<- function(trainingData, causalFactor, outcomeName, splitRule='CT', cvRule = 'CT'){
  
  reg<-glm(as.formula(paste(causalFactor, ' ~. ', '-',outcomeName, sep= ""))
           , family=binomial
           , data=trainingData)
  
  propensity_scores = reg$fitted
  tree1 <- causalTree(as.formula(paste(outcomeName, ' ~. ', sep= ""))
                      , data = trainingData, treatment = trainingData[[causalFactor]],
                      split.Rule = splitRule, cv.option = cvRule, split.Honest = T, cv.Honest = T, split.Bucket = F, 
                      xval = 5, cp = 0, propensity = propensity_scores)
  
  opcp <- tree1$cptable[,1][which.min(tree1$cptable[,4])]
  tree1 <- prune(tree1, opcp)  
  results <- list()
  treeModel1<- list()
  treeModel1$model <- tree1
  treeModel1$factor <- 'F'
  results = c(results, list(treeModel1))
  
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
