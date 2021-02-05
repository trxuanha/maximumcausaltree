import numpy as np
import pandas as pd

class InterventionRecommender:
    
    def __init__(self, interventionModels):
        self.interventionModels = interventionModels    
    def makeRecommendation(self, injobSeekers, isBoosted = False, Version = 1):
        jobSeekers = injobSeekers.copy()
        jobSeekers.reset_index(drop=True, inplace=True)
        Improvement = np.array([])
        Treatment = np.array([])
        MinTreatment = np.array([])
        FollowRec    = np.array([])
        GroupdIList    = np.array([])
        
        for index, individual in jobSeekers.iterrows():
            selectedImprovement = -999
            selectedTreatment = 'NA'
            selectedMinTreatment = 0
            selectedGroupId = 0
            for model in self.interventionModels: 
                improvement, minTreatment, subset, controlMean, treatmentMean, propenScore = model.predictIndividual(individual)            
                if(improvement > selectedImprovement):
                    selectedImprovement = improvement
                    selectedTreatment = model.treatmentName
                    selectedMinTreatment = minTreatment
                    selectedGroupId = subset
                    
            # End for 1      
            Improvement = np.append (Improvement, selectedImprovement)
            Treatment = np.append (Treatment, selectedTreatment)
            MinTreatment = np.append (MinTreatment, selectedMinTreatment)
            GroupdIList = np.append (GroupdIList, selectedGroupId)
            selectFollowRec = 0
            if(selectedImprovement > 0):
                if(individual[selectedTreatment] >= selectedMinTreatment):
                    selectFollowRec = 1
                else: 
                    selectFollowRec = 0
            else:
                if(individual[selectedTreatment] >= selectedMinTreatment):
                    selectFollowRec = 0
                else: 
                    selectFollowRec = 1                   
            FollowRec = np.append (FollowRec, selectFollowRec)  
        # end for individuals
        ser = pd.Series(Improvement)
        jobSeekers['Improvement'] = ser
        ser = pd.Series(Treatment)
        jobSeekers['TREATMENT_NAME'] = ser
        ser = pd.Series(MinTreatment)
        jobSeekers['MinTreatment'] = ser
        ser = pd.Series(FollowRec)
        jobSeekers['FollowRec'] = ser  
        ser = pd.Series(GroupdIList)
        jobSeekers['GroupId'] = ser         
        return jobSeekers, np.var(Improvement)
        