from EvaluationUtil import *

BASE_DIR = os.getcwd()
basePah = os.path.join(BASE_DIR, 'output', 'MCT', 'synthetic', 'Case3')
imagePath = os.path.join(basePah, 'Synimage')
prefileName = 'dis_empl_test_'
postfileName = ''
causeFile = os.path.join(BASE_DIR, 'output/Cause/synthetic/causes.txt')
treatmentNameList, outcomeName =  populateCausesOutcome(causeFile)
epsilonObList = [0.05, 0.1, 0.15]
epsilonEffList = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
epsilonTreatList = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
resultLogF = pd.DataFrame({'AUC':[], 'Kendall':[],'Spearman':[],'epsilonOb':[] , 'epsilonEff':[], 'epsilonTreat':[]})
vcount = 0
for epsilonOb in epsilonObList:
    for epsilonEff in epsilonEffList:
        for epsilonTreat in epsilonTreatList:            
            folderName = 'ep1_' + str(epsilonOb) + '_ep2_' + str(epsilonEff) + '_ep3_' + str(epsilonTreat)
            upliftPath = os.path.join(basePah, folderName)
            if(not (path.exists(upliftPath) )):
                continue
            dir = os.listdir(upliftPath)
            if len(dir) == 0: 
                continue    
            if not os.path.exists(imagePath):
                try:
                    os.makedirs(imagePath)
                except:
                    print("An exception occurred making dir: " + imagePath)        
            prefileName = 'recommendation_' + folderName + '_disEmp_'
            vcount = vcount + 1
            auc = getAUCNTopPopReSort(upliftPath, 5, prefileName, postfileName, outcomeName, folderName, imagePath)
            kendal, variation, variance, spear = getDiversification(upliftPath, 5, prefileName, postfileName, outcomeName)
            q0 = pd.DataFrame({'AUC': auc,'Kendall':kendal,'Spearman':spear ,'epsilonOb':epsilonOb, 'epsilonEff':epsilonEff,'epsilonTreat':epsilonTreat }, index =[0]) 
            resultLogF = pd.concat([q0, resultLogF]).reset_index(drop = True)            
maxAUUC = resultLogF['AUC'].max()
maxKendall = resultLogF['Kendall'].max()
resultLogF['AUC_Kendal'] = (resultLogF['AUC']/maxAUUC) + (resultLogF['Kendall']/maxKendall) 
resultLogF.sort_values(by=['AUC_Kendal'], ascending = False, inplace=True, axis=0)
resultFile = os.path.join(basePah, 'AUUC.csv')  
resultLogF.to_csv(resultFile, index=False)