
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import uuid 
import mord as m
from anytree import *
from anytree.exporter import DotExporter

from BaseTree import *
from GlobalVariables import *
from Utility import *
    
class CausalNode(BsNode):
    
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.splitVal = 0
        self.splitVar = ''
        self.condition = ''
        self.minTreatment = 0
        self.nbrInstance = 0
        self.variance = 0
        self.id = uuid.uuid1()
        self.tree = None

    # Search for the best treatment level: go through all levels and compute treatment effects. 
    # The best level is the one with the largest treatment effect (Algorithm 3 in the paper)
    def searchForBestMinTreatment(self, t, covX, y, datatset, treatmentName, minSize, alpha, train_to_est_ratio):     
        global globalDataSet
        noneVal = (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)
        minVal = np.min(t)
        maxVal = np.max(t)
        minIndex = -9999
        maxIndex = -9999
        treatmentIntervals = self.tree.treatmentIntervals
        valCount = 0
        for treatinterval in treatmentIntervals:
            
            if (treatinterval > minVal) and ( minIndex == -9999):
                minIndex = valCount
            if (treatinterval > maxVal) and ( minIndex == -9999):
                maxIndex = valCount -1   
            valCount = valCount + 1    
        if(maxIndex == -9999):
            maxIndex = valCount   
        uniqueTrVals = treatmentIntervals[minIndex: (maxIndex + 1)]
        propenModels = self.tree.propenModels[minIndex: (maxIndex + 1)]
        dupOutcome= np.tile(y, (uniqueTrVals.shape[0], 1))
        dupTreatment = np.tile(t, (uniqueTrVals.shape[0], 1))
        propensities = np.empty([uniqueTrVals.shape[0], t.shape[0]])
        
        valCount = 0        
        for irangeIndex in range (0, len(uniqueTrVals)): 
            propen = propenModels[irangeIndex].predict_proba(covX)
            propensities[valCount, :] = propen[:, 1]            
            valCount = valCount + 1
                
        index = np.transpose(np.transpose(dupTreatment) >= uniqueTrVals)
        propensities[np.logical_not(index)] = 1 - propensities[np.logical_not(index)]
        w = propensities
        dupTreatment[index] = 1
        dupTreatment[np.logical_not(index)] = 0
        sum_treat = np.sum(dupTreatment == 1, axis=1)
        sum_control = np.sum(dupTreatment == 0, axis=1)
        min_size_idx = np.where((sum_treat >= minSize) & (sum_control >= minSize))
        uniqueTrVals = uniqueTrVals[min_size_idx]
        dupTreatment = dupTreatment[min_size_idx]
        dupOutcome = dupOutcome[min_size_idx]
        w = w[min_size_idx]
        if dupTreatment.shape[0] == 0:
            return noneVal
            
        num_treatment = np.sum((1/w)*(dupTreatment == 1), axis=1)
        num_control = np.sum((1/(1-w))*(dupTreatment == 0), axis=1)
        treatmentMean = np.sum(((1/w)*dupOutcome * (dupTreatment == 1)), axis=1) / num_treatment
        controlMean = np.sum(((1/(1-w))*dupOutcome * (dupTreatment == 0)), axis=1) / num_control
        effect = treatmentMean - controlMean
        err = effect ** 2
        eOfy2_treatment  = np.sum((w*(dupOutcome **2) * (dupTreatment == 1)), axis=1) / num_treatment 
        ey2_treatment = (np.sum((w*dupOutcome * (dupTreatment == 1)), axis=1) / num_treatment)**2
        var_treatment = eOfy2_treatment - ey2_treatment
        
        eOfy2_control  = np.sum((w*(dupOutcome **2) * (dupTreatment == 0)), axis=1) / num_control 
        ey2_control = (np.sum((w*dupOutcome * (dupTreatment == 0)), axis=1) / num_control)**2
        var_control = eOfy2_control - ey2_control        
        mse = effect ** 2 - alpha* (1 + train_to_est_ratio)*(var_treatment/num_treatment + var_control/num_control)
        max_err = np.argmax(effect)
        bestEffect = effect[max_err]
        bestMse = err[max_err]
        bestSplit = uniqueTrVals[max_err]
        bestVMse = mse[max_err]
        bestControlMean = controlMean[max_err]
        bestTreatmentMean = treatmentMean[max_err]
        return bestEffect, bestMse, bestSplit, bestVMse, bestControlMean, bestTreatmentMean
    
    #Compute the min treatment level (the best level) and corresponding treatment effect for a node (a subgroup)
    def estimateValues(self, t, covX, y, datatset, treatmentName, minSize, alpha, train_to_est_ratio): 
        
        bestEffect, bestMse, bestSplit, bestVMse, bestControlMean, bestTreatmentMean = self.searchForBestMinTreatment(t, covX, y, datatset, treatmentName, minSize, alpha, train_to_est_ratio)
        self.effect = bestEffect
        self.negRisk = bestVMse * t.shape[0] # include variance
        self.minTreatment  = bestSplit
        self.nbrInstance = t.shape[0]
        self.variance  = (bestMse - bestVMse)*t.shape[0]       
        self.controlMean = bestControlMean
        self.treatmentMean = bestTreatmentMean
        
    #Recursively split the node. This is Algorithm 2 in the paper
    def doSplit(self, covariateNames, treatmentName, outcomeName,  maxTreeDepth, minSize, alpha, train_to_est_ratio, criteria=None):
        global globalDataSet
        if(self.depth >= maxTreeDepth):
            self.isLeaf = True
            self.tree.leafNbr = self.tree.leafNbr  + 1
            self.id =  str(self.tree.leafNbr)         
            return
        if(criteria != None):
            dataset = globalDataSet.query(criteria)
        else:
            dataset = globalDataSet
        bestRightCriteria = ''
        bestLeftCriteria = ''
        archiveConditionList  = np.empty((0, len(self.tree.epsilon)))
        archiveBoxList  = np.empty((0, len(self.tree.epsilon)))                
        archiveSplitList  = []        
        splitPointList  = []        
        tempConditionList  = np.empty((0, len(self.tree.epsilon)))
                
        for cov in covariateNames:           
            orinUnique_vals = sorted(dataset[cov].unique())
            orinUnique_vals = np.array(orinUnique_vals)              
            ## process unique values, splitttig point is a middle between two points
            unique_vals = (orinUnique_vals[1:] + orinUnique_vals[:-1]) / 2           
            for val in unique_vals:
                if(unique_vals.shape[0] == 1):# binary
                    rightOper = cov + '==' + str(orinUnique_vals[1])
                    leftOper = cov + '==' + str(orinUnique_vals[0])
                    rightOperStr = cov + '==' + str(orinUnique_vals[1])
                    leftOperStr = cov + '==' + str(orinUnique_vals[0])
                else:
                    rightOper = cov + '>=' + str(val)
                    leftOper = cov + '<' + str(val)
                    rightOperStr = cov + '>=' + "%.2f" % val
                    leftOperStr = cov + '<' + "%.2f" % val                    
                if(criteria == None):
                    newRightCriteria = rightOper
                    newLeftCriteria = leftOper  
                else:
                    newRightCriteria = criteria + ' & ' + rightOper 
                    newLeftCriteria = criteria + ' & ' + leftOper   
                rightData = globalDataSet.query(newRightCriteria)
                leftData = globalDataSet.query(newLeftCriteria)
                r_t = rightData[treatmentName].to_numpy()
                r_y = rightData[outcomeName].to_numpy()
                l_t = leftData[treatmentName].to_numpy()
                l_y = leftData[outcomeName].to_numpy()
                                    
                # estimate risk for the right node
                rightNode = CausalNode(self.depth + 1)
                rightNode.tree = self.tree
                
                covXr = rightData[self.tree.confdCovariateNames].to_numpy()
                rightNode.estimateValues(r_t, covXr, r_y, rightData, treatmentName, minSize, alpha, train_to_est_ratio)
                rightNode.condition = rightOperStr
                                           
                # estimate risk for the left node
                leftNode = CausalNode(self.depth + 1)
                leftNode.tree = self.tree
                covXl = leftData[self.tree.confdCovariateNames].to_numpy()
                leftNode.estimateValues(l_t, covXl, l_y, leftData, treatmentName, minSize, alpha, train_to_est_ratio) 
                leftNode.condition = leftOperStr                
                condition1 = rightNode.negRisk + leftNode.negRisk - self.negRisk
                if(np.isinf(condition1)):
                    continue 
                condition2 = (leftNode.effect - rightNode.effect)**2
                condition3 = (leftNode.minTreatment - rightNode.minTreatment)**2
                splitPoint = [leftNode, rightNode, cov, val, newLeftCriteria, newRightCriteria]
                splitPointList.append(splitPoint)               
                tempConditionList = np.vstack([tempConditionList, [condition1, condition2, condition3]])
            # End for val
        # End for covariates 
        ## Shift condition to ensure not negative       
        if(len(tempConditionList) > 0):
            minVal = np.min(tempConditionList, axis = 0)
            tempConditionList = tempConditionList + 2*np.absolute(minVal) 
        icountc = 0
        for conditionIns in tempConditionList:    
            splitPoint = splitPointList[icountc]
            # Search for an optimal split set
            archiveConditionList, archiveBoxList, archiveSplitList = self.__searchForEpsiNonDominance(conditionIns, splitPoint, 
                                                                      archiveConditionList, archiveBoxList, archiveSplitList, self.tree.epsilon, False)
            icountc = icountc + 1
   
        bestImproveRisk = -99999
        bestSplit = None
        for icount in range(0, len(archiveConditionList)):
            if(archiveConditionList[icount][0] > bestImproveRisk):
                bestImproveRisk = archiveConditionList[icount][0] 
                bestSplit = archiveSplitList[icount]
        if(bestImproveRisk >= self.tree.effectImp):            
            self.leftChild = bestSplit[0]
            self.rightChild = bestSplit[1]
            self.splitVar = bestSplit[2]
            self.splitVal = bestSplit[3]
            bestLeftCriteria = bestSplit[4]
            bestRightCriteria = bestSplit[5]
            self.leftChild.doSplit(covariateNames, treatmentName, outcomeName, maxTreeDepth, minSize, alpha, train_to_est_ratio, bestLeftCriteria)
            self.rightChild.doSplit(covariateNames, treatmentName, outcomeName, maxTreeDepth, minSize, alpha, train_to_est_ratio, bestRightCriteria)           
            
        else:
            self.isLeaf = True
            self.tree.leafNbr = self.tree.leafNbr  + 1
            self.id =  str(self.tree.leafNbr)

    def __epsiDominateOver(self, a, b, epsilon, nbrObj):
        npEpsilon = np.array(epsilon)
        return (sum((1+npEpsilon*np.sign(a))*a >= b) == nbrObj)
            
    def boxEstimate(self, candidates, epsilon):
        if(len(candidates) == 0):
            return None
        npEpsilon = np.array(epsilon) 
        return np.floor(np.log(candidates)/np.log(1 + npEpsilon))
    
    # Search for an optimal split set (Algorithm 1 in the paper)
    def __searchForEpsiNonDominance(self, candidate, splitPoint, archiveList, archiveBoxList, archiveSplitList, epsilon, debugs):
        nbrObj = len(epsilon)
        cBox = self.boxEstimate(candidate, epsilon)
        ai = -1 # ai: archive index
        asize = len(archiveList)
        sdominate = False # solution dominates
        noneDoniminate = False
        while ai < asize - 1:
            ai += 1
            aBox = archiveBoxList[ai]
            ## The idea is that new points are only accepted if they are not 
            ##epsilon dominated by any other point of the current archive. 
            ## If a point is accepted, all dominated points are removed.
            ## The same box
            if (np.array_equal(aBox, cBox)):
                ## solution dominate archives ==> remove archives
                if(self.__epsiDominateOver(candidate, archiveList[ai], epsilon, nbrObj)):
                    archiveList = np.delete(archiveList, ai, 0)
                    archiveBoxList = np.delete(archiveBoxList, ai, 0)
                    archiveSplitList.pop(ai)
                    ai -= 1
                    asize -= 1
                    sdominate =  True
                    break
                else:
                    sdominate =  False
                    noneDoniminate = False
                    break
            elif (self.__epsiDominateOver(aBox, cBox, epsilon, nbrObj)):## archive dominates
                sdominate = False
                noneDoniminate = False
                break
            # from this solutions is not dominated
            ## solution dominates
            elif(self.__epsiDominateOver(cBox, aBox, epsilon, nbrObj)):## solution dominates
                #remove dominated points
                archiveList = np.delete(archiveList, ai, 0)
                archiveBoxList = np.delete(archiveBoxList, ai, 0)                
                archiveSplitList.pop(ai)
                ai -= 1
                asize -= 1
                sdominate =  True
            else: ## Non dominate
                noneDoniminate = True   
        if(sdominate or noneDoniminate or asize == 0):
            archiveList = np.vstack([archiveList, candidate])
            archiveBoxList = np.vstack([archiveBoxList, cBox])
            archiveSplitList.append(splitPoint)
            
        return archiveList, archiveBoxList, archiveSplitList 
        
class MaximumCausalTree(BsTree):
    
    def __init__(self, dataset, treatmentName, outcomeName, covariateNames, maxTreeDepth, minSize, propenModel):
        global globalDataSet
        super().__init__()
        self.train_to_est_ratio = 0
        self.root = CausalNode(0)
        self.root.tree = self
        self.dataset = dataset
        self.maxTreeDepth = maxTreeDepth
        self.minSize = minSize
        self.varianceAlpha = 0
        self.outcomeName = outcomeName
        self.treatmentName = treatmentName
        self.covariateNames = covariateNames
        self.id = uuid.uuid1()
        self.propenModel = propenModel             
        self.treatmentIntervals = None
        self.propenModels =  None
        self.confdCovariateNames = None
        self.newMethodOfPropen =  True
        self.epsilon =  []
        self.effectImp =  0
        self.leafNbr =  0
        globalDataSet = dataset
        
    def __predict(self, currentNode, individual):
        if currentNode.isLeaf:
            
            return currentNode.effect, currentNode.minTreatment, currentNode.id, currentNode.controlMean, currentNode.treatmentMean
        else:
            xVal = individual[currentNode.splitVar]
            if xVal >= currentNode.splitVal:
                whereTogo = currentNode.rightChild
            else:
                whereTogo = currentNode.leftChild

        return self.__predict(whereTogo, individual)
        
    def predictIndividual(self, jobSeeker):
        return self.__predict(self.root, jobSeeker) 
        
    
    def constructTree(self):

        global globalDataSet
        t = self.dataset[self.treatmentName].to_numpy()
        y = self.dataset[self.outcomeName].to_numpy()        
        covX = self.dataset[self.confdCovariateNames].to_numpy()
        
        # Estimate min treatment level and treatment effect for the root node
        self.root.estimateValues(t, covX, y, self.dataset, self.treatmentName, self.minSize, self.varianceAlpha, self.train_to_est_ratio )
        
        # Recursively split the root node to build the tree
        self.root.doSplit(self.covariateNames, self.treatmentName, self.outcomeName, self.maxTreeDepth, self.minSize, self.varianceAlpha, self.train_to_est_ratio,None)                
                
    def __buildVisTree(self, causalNode, visPa=None):
        curNode= Node(causalNode.condition, parent = visPa, condition = causalNode.condition, \
                      effect= '{0:.2f}'.format(causalNode.effect), negRisk= '{0:.2f}'.format(causalNode.negRisk), \
                      minTreatment= '{0:.5f}'.format(causalNode.minTreatment),
                      variance= '{0:.2f}'.format(causalNode.variance),
                      pVal= '{0:.3f}'.format(causalNode.pVal),
                      idnode = causalNode.id,
                      controlMean= '{0:.2f}'.format(causalNode.controlMean),
                      treatmentMean= '{0:.2f}'.format(causalNode.treatmentMean)
                      )
        if(causalNode.leftChild != None):
            self.__buildVisTree(causalNode.leftChild, curNode)
    
        if(causalNode.rightChild != None):
            self.__buildVisTree(causalNode.rightChild, curNode)
        return curNode
    def __buildVisTreeV2(self, causalNode, visPa, breakPoints, breakPointStyle):
        
        tempCondition = self.convertCondition(causalNode.condition, breakPoints, breakPointStyle)
        originalVal = causalNode.minTreatment
        if(self.treatmentName in breakPoints):
            None
            originalVal = convertToOrignalBreakPoint(originalVal, breakPoints[self.treatmentName], breakPointStyle[self.treatmentName])
            
        
        stringOriginalVal = ''
        
        try:
            originalVal = float(originalVal)
            stringOriginalVal = '{0:.2f}'.format(originalVal)
            
        except ValueError:
            stringOriginalVal = str(originalVal)
            
    
        hasParent = True
        if(visPa == None):
            hasParent = False
            
        if(tempCondition != None):
            tempCondition = tempCondition.replace('==',' = ')
            tempCondition = tempCondition.replace('>=',' >= ')
            tempCondition = tempCondition.replace('<',' < ')
            
        causalNode.effect = causalNode.effect if round(causalNode.effect, 2) != 0 else abs(causalNode.effect)
        curNode= Node(tempCondition, parent = visPa, condition = tempCondition, \
                      effect= '{0:.2f}'.format(causalNode.effect ), negRisk= '{0:.2f}'.format(causalNode.negRisk), \
                      minTreatment= stringOriginalVal,
                      variance= '{0:.2f}'.format(causalNode.variance),
                      pVal= '{0:.3f}'.format(causalNode.pVal),
                      idnode = causalNode.id,
                      controlMean= '{0:.2f}'.format(causalNode.controlMean),
                      treatmentMean= '{0:.2f}'.format(causalNode.treatmentMean),
                      isLeaf = causalNode.isLeaf,
                      hasParent = hasParent,
                      treatmentName = self.treatmentName
                      )
        
        if(causalNode.leftChild != None):
            self.__buildVisTreeV2(causalNode.leftChild, curNode, breakPoints, breakPointStyle)
    
        if(causalNode.rightChild != None):
            self.__buildVisTreeV2(causalNode.rightChild, curNode, breakPoints, breakPointStyle)
            
        
        return curNode  
            
    def __getAllLeafNodes(self, currentNode):
        if currentNode.isLeaf:
            return [currentNode]
        leftLeaves = self.__getAllLeafNodes(currentNode.leftChild)
        rightLeaves = self.__getAllLeafNodes(currentNode.rightChild)
        return leftLeaves + rightLeaves
        
    def convertCondition(self, cond, breakPoints, breakPointStyle):
        if(len(cond) == 0):
            return cond
        print('cond ' + cond)
        tempRes = cond.split( '<') 
        if(len(tempRes) == 2):
            tempVal = float(tempRes[1])   
            if(tempRes[0] in breakPoints):
                return tempRes[0] + '<' +str(convertToOrignalBreakPoint(tempVal, breakPoints[tempRes[0]], breakPointStyle[tempRes[0]] ))
            else:
                return cond              
        tempRes = cond.split( '>=') 
        if(len(tempRes) == 2):
            tempVal = float(tempRes[1])
            if(tempRes[0] in breakPoints):   
                return tempRes[0] + '>=' +str(convertToOrignalBreakPoint(tempVal, breakPoints[tempRes[0]], breakPointStyle[tempRes[0]] ))
        tempRes = cond.split( '==')
        if(len(tempRes) == 2):    
            tempVal = float(tempRes[1])
            if(tempRes[0] in breakPoints):
                return tempRes[0] + ' = ' +str(convertToOrignalBreakPoint(tempVal, breakPoints[tempRes[0]], breakPointStyle[tempRes[0]] ))
              
        tempRes = cond.split( '==')
            
        if(len(tempRes) == 2):
            if(int(tempRes[1]) ==  1):
                return tempRes[0] + ' = Yes'
            else:
                return tempRes[0] + ' = No'                
        return cond
        
    def plotTreeV2(self, namPic, breakPoints, breakPointStyle):
        
        visRoot = self.__buildVisTreeV2(self.root, None, breakPoints, breakPointStyle)
         
        DotExporter(visRoot, options=["label_scheme=3;splines=polyline; ranksep=equally;compound=true;"]
                    , nodeattrfunc=lambda node: 
                        'fontsize=22, fontname =helvetica, shape=box, style="rounded,filled", fillcolor=green'
                        if ( not node.hasParent)
                        else("fontsize=22, fontname =helvetica, shape=box, style=rounded")
                        if (node.isLeaf)
                        else("fontsize=0, shape=diamond,  splines=line, style=rounded, fontcolor=white")
                    , edgeattrfunc = lambda node, child:  'splines=line, fontsize=22, fontname =helvetica, label=" %s "' % (child.condition),\
                    nodenamefunc = lambda node:  
                    (    'Effect: %s \n Intervention level : %s \n' 
                    % (node.effect, node.minTreatment) 
                    )
                    if (node.isLeaf) 
                    else ( ' Intervention = %s' 
                    % (node.treatmentName) 
                    ) 
                    if(not node.hasParent) else (    'Int. Effect: %s' 
                    % (node.effect) 
                    )
                    ).to_picture(namPic + ".png")
        
    def plotTree(self, namPic='NoneName'):
        visRoot = self.__buildVisTree(self.root)
        DotExporter(visRoot, edgeattrfunc = lambda node, child:  'label="%s"' % (child.condition),\
                    nodenamefunc = lambda node:  'CATE: %s, treatment: %s \n Risk: %s, Variance: %s,  p: %s \n control mean:%s, treatment mean:%s' 
                    % (node.effect, node.minTreatment, node.negRisk, node.variance, node.pVal, node.controlMean, node.treatmentMean) ).to_picture(namPic + ".png")   