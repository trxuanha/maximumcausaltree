library(pcalg)
library(graph)
inBasePath <- paste (dirname(getwd()), '/input/synthetic/', sep='')

targetVairable<-'Y'

factors <- c('F') # manupulable factors

data_file <- paste (inBasePath, 'Case1.csv', sep='')
inputDat <-read.csv(file = data_file)
suffStat <- list(C = cor(inputDat), n = nrow(inputDat))
res <- pc(suffStat, labels = names(inputDat), indepTest = gaussCItest, alpha = 0.05) #

mygraph = attr(res, 'graph')
nodes(mygraph)
plot(mygraph)

potentialCauses <- inEdges(targetVairable, mygraph)

causes <- c()

for(cause in potentialCauses[[targetVairable]]){
    
    if( cause %in%  factors){
        causes <- c(causes, cause)
        
    }
    
}

print(causes)
outFilePath <- paste (dirname(getwd()), '/output/Cause/synthetic/causes.txt', sep='')
write(causes, file = outFilePath, sep = "")
write('==Outcome==Must keep this line to mark outcome variable', file = outFilePath, sep = "", append=TRUE)
write(targetVairable, file = outFilePath, sep = "", append=TRUE)