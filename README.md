# Maximum Causal Tree
A python implementation of Maximium Causal Tree (MCT) in paper "Recommending the Most Effective Intervention to Improve Employment for Australians with Disability". This implementation also uses R packages to build a causal DAG and to execute baseline methods.

# Installation
Installation requirements for Maximum Causal Tree

* Python >= 3.6
* numpy
* pandas
* scipy
* seaborn
* sklearn
* R >= 3.6.2
* pcalg
* Rgraphviz

Installtion requirements and how to install baseline methods can be found at https://github.com/susanathey/causalTree

# Usage

**1. Reproduce results in the paper with existing data**

    Run python script "GeneratePaperResultsForSyn.py". Results are stored in folder "output/PerformanceEval/synthetic"

**2. Reproduce results in the paper from scratch**

Step 1 - Generate synthetic data.
    
    Run python script GenerateSyntheticData.py to generate 4 synthetic datasets (four test scenarios) and 200 derived datasets.
    
    Generated datasets are stored in folder "input/synthetic".
    
Step 2 - Run four baselines methods with 200 derived datasets.

    Run R script ProcessSynData.R in folder "RCode".
    
    Results are stored in folder "output/CausalTree", "output/FitBasedTree", "output/TOTree", and "output/TStatisticTree"

Step 3 - Convert these results to a summary format.

    Run python script ProcessBaselineResults.py to convert the data.
    
    Summary files for each method are stored in the folder of each test scenario. 

Step 4 - Run the MCT method with four synthetic datasets.

    Run shell script ProcessSynData to process data with MCT. This script calls python scripts ProbSyn*.py and PostProcSyn*.py to process data.  Note that python scripts ProbSyn*.py are written to run in parallel mode on a Unix-like system.
    
    Results are stored in folder "output/MCT".
    
Step 5 - Produce results in the paper.

    Run python script "GeneratePaperResultsForSyn.py". Results are stored in folder "output/PerformanceEval/synthetic".

**3. Setps to run MCT with other data**

Step 1 - Detect causal factors from data

    Use R package pcalg to identify causes of an outcome. Sample file is DiscoverCauses.R in folder "RCode".

Step 2 - Build maximum causal trees for identified factors.

    Build MCT trees using class DyCausalTree for each factor. Call method constructTree() to build a tree.

Step 3 - Recommend the most effective factor

    Construct an object from class InterventionRecommender with a list of maximum causal trees. Use method makeRecommendation to generate recommendations.

