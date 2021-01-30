# Maximum Causal Tree
A python implementation of Maximium Causal Tree (MCT) in paper "Recommending the Most Effective Interventions to Improve Employment for Australians with Disability". This implementation also use some R packages to build a causal DAG and to convert results of baseline methods, which are implemented in R.

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


**2. Reproduce results in the paper from sratch**

It will take significant times to run the baselines and MCT.

**3. Generate synthetic data**

**3. Setps to run MCT with other data**

*Step 1*: Detect causal factors from data

*Step 2*: Generate Maximum causal trees for detected causal factors

These causal trees can then be used to recommend the most effective interventions.

