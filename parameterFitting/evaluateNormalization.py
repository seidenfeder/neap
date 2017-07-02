#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################################################################################
#
# Script to evaluate the classification and regression performance of the different normalization methods 
# in normlaization.py using 10-fold cross-validation
#
##################################################################################################################

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Normalize inputfile
os.system("python preprocessing/normalizations.py -i ~/Desktop/input_mRNA.txt -o ~/Desktop/input_mRNA_scaled.txt -m Scale -n")  
os.system("python preprocessing/normalizations.py -i ~/Desktop/input_mRNA.txt -o ~/Desktop/input_mRNA_normalized.txt -m Norm -n")  

createdInputFiles=['~/Desktop/input_mRNA.txt','~/Desktop/input_mRNA_scaled.txt', '~/Desktop/input_mRNA_normalized.txt']

#####################################################################################
# For classification
classificationMethods=["RF","SVM"]
for cM in classificationMethods:
    print(cM)
    for file in createdInputFiles:
        print(file)
        os.system("python methods/classification.py -i "+ file +" -l ~/Desktop/labels_mRNA.txt "+ 
                  "-c 10 -a -o ~/Desktop/classification_normalization.txt -n -m "+cM)

#Plotting the classification results
fileName="/home/sch/schmidka/Desktop/classification_normalization.txt"

resultFile=open(fileName)
score=[]
for line in resultFile:
    line=line.rstrip()
    if not line.startswith('#'):
        results=line.split('\t')
        score.append(list(map(float,results[2:])))

scaling=['RF','RF (Scaled)','RF (Normalized)','SVM','SVM (Scaled)','SVM (Normalized)']

plt.boxplot(score)
plt.xticks(range(1,len(scaling)+1),scaling, rotation='vertical')
plt.xlabel("Method (Normalization)")
plt.ylabel("AUC score")
plt.title('AUC Scores of RF and SVM with different normalizations')
plt.tight_layout()
plt.savefig('/home/sch/schmidka/Desktop/Vergleich_Klassifikation.png')

#####################################################################################
# For regression
regressionMethods=["LR","RF","SVM"]
for rM in regressionMethods:
    print(rM)
    for file in createdInputFiles:
        print(file)
        os.system("python methods/regression.py -i "+ file +
                  " -c 10 -a -o ~/Desktop/regression_normalization.txt -n -m "+rM)
        
#Plotting the regression results
fileName="/home/sch/schmidka/Desktop/regression_normalization.txt"

resultFile=open(fileName)
score=[]
for line in resultFile:
    line=line.rstrip()
    if not line.startswith('#'):
        results=line.split('\t')
        score.append(list(map(float,results[2:])))

scaling=['LR','LR (Scaled)','LR (Normalized)','RF','RF (Scaled)','RF (Normalized)','SVM','SVM (Scaled)','SVM (Normalized)']

plt.boxplot(score)
plt.xticks(range(1,len(scaling)+1),scaling, rotation='vertical')
plt.xlabel("Method (Normalization)")
plt.ylabel("R2 score")
plt.title('R2 Scores of LR, RF and SVM with different normalizations')
plt.tight_layout()
plt.savefig('/home/sch/schmidka/Desktop/Vergleich_Regression.png')
