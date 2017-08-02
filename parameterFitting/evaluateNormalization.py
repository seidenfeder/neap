#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################################################################################
#
# Script to evaluate the classification and regression performance of the different normalization methods 
# in normalization.py using 10-fold cross-validation
#
##################################################################################################################

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-i",dest="input", help="This gives the path to the file with the input data (the output of the binning)")
parser.add_option("-n",dest="name", help="Give the name of the cell line for better naming", default="")
parser.add_option("-l",dest="labels", help="This gives the path to the file with the labels")
#save the given options
(options, args) = parser.parse_args()
features=options.input
name = options.name
labels=options.labels

#Normalize inputfile
#os.system("python preprocessing/normalizations.py -i "+features+ " -o "+name+"_scaled.txt -m Scale -n")  
#os.system("python preprocessing/normalizations.py -i "+features+ " -o "+name+"_normalized.txt -m Norm -n")  

createdInputFiles=[options.input, name+'_scaled.txt', name+'_normalized.txt']
normalizationMethods=["None","Scaled","Normalized"]
#####################################################################################
# For classification
#classificationMethods=["RF","SVM"]
#for cM in classificationMethods:
#    print(cM)
#    for i in range(0,len(normalizationMethods)):
#        file=createdInputFiles[i]
#        print(file)
#        os.system("python methods/classification.py -i "+ file +" -l "+labels+" "+ 
#                  "-c 10 -a -o classification_"+name+"_normalization.txt -n -m "+cM + " --add " + normalizationMethods[i])
#
#Plotting the classification results
fileName="classification_"+name+"_normalization.txt"

resultFile=open(fileName)
score=[]
scaling=[]
for line in resultFile:
    line=line.rstrip()
    if not line.startswith('#'):
        results=line.split('\t')
        score.append(list(map(float,results[4:])))
        scaling.append(results[1]+" ("+results[2]+")")


plt.boxplot(score)
plt.xticks(range(1,len(scaling)+1),scaling, rotation='vertical')
plt.xlabel("Method (Normalization)")
plt.ylabel("AUC score")
plt.title('AUC Scores of RF and SVM with different normalizations')
plt.tight_layout()
plt.savefig('Vergleich_Klassifikation.png')

#####################################################################################
# For regression
regressionMethods=["LR","RF","SVM"]
for rM in regressionMethods:
    print(rM)
    for file in createdInputFiles:
        print(file)
        os.system("python methods/regression.py -i "+ file +
                  " -c 10 -a -o regression_"+name+"_normalization.txt -n -m "+rM)
        
#Plotting the regression results
fileName="regression_"+name+"_normalization.txt"

resultFile=open(fileName)
score=[]
scaling=[]
for line in resultFile:
    line=line.rstrip()
    if not line.startswith('#'):
        results=line.split('\t')
        score.append(list(map(float,results[4:])))
        scaling.append(results[1]+" ("+results[2]+")")

#scaling=['LR','LR (Scaled)','LR (Normalized)','RF','RF (Scaled)','RF (Normalized)','SVM','SVM (Scaled)','SVM (Normalized)']
plt.figure(0)
plt.boxplot(score)
plt.xticks(range(1,len(scaling)+1),scaling, rotation='vertical')
plt.xlabel("Method (Normalization)")
plt.ylabel("R2 score")
plt.title('R2 Scores of LR, RF and SVM with different normalizations')
plt.tight_layout()
plt.savefig('Vergleich_Regression.png')
