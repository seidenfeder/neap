#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################
#
# This script test which is the best kernel method for SVM
#
####################################################################################################

import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i",dest="input", help="This gives the path to the file with the input data (the output of the binning)")
parser.add_option("-l",dest="labels", help="This gives the path to the file with the labels")
parser.add_option("-b",type = "int",dest="bin", help="tells which bin should be used for the classification")
parser.add_option("-c",type = "int",dest="crossVal", help="Number of iterations in the cross validation", default=5)
parser.add_option("-o", dest="outputfile", help="Output file to save the results", default="SVMkernelMethods.txt")
(options, args) = parser.parse_args()

labelFilename=options.labels
featureFilename=options.input
outputfile=options.outputfile

#Read labels
labelFile = open(labelFilename)
labelDict = dict()
for line in labelFile.readlines():
    if not line.startswith("##"):
        lineSplit=line.split()
        labelDict[lineSplit[0]]=int(lineSplit[2])
    

#Read features
featureFile=open(featureFilename)
#Name of the data set (from the header)
dataset=featureFile.readline().rstrip()[2:]
#All modifications
modifications=featureFile.readline().rstrip()[2:].split(" ")

genesModis=dict()
for line in featureFile.readlines():
    line=line.rstrip()
    if(line.startswith('#')):
        lineSplit=line.split(" ")
        geneID=lineSplit[0]
        #Remove the hashtag at the beginning of the line
        geneID=geneID[1:]
        genesModis[geneID]=[]
    else:
        valueList=line.split(",")
        valueList=list(map(float,valueList))
        genesModis[geneID].append(valueList)

        
#Sort labels according to the feature list
y=[]
X=[]
binNumber=options.bin
#Create feature matrix of the given bin 
for geneID in genesModis:
    y.append(labelDict[geneID])
    valueMatrix=np.array(genesModis[geneID])
    X.append(valueMatrix[:,binNumber])

fileHandle = open(outputfile, 'a' )

#Validate different kernel functions of SVM
scores=[]
kernels=['poly','linear', 'rbf', 'sigmoid'] 
for kernelTyp in kernels:
    print(kernelTyp)
    clf=svm.SVC(cache_size=1000, kernel=kernelTyp, degree=2)
    result=cross_val_score(clf, X, y, cv=options.crossVal, scoring='roc_auc')
    scores.append(result)
    print(result)
    fileHandle.write(kernelTyp+"\t"+'\t'.join(map(str,result))+"\n")

fileHandle.close()

plt.boxplot(scores)
plt.xticks(range(1,len(kernels)+1),kernels)
plt.xlabel("Kernel method")
plt.ylabel("AUC score")
plt.title('AUC Score of SVM with different kernel methods')
plt.savefig('VergleichKernelsSVM.png')

