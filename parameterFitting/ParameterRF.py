#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################
#
#This script test which is the best number of trees for the Random Forest
#
####################################################################################################

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from optparse import OptionParser

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-i",dest="input", help="This gives the path to the file with the input data (the output of the binning)")
parser.add_option("-l",dest="labels", help="This gives the path to the file with the labels")
parser.add_option("-b",type = "integer",dest="bin", help="tells which bin should be used for the classification")
parser.add_option("-c",type = "int",dest="crossVal", help="Number of iterations in the cross validation", default=5)
(options, args) = parser.parse_args()

labelFilename=options.labels
featureFilename=options.input

#Read labels
labelFile = open(labelFilename)
labelDict = dict()
for line in labelFile.readlines():
    lineSplit=line.split()
    labelDict[lineSplit[0]]=int(lineSplit[2])
    

#Read features
featureFile=open(featureFilename)
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
#Maybe for some genes no GENCODE entry could be found, these are only in the features list
y=[]
X=[]
binNumber=options.bin
#Create feature matrix of the given bin 
for geneID in genesModis:
    y.append(labelDict[geneID])
    valueMatrix=np.array(genesModis[geneID])
    X.append(valueMatrix[:,binNumber])

score=[]
for estimator in range(2,20):
	#Random Forest
	clf=RandomForestClassifier(n_estimators=estimator)
	scores = cross_val_score(clf, X, y, cv=options.crossVal, scoring='roc_auc')
	score.append(scores)


import matplotlib.pyplot as plt
plt.boxplot(score, labels=range(2,20))
plt.xlabel("#Trees")
plt.ylabel("AUC score")
plt.title('AUC Score of Random Forest with different #Trees')
plt.savefig('VergleichParameterRF.png')
plt.show()

