#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################
#
# This script is able to run the two classification methods Support Vector Machine and Random Forest
# Output gives the AUROC scores of the cross validation with the chosen method 
#
####################################################################################################

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from optparse import OptionParser

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-m", type="string", dest="method", help = "the method you want to use Support Vector Machine (SVM) or Random Forest (RF) default= RF", default="RF")
parser.add_option("-i",dest="input", help="This gives the path to the file with the input data (the output of the binning)")
parser.add_option("-l",dest="labels", help="This gives the path to the file with the labels")
parser.add_option("-b",type = "int",dest="bin", help="Tells which bin should be used for the classification")
parser.add_option("-c",type = "int",dest="crossVal", help="Number of iterations in the cross validation", default=5)
parser.add_option("-a", dest="allBins", help = "Tells if all bins should be used", default=False)
parser.add_option("-o",dest="output", help="The name of the outputfile", default="classification.txt")
(options, args) = parser.parse_args()
method=options.method

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
#If not all bins should be used
if(not options.allBins):
	binNumber=options.bin
	#Create feature matrix of the given bin 
	for geneID in genesModis:
	    y.append(labelDict[geneID])
	    valueMatrix=np.array(genesModis[geneID])
	    X.append(valueMatrix[:,binNumber])
#if you want the classification with all bins
else:
	for geneID in genesModis:
	    y.append(labelDict[geneID])
	    valueMatrix=np.array(genesModis[geneID])
	    X.append(valueMatrix.flatten())

#Support Vector Machines
if(method=="SVM"):
    clf=svm.SVC(cache_size=500)
#Random Forest
elif(method=="RF"):
    clf=RandomForestClassifier(n_estimators=16)

scores = cross_val_score(clf, X, y, cv=options.crossVal, scoring='roc_auc')

#write the output into a file but don't delete the previous text
#this is necessary that we can compare different data sets or binnings or methods
fileHandle = open ( options.output, 'a' )
fileHandle.write(method+"\t"+'\t'.join(map(str,scores))+"\n")
fileHandle.close()

