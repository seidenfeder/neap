#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################
#
# Adaption of the method classification_storeModel to run on our big merged dataset
# The merged file need a line ##<datasetname> before each cell type in the labelfile
# and also the typical binning header before each cell type in the feature file
#
#
# This script is able to run the two classification methods Support Vector Machine and Random Forest 
# on a training set and save the gotten model
#
####################################################################################################

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from optparse import OptionParser
from sklearn.externals import joblib

parser = OptionParser()
parser.add_option("-m", type="string", dest="method", help = "the method you want to use Support Vector Machine (SVM) or Random Forest (RF) default= RF", default="RF")
parser.add_option("-i",dest="train", help="This gives the path to the file with the train dataset (binning file)")
parser.add_option("-l",dest="labels", help="This gives the path to the file with the labels fot´r the training data")
parser.add_option("-b",type = "int",dest="bin", help="Tells which bin should be used for the classification")
parser.add_option("-a", action="store_true", dest="allBins", help = "Tells if all bins should be used", default=False)
parser.add_option("-o",dest="output", help="The name of the outputfile to store the model in", default="model.pkl")
parser.add_option("-n", action="store_true", dest="newFormat", help="Feature file created by bins annotated, containing ENCODE metadata infos", default=False)

(options, args) = parser.parse_args()
method=options.method
labelFilename=options.labels
featureFilename=options.train
modelFilename=options.output

#Read labels
labelFile = open(labelFilename)
labelDict = dict()
for line in labelFile.readlines():
    line=line.rstrip()
    if(line.startswith("##")):
        dataset=line[2:]
    else:
        lineSplit=line.split()
        labelDict[dataset+lineSplit[0]] = int(lineSplit[2])
        

#Read features
featureFile=open(featureFilename)
genesModis=dict()
for line in featureFile.readlines():
    line=line.rstrip()
    if(line.startswith('##')):
        #line with the histone modification
        if(not line.startswith('##H')):
            dataset=line[2:]
    elif(line.startswith('#')):
        lineSplit=line.split(" ")
        geneID=lineSplit[0]
        #Remove the hashtag at the beginning of the line
        geneID=geneID[1:]
        genesModis[dataset+geneID]=[]
    else:
        valueList=line.split(",")
        valueList=list(map(float,valueList))
        genesModis[dataset+geneID].append(valueList)
    

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
    clf=svm.SVC(kernel='rbf', probability=True)
#Random Forest
elif(method=="RF"):
    clf=RandomForestClassifier(n_estimators=12)
#Train the model
clf.fit(X,y)  

#Save the model
joblib.dump(clf, modelFilename) 



