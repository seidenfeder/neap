#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#######################################################################################################################
#
# Adaption of the method classification_withStoredModel to run on our big merged dataset (with option mergedData)
# The merged file need a line ##<datasetname> before each cell type in the labelfile
# and also the typical binning header before each cell type in the feature file
#
# This script is able to run a classification task if the trained model is stored in a file.
#
#########################################################################################################################

import numpy as np
from optparse import OptionParser
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-i", dest="testset",help="This is the path to the test dataset")
parser.add_option("-l",dest="labelsTest", help="This gives the path to the file with the labels for the training data")
parser.add_option("-b",type = "int",dest="bin", help="Tells which bin should be used for the classification")
parser.add_option("-a", action="store_true", dest="allBins", help = "Tells if all bins should be used", default=False)
parser.add_option("-m",dest="modelFile", help="Model file where the trained classificator is saved in.")
parser.add_option("-o",dest="output", help="The name of the outputfile", default="regression.txt")
parser.add_option("-n", action="store_true", dest="newFormat", help="Feature file created by bins annotated, containing ENCODE metadata infos", default=False)
parser.add_option("-t",dest="trainset", help="The name of the trainset", default="")
parser.add_option("--mergedData",action="store_true",help="If the test file contains merged data")

(options, args) = parser.parse_args()

testFile= options.testset
labelTest=options.labelsTest
modelFile=options.modelFile
trainset=options.trainset

if(options.mergedData):
    #Read labels
    labelFile = open(labelTest)
    labelDict = dict()
    for line in labelFile.readlines():
        line=line.rstrip()
        if(line.startswith("##")):
            dataset=line[2:]
        else:
            lineSplit=line.split()
            labelDict[dataset+lineSplit[0]] = int(lineSplit[2])

    #Read features
    testfile=open(testFile)
    testGenes=dict()
    for line in testfile.readlines():
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
            testGenes[dataset+geneID]=[]
        else:
            valueList=line.split(",")
            valueList=list(map(float,valueList))
            testGenes[dataset+geneID].append(valueList)
     
    testset="BigData"
else:
    testfile=open(testFile)
    if options.newFormat :
        #Name of the data set (from the header)
        testset=testfile.readline().rstrip()[2:]
        #All modifications
        modificationsTest=testfile.readline().rstrip()[2:].split(" ")
    
    testGenes=dict()
    for line in testfile.readlines():
        line=line.rstrip()
        if(line.startswith('#')):
            lineSplit=line.split(" ")
            geneID=lineSplit[0]
            #Remove the hashtag at the beginning of the line
            geneID=geneID[1:]
            testGenes[geneID]=[]
        else:
            valueList=line.split(",")
            valueList=list(map(float,valueList))
            testGenes[geneID].append(valueList)
    
    #Read labels for the test set
    labelFile = open(labelTest)
    labelDict = dict()
    for line in labelFile.readlines():
        if not line.startswith("##"):
            lineSplit=line.split()
            labelDict[lineSplit[0]]=int(lineSplit[2])

w=[]
Z=[]
#If not all bins should be used
if(not options.allBins):
	binNumber=options.bin
	#Create feature matrix of the given bin 
	for geneID in testGenes:
	    w.append(labelDict[geneID])
	    valueMatrix=np.array(testGenes[geneID])
	    Z.append(valueMatrix[:,binNumber])
#if you want the classification with all bins
else:
	for geneID in testGenes:
	    w.append(labelDict[geneID])
	    valueMatrix=np.array(testGenes[geneID])
	    Z.append(valueMatrix.flatten())

####### Test the model #########
clf = joblib.load(modelFile) 
pred = clf.predict_proba(Z)
score = roc_auc_score(w,pred[:,1])

#get the used method
if str(clf).split("(")[0] == "RandomForestClassifier":
	method="RF"
else:
	method = "SVM"

#write the output into a file but don't delete the previous text
fileHandle = open ( options.output, 'a')

fileHandle.write(method+"\t"+trainset+"\t"+testset+"\t"+str(score) +"\n")
fileHandle.close()
print(score)

