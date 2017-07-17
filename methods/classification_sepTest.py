#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################
#
# This script is able to run the two classification methods Support Vector Machine and Random Forest 
# With a trainset for training and the model is tested on a testset afterwards
#
####################################################################################################

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from optparse import OptionParser
import pickle
from sklearn.metrics import roc_auc_score

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-m", type="string", dest="method", help = "the method you want to use Support Vector Machine (SVM) or Random Forest (RF) default= RF", default="RF")
parser.add_option("-i",dest="train", help="This gives the path to the file with the train dataset (binning file)")
parser.add_option("-l",dest="labels", help="This gives the path to the file with the labels fot´r the training data")
parser.add_option("-b",type = "int",dest="bin", help="Tells which bin should be used for the classification")
parser.add_option("-a", action="store_true", dest="allBins", help = "Tells if all bins should be used", default=False)
parser.add_option("-o",dest="output", help="The name of the outputfile", default="classification.txt")
parser.add_option("-n", action="store_true", dest="newFormat", help="Feature file created by bins annotated, containing ENCODE metadata infos", default=False)
parser.add_option("-t", dest="testset",help="This is the path to the test dataset, this is necessary if you used -t, otherwise it will be ignored")
parser.add_option("-L",dest="labelsTest", help="This gives the path to the file with the labels fot´r the training data")

(options, args) = parser.parse_args()
method=options.method
labelFilename=options.labels
featureFilename=options.train
testFile= options.testset
labelTest=options.labelsTest

#Read labels
labelFile = open(labelFilename)
labelDict = dict()
for line in labelFile.readlines():
    if not line.startswith("##"):
        lineSplit=line.split()
        labelDict[lineSplit[0]]=int(lineSplit[2])

#Read features
featureFile=open(featureFilename)

#In the new version of the annotated feature file there are additionally two header lines    
if options.newFormat :
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

###### Train the Model #######

#Support Vector Machines
if(method=="SVM"):
    clf=svm.SVC(kernel='rbf')
#Random Forest
elif(method=="RF"):
    clf=RandomForestClassifier(n_estimators=12)
#trainiere das model
clf.fit(X,y)  

##############################

##Read now the testset
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
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
pred = clf2.decision_function(Z)
scores = roc_auc_score(w,pred)

#write the output into a file but don't delete the previous text
#this is necessary that we can compare different data sets or binnings or methods
fileHandle = open ( options.output, 'a' )
if(not options.allBins):
    fileHandle.write(dataset+"\t"+method+"\t"+str(binNumber)+"\t"+'\t'.join(map(str,scores))+"\n")
else:
    fileHandle.write(dataset+"\t"+method+"\tall\t"+'\t'.join(map(str,scores))+"\n")
fileHandle.close()
