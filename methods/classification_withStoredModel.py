#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################
#
# This script is able to run a classification task if the trained model is stored in a file.
#
####################################################################################################

import numpy as np
from optparse import OptionParser
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-i", dest="testset",help="This is the path to the test dataset, this is necessary if you used -t, otherwise it will be ignored")
parser.add_option("-l",dest="labelsTest", help="This gives the path to the file with the labels fot´r the training data")
parser.add_option("-b",type = "int",dest="bin", help="Tells which bin should be used for the classification")
parser.add_option("-a", action="store_true", dest="allBins", help = "Tells if all bins should be used", default=False)
parser.add_option("-m",dest="modelFile", help="Model file where the trained classificator is saved in.")
parser.add_option("-o",dest="output", help="The name of the outputfile", default="classification_pretrained.txt")
parser.add_option("-n", action="store_true", dest="newFormat", help="Feature file created by bins annotated, containing ENCODE metadata infos", default=False)

(options, args) = parser.parse_args()

testFile= options.testset
labelTest=options.labelsTest
modelFile=options.modelFile

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

#write the output into a file but don't delete the previous text
#this is necessary that we can compare different data sets or binnings or methods
fileHandle = open ( options.output, 'a' )
if(not options.allBins):
    fileHandle.write(testset+"\t"+str(clf).split("(")[0]+"\t"+str(binNumber)+"\t"+ str(score) +"\n")
else:
    fileHandle.write(testset+"\t"+str(clf).split("(")[0]+"\tall\t"+str(score) +"\n")
fileHandle.close()
print(score)

