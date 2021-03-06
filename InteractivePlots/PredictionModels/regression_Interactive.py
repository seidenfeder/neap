#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################
#
# This script is able to run a regression task if the trained model is stored in a file.
# Implemented to be called by the interactive server to predict a new data set.
#
####################################################################################################

import numpy as np
from optparse import OptionParser
from sklearn.metrics import r2_score
from math import log

from sklearn.externals import joblib

#this is necessary to get the parameters from the comand line
parser = OptionParser()

parser.add_option("-i", dest="testset",help="This is the path to the test dataset")
parser.add_option("-b",type = "int",dest="bin", help="Tells which bin should be used for the classification")
parser.add_option("-m",dest="modelFile", help="Model file where the trained classificator is saved in.")
parser.add_option("-a", action="store_true", dest="allBins", help = "Tells if all bins should be used", default=False)
parser.add_option("-n", action="store_true", dest="newFormat", help="Feature file created by bins annotated, containing ENCODE metadata infos", default=False)
parser.add_option("-z", action="store_true", dest="zeros", help="Tells if you want to include the expression values 0.0 (default=false)", default=False)

(options, args) = parser.parse_args()
testFile= options.testset
modelFile=options.modelFile

#Read values and features
testfile=open(testFile)

#In the new version of the annotated feature file there are additionally two header lines    
if options.newFormat :
    #Name of the data set (from the header)
    testset=testfile.readline().rstrip()[2:]
    #All modifications
    modifications=testfile.readline().rstrip()[2:].split(" ")
    
genesModis=dict()
values = dict()
for line in testfile.readlines():
    line=line.rstrip()
    if(line.startswith('#')):
        lineSplit=line.split(" ")
        geneID=lineSplit[0]
        #Remove the hashtag at the beginning of the line
        geneID=geneID[1:]
        genesModis[geneID]=[]
        values[geneID]=float(lineSplit[1])
    else:
        valueList=line.split(",")
        valueList=list(map(float,valueList))
        genesModis[geneID].append(valueList)

        

#Get the expression Value and the signal of the histone modifications in the bins
y=[]
X=[]
#If not all bins should be used
if(not options.allBins):
	binNumber=options.bin
	#Create feature matrix of the given bin 
	for geneID in genesModis:
	    val = values[geneID]
	    if options.zeros:
		    val=val+0.0001
		    y.append(log(val))
		    valueMatrix=np.array(genesModis[geneID])
		    X.append(valueMatrix[:,binNumber])
	    else:
		    if not val==0: 
			    y.append(log(val))
			    valueMatrix=np.array(genesModis[geneID])
			    X.append(valueMatrix[:,binNumber])
#if you want the regression with all bins
else:
	for geneID in genesModis:
	    val = values[geneID]
	    if options.zeros:
		    val=val+0.0001
		    y.append(log(val))
		    valueMatrix=np.array(genesModis[geneID])
		    X.append(valueMatrix.flatten())
	    else:
		    if not val==0:
			    y.append(log(val))
			    valueMatrix=np.array(genesModis[geneID])
			    X.append(valueMatrix.flatten())



############ Test the model #######################

rg = joblib.load(modelFile) 
pred = rg.predict(X)
score=r2_score(y, pred)

#get the used method
if str(rg).split("(")[0] == "RandomForestRegressor":
	method="RF"
elif str(rg).split("(")[0] == "LinearRegression":
	method = "LR"
else:
	method = "SVM"


print(score)
