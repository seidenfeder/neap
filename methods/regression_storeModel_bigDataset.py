#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################
#
# Adaption of the method regression_storeModel to run on our big merged dataset
# In the feature file only one header at the beginning of all cell types is allowed
#
# This script is able to run the three regression methods
# on a training set and save the gotten model.
#
####################################################################################################

import numpy as np
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestRegressor
from optparse import OptionParser
from math import log
from sklearn.externals import joblib

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-m", type="string", dest="method", help = "the method you want to use Support Vector Machine (SVM) or Random Forest (RF) or linear Regression (LR) default= LR", default="LR")
parser.add_option("-i",dest="input", help="This gives the path to the file with the input data (the output of the binning)")
parser.add_option("-b",type = "int",dest="bin", help="Tells which bin should be used for the classification")
parser.add_option("-a", action="store_true", dest="allBins", help = "Tells if all bins should be used", default=False)
parser.add_option("-o",dest="output", help="The name of the modelfile", default="regression.txt")
parser.add_option("-n", action="store_true", dest="newFormat", help="Feature file created by bins annotated, containing ENCODE metadata infos", default=False)
parser.add_option("-z", action="store_true", dest="zeros", help="Tells if you want to include the expression values 0.0 (default=false)", default=False)

(options, args) = parser.parse_args()
method=options.method
featureFilename=options.input
model=options.output


#Read values and features
featureFile=open(featureFilename)

#In the new version of the annotated feature file there are additionally two header lines    
if options.newFormat :
    #Name of the data set (from the header)
    dataset=featureFile.readline().rstrip()[2:]
    #All modifications
    modifications=featureFile.readline().rstrip()[2:].split(" ")
    
genesModis=dict()
values = dict()
#Gene id not unique anymore for the big data set, use a counter instead
c=0
for line in featureFile.readlines():
    line=line.rstrip()
    if(line.startswith('#')):
        c=c+1
        lineSplit=line.split(" ")
        geneID=lineSplit[0]
        #Remove the hashtag at the beginning of the line
        geneID=geneID[1:]
        genesModis[c]=[]
        values[c]=float(lineSplit[1])
    else:
        valueList=line.split(",")
        valueList=list(map(float,valueList))
        genesModis[c].append(valueList)

        

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

#Support Vector Machines
if(method=="SVM"):
    rg=svm.SVR(kernel='rbf')
#Random Forest
elif(method=="RF"):
    rg=RandomForestRegressor(n_estimators=12)
elif(method=="LR"):
    rg=linear_model.LinearRegression()

#Train the model
rg.fit(X,y)  

#Save the model
joblib.dump(rg, model) 
