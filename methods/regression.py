#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################
#
# This script is able to run the three Regression methods:
# Linear Regression, Support Vector Machine and Random Forest
# The output is the r2 scores of the cross validation with the chosen method
# If wanted the regression will be plotted
#
####################################################################################################

import numpy as np
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,cross_val_predict
from optparse import OptionParser
from math import log
import matplotlib.pyplot as plt

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-m", type="string", dest="method", help = "the method you want to use Support Vector Machine (SVM) or Random Forest (RF) or linear Regression (LR) default= LR", default="LR")
parser.add_option("-i",dest="input", help="This gives the path to the file with the input data (the output of the binning)")
parser.add_option("-b",type = "int",dest="bin", help="Tells which bin should be used for the classification")
parser.add_option("-c",type = "int",dest="crossVal", help="Number of iterations in the cross validation", default=5)
parser.add_option("-a", action="store_true", dest="allBins", help = "Tells if all bins should be used", default=False)
parser.add_option("-p", dest="plot", help = "True it makes you a plot, Flase(default) it makes no plot", default=False)
parser.add_option("-o",dest="output", help="The name of the outputfile", default="regression.txt")
parser.add_option("-n", action="store_true", dest="newFormat", help="Feature file created by bins annotated, containing ENCODE metadata infos", default=False)

(options, args) = parser.parse_args()
method=options.method

featureFilename=options.input

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
for line in featureFile.readlines():
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
	    if not val==0:
		    y.append(log(val))
		    valueMatrix=np.array(genesModis[geneID])
		    X.append(valueMatrix[:,binNumber])
#if you want the regression with all bins
else:
	for geneID in genesModis:
	    val = values[geneID]
	    if not val==0:
		    y.append(log(val))
		    valueMatrix=np.array(genesModis[geneID])
		    X.append(valueMatrix.flatten())


#Support Vector Machines
if(method=="SVM"):
    rg=svm.SVR(cache_size=500)
#Random Forest
elif(method=="RF"):
    rg=RandomForestRegressor(n_estimators=12)
elif(method=="LR"):
    rg=linear_model.LinearRegression()

#make the cross validation
scores = cross_val_score(rg, X, y, cv=options.crossVal, scoring="r2")

# plot the Regression if a plot is wanted
if options.plot:
    pred = cross_val_predict(rg, X, y, cv=options.crossVal)
    fig, ax = plt.subplots()
    ax.scatter(y, pred)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    corrCoef = np.mean(scores)
    plt.title("Regression calculated with "+ method+"\nCorrelation Coefficient: "+str(corrCoef))
    plt.show()

#write the output into a file but don't delete the previous text
fileHandle = open ( options.output, 'a')
if(not options.allBins):
    fileHandle.write(method+"\t"+str(binNumber)+"\t"+'\t'.join(map(str,scores))+"\n")
else:
    fileHandle.write(method+"\tall\t"+'\t'.join(map(str,scores))+"\n")
fileHandle.close()
