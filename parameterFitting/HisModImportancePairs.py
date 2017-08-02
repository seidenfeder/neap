#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#####################################################################################
#
# This script tests the relative Importance of all tested Histone Modifications
# by running the classification with different subsets of the Histone Modifications 
#
#####################################################################################

import numpy as np
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import cross_val_score
from optparse import OptionParser
import matplotlib.pyplot as plt

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-m", type="string", dest="method", help = "the method you want to use Support Vector Machine (SVM) or Random Forest (RF) default= RF", default="RF")
parser.add_option("-i",dest="input", help="This gives the path to the file with the input data (the output of the binning)")
parser.add_option("-l",dest="labels", help="This gives the path to the file with the labels")
parser.add_option("-b",type = "int",dest="bin", help="Tells which bin should be used for the classification")
parser.add_option("-c",type = "int",dest="crossVal", help="Number of iterations in the cross validation", default=5)
parser.add_option("-a", action="store_true", dest="allBins", help = "Tells if all bins should be used", default=False)
parser.add_option("-o",dest="output", help="The name of the outputfile", default="classification.txt")
parser.add_option("-n", action="store_true", dest="newFormat", help="Feature file created by bins annotated, containing ENCODE metadata infos", default=False)
parser.add_option("--plotname",dest="plotname", help="Name of the result plot", default='HistImportanceSingleBars.png')
(options, args) = parser.parse_args()
method=options.method

labelFilename=options.labels
featureFilename=options.input
plotname=options.plotname

#Read labels
labelFile = open(labelFilename)
labelDict = dict()
for line in labelFile.readlines():
    if not line.startswith("##"):
        lineSplit=line.split()
        labelDict[lineSplit[0]]=int(lineSplit[2])


#Read features
featureFile=open(featureFilename)
genesModis=dict()

#In the new version of the annotated feature file there are additionally two header lines    
if options.newFormat :
    #Name of the data set (from the header)
    dataset=featureFile.readline().rstrip()[2:]
    dataset=dataset.split(" ")[0]
    #All modifications
    modifications=featureFile.readline().rstrip()[2:].split(" ")

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

##Sort labels according to the feature list
##Maybe for some genes no GENCODE entry could be found, these are only in the features list
#y=[]
#X=[]
##If not all bins should be used
#if(not options.allBins):
#	binNumber=options.bin
#	#Create feature matrix of the given bin 
#	for geneID in genesModis:
#	    y.append(labelDict[geneID])
#	    valueMatrix=np.array(genesModis[geneID])
#	    X.append(valueMatrix[:,binNumber])
##if you want the classification with all bins
#else:
#	for geneID in genesModis:
#	    y.append(labelDict[geneID])
#	    valueMatrix=np.array(genesModis[geneID])
#	    X.append(valueMatrix.flatten())
#
##Support Vector Machines
#if(method=="SVM"):
#    clf=svm.SVC(kernel="rbf")
##Random Forest
#elif(method=="RF"):
#    clf=RandomForestClassifier(n_estimators=12)
#
#scores = cross_val_score(clf, X, y, cv=options.crossVal, scoring='roc_auc')
#
##write the output into a file but don't delete the previous text
##this is necessary that we can compare different data sets or binnings or methods
#fileHandle = open ( options.output, 'a' )
#if(not options.allBins):
#    fileHandle.write(dataset+"\t"+method+"\t"+str(binNumber)+"\tNone\t"+'\t'.join(map(str,scores))+"\n")
#else:
#    fileHandle.write(dataset+"\t"+method+"\tall\tNone\t"+'\t'.join(map(str,scores))+"\n")
#fileHandle.close()

#Now we iterate through all Histone Modifications and run the classification method without the histone modification
for i in range(0,len(modifications)):
    for j in range(i+1,len(modifications)):
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
                valueMatrix=valueMatrix[i]
                X.append(valueMatrix[:,binNumber])
		#if you want the classification with all bins
        else:
            for geneID in genesModis:
                y.append(labelDict[geneID])
                valueMatrix=np.array(genesModis[geneID])
                modis=valueMatrix[i]
                modis=np.append(modis,valueMatrix[j])
                X.append(modis.flatten())

		#Support Vector Machines
        if(method == "RFC"):
            clf=RandomForestClassifier(n_estimators=12)
            scores = cross_val_score(clf, X, y, cv=options.crossVal, scoring='roc_auc')
        elif(method == "SVC"):
            clf=svm.SVC(kernel="rbf")
            scores = cross_val_score(clf, X, y, cv=options.crossVal, scoring='roc_auc')
        elif(method == "RFR"):
            rg=RandomForestRegressor(n_estimators=12)
            scores = cross_val_score(rg, X, y, cv=options.crossVal, scoring="r2")
        elif(method == "SVR"):
            rg=svm.SVR(cache_size=500)
            scores = cross_val_score(rg, X, y, cv=options.crossVal, scoring="r2")
        elif(method == "LR"):
            rg=linear_model.LinearRegression()
            scores = cross_val_score(rg, X, y, cv=options.crossVal, scoring="r2")

		#write the output into a file but don't delete the previous text
		#this is necessary that we can compare different data sets or binnings or methods
        fileHandle = open ( options.output, 'a' )
        if(not options.allBins):
            fileHandle.write(dataset+"\t"+method+"\t"+str(binNumber)+"\t"+modifications[i]+"-"+modifications[j]+"\t"+'\t'.join(map(str,scores))+"\n")
        else:
            fileHandle.write(dataset+"\t"+method+"\tall\t"+modifications[i]+"-"+modifications[j]+"\t"+'\t'.join(map(str,scores))+"\n")
        fileHandle.close()

aucs=[]
modis=[""]
fileRF = open(options.output)
for line in fileRF.readlines():
	lineSplit=line.split()
	aucs.append(list(map(float,lineSplit[4:])))
	modis.append(lineSplit[3])

#Remove all human notations in the string
modis=list(map(lambda x: x.replace("-human",""), modis))
modis=list(map(lambda x: x.replace("None","All histones"), modis))

#plot how the performance changes when we miss a histone modification
#plt.boxplot(aucs)
#plt.xlabel("Used Histone Modification")
#plt.ylabel("AUC score")
#plt.title("Performance of each possible pairwise histone combination")
#plt.xticks(list(range(0,len(modis))),modis,fontsize=7,rotation=90)
#plt.tight_layout()
#plt.savefig('HistImportance.png')
#plt.show()

##calculate the mean for bar plots
#aucMean = np.mean(aucs, axis=1)
#aucMean = aucMean.flatten()
#aucMean= aucMean.tolist() 
#
##remove one label because it's not needed for the barplot
#modis.remove("")
#
##Sort the aucs (and corresponding the labels after size)
#modis_sorted=[x for (y,x) in sorted(zip(aucMean,modis), reverse=True)]
#aucMean_sorted=sorted(aucMean,reverse=True)
#
## plot the mean as barplot
#plt.figure(0)
#plt.bar(range(0,len(aucMean_sorted)),aucMean_sorted,width=0.6, align="center")
#plt.xlabel("Used Histone Modification")
#plt.ylabel("Mean of AUC score")
#plt.title("Performance of each possible pairwise histone combination")
#plt.xticks(list(range(0,len(modis_sorted))),modis_sorted,fontsize=7,rotation=90)
##plt.ylim(0.84,0.9)
#plt.tight_layout()
#plt.savefig(plotname)
#plt.show()

