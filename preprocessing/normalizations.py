#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:35:53 2017

@author: schmidka
"""
####################################################################################################
#
# This script scales or normalizes the binned feature data after different ideas:
# ZScoreHist: Scale each histon row of each sample using Z-scores 
# logZScoreHist: Scale each histon row of each sample using logarithm and Z-scores
# logHist: Scale values using logarithm   
# Scale: Scaling of all data (Z scores) (implemented by sklearn)
# robustScale: Robust scaling (if there are many outliers) (implemented by sklearn)
# Norm: Normalization of data (implemented by sklearn)
#
####################################################################################################

import numpy as np
from sklearn import preprocessing

from optparse import OptionParser

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-i",dest="featureFilename", help="Feature file containing binned histone values for normalization")
parser.add_option("-o",dest="outputFilename", help="Output file with the normalized values")
parser.add_option("-m",dest="normalizationMethod", help="Normalize feature matrix with different possible methods (individual or predefined by sklearn)", default="Norm")
parser.add_option("-n", action="store_true", dest="newFormat", help="Feature file created by bins annotated, containing ENCODE metadata infos", default=False)
(options, args) = parser.parse_args()

newFormat=options.newFormat
featureFilename=options.featureFilename
outputFilename=options.outputFilename
normalizationMethod=options.normalizationMethod


#Read features
featureFile=open(featureFilename)

#In the new version of the annotated feature file there are additionally two header lines    
if newFormat :
    #Name of the data set (from the header)
    dataset=featureFile.readline()
    #All modifications
    modifications=featureFile.readline()
    

genesModis=dict()
geneLine=dict()
for line in featureFile.readlines():
    line=line.rstrip()
    if(line.startswith('#')):
        lineSplit=line.split(" ")
        geneID=lineSplit[0]
        #Remove the hashtag at the beginning of the line
        geneID=geneID[1:]
        genesModis[geneID]=[]
        geneLine[geneID]=line
    else:
        valueList=line.split(",")
        valueList=list(map(float,valueList))
        genesModis[geneID].append(valueList)

#get the number of histone modifications and the number of bins from the feature file
numberHistons=len(genesModis[geneID])
numberBins=len(genesModis[geneID][0])

    
#Different normalization methods
normalizedModis=dict()
#Scale each histon row of each sample using Z-scores
if normalizationMethod=='ZScoreHist':

    for gene in genesModis:
        normalizedModis[gene]=[]
        for hist in range(0,len(genesModis[gene])):
            histonRow=genesModis[gene][hist]
            mean=np.mean(histonRow)
            std=np.std(histonRow)
            #Check if standard deviation is 0 (no division by 0!)
            if std==0:
                normalized=[x-mean for x in histonRow]
            else:
                normalized=[(x-mean)/std for x in histonRow]
            normalizedModis[gene].append(normalized)
#Scale each histon row of each sample using logarithm and Z-scores
elif normalizationMethod=='logZScoreHist':
    for gene in genesModis:
        normalizedModis[gene]=[]
        for hist in range(0,len(genesModis[gene])):
            #Calculate logarithm for each histone
            histonRow=[np.log(x+0.0000000001) for x in genesModis[gene][hist]]
            mean=np.mean(histonRow)
            std=np.std(histonRow)
            #Check if standard deviation is 0 (no division by 0!)
            if std==0:
                normalized=[x-mean for x in histonRow]
            else:
                normalized=[(x-mean)/std for x in histonRow]
            normalizedModis[gene].append(normalized)
#Scale values using logarithm         
elif normalizationMethod =='logHist':
    for gene in genesModis:
        normalizedModis[gene]=[]
        for hist in range(0,len(genesModis[gene])):   
            histonRow=[np.log(x+0.0000000001) for x in genesModis[gene][hist]]
            normalizedModis[gene].append(histonRow)
#Normalization methods of sklearn
elif normalizationMethod in ['Scale','robustScale','Norm']:
    #Create feature matrix
    X=[]
    for geneID in genesModis:
        valueMatrix=np.array(genesModis[geneID])
        X.append(valueMatrix.flatten())        
    
    #Scaling of the data (Z scores)
    if normalizationMethod=='Scale':
        scaled_X=preprocessing.scale(X)
    #Robust scaling (if there are many outliers)
    elif normalizationMethod=='robustScale':
        scaled_X=preprocessing.robust_scale(X)
    #Normalization of data
    elif normalizationMethod=='Norm':
        scaled_X=preprocessing.normalize(X)
    
    #Save normalized matrix again for normalizedModis
    genNames=list(genesModis.keys())
    for i in range(0,len(scaled_X)):
        a=scaled_X[i]
        a.resize(numberHistons,numberBins)
        normalizedModis[genNames[i]]=a
else:
    print("No known normalization method used")
    exit()
    

#Write result to the output file
outputFile=open(outputFilename,'w')
if newFormat :
    outputFile.write(dataset)
    outputFile.write(modifications)
    
#Write normalized values to file
for gene in normalizedModis:
    #Write header of the gene
    outputFile.write(geneLine[gene]+"\n")
    for hist in range(0,len(normalizedModis[gene])):
        row=['{:.4f}'.format(x) for x in normalizedModis[gene][hist]]
        outputFile.write(','.join(row)+"\n")
        
        
