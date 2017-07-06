#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#
# This script runs the Rnadom Forest method with every bin 
# to evaluate which is the most important bin
#
##############################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-m", type="string", dest="method", help = "the method you want to use Support Vector Machine - Classification (SVC) or Regression (SVR) or Random Forest - Classification (RFC) or Regression (RFR) or Linear Regression (LR) default= RFC", default="RFC")
parser.add_option("-i",dest="input", help="This gives the path to the file with the input data (the output of the binning)")
parser.add_option("-n",dest="name", help="Give the name of the cell line for better naming", default="")
parser.add_option("-l",dest="labels", help="This gives the path to the file with the labels")
(options, args) = parser.parse_args()
method=options.method
features=options.input
name = options.name
labels=options.labels

#run the method once for each bin
for i in range(0,160):
	if(method == "RFC"):
		os.system("python methods/classification.py -i "+features+" -l "+labels+" -n -c 5 -o "+method+"Bins.txt -b"+ str(i))
		score="AUC"
	elif(method == "SVC"):
		os.system("python methods/classification.py -i "+features+" -l "+labels+" -n -c 5 -o "+method+"Bins.txt -m SVM -b"+ str(i))
		score="AUC"
	elif(method == "RFR"):
		os.system("python methods/regression.py -i "+features+" -m RF -n -c 5 -o "+method+"Bins.txt -b"+ str(i))
		score="r2"
	elif(method == "SVR"):
		os.system("python methods/regression.py -i "+features+" -m SVM -n -c 5 -o "+method+"Bins.txt -b"+ str(i))
		score="r2"
	elif(method == "LR"):
		os.system("python methods/regression.py -i "+features+" -n -c 5 -o "+method+"Bins.txt -b"+ str(i))
		score="r2"

#get the calculated score values from the file
aucs=[]
fileRF = open(method+"Bins.txt")
for line in fileRF.readlines():
	lineSplit=line.split()
	aucs.append(list(map(float,lineSplit[2:])))

#calculate the mean for each bin
aucMean = np.mean(aucs, axis=1)
aucMean = aucMean.flatten()
aucMean= aucMean.tolist() 
#calculate the steps 
numBins=int(len(aucMean)/2)
numBins2=int(len(aucMean)/4)
numBins4=int(len(aucMean)/8)
#make a plot with the mean values
plt.plot(aucMean)
plt.xlabel("Bin")
plt.ylabel("Mean of "+score+" Score")
plt.axvline(x=80,color='black')
plt.axvline(x=40,color='r')
plt.axvline(x=120,color='r')
plt.xticks(list(range(0,numBins*2+1,numBins4)),[-numBins2,-numBins4,'TSS',numBins4,'',-numBins4,'TTS',numBins4,numBins2])
plt.title(score+' Score of '+method+' for each Bin')
plt.savefig(method+'CompareBinsRFPlot.png')
plt.show()
