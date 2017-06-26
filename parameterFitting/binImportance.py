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
(options, args) = parser.parse_args()
method=options.method

#run the method once for each bin
for i in range(0,160):
	if(method == "RFC"):
		os.system("python classification.py -i input_mRNA.txt -l labels_mRNA.tsv -n -c 5 -o "+method+"Bins.txt -b"+ str(i))
		score="AUC"
	elif(method == "SVC"):
		os.system("python classification.py -i input_mRNA.txt -l labels_mRNA.tsv -n -c 5 -o "+method+"Bins.txt -m SVM -b"+ str(i))
		score="AUC"
	elif(method == "RFR"):
		os.system("python regression.py -i input_mRNA.txt -m RF -n -c 5 -o "+method+"Bins.txt -b"+ str(i))
		score="r2"
	elif(method == "SVR"):
		os.system("python regression.py -i input_mRNA.txt -m SVM -n -c 5 -o "+method+"Bins.txt -b"+ str(i))
		score="r2"
	elif(method == "LR"):
		os.system("python regression.py -i input_mRNA.txt -n -c 5 -o "+method+"Bins.txt -b"+ str(i))
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
