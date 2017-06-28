#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################
#
# This script tests how the Performance variert with a diffent number of total bins
#
####################################################################################

import os
import matplotlib.pyplot as plt
from optparse import OptionParser

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-t",action="store_true", dest="onlyTSS", help = "Tells if we only look at the TSS", default="False")

(options, args) = parser.parse_args()

#run the binning with different number of bins and run afterwards always the RF classification
for i in range(240,280,40):
	if options.onlyTSS:
		os.system("python binning.py -o -a FirstExampleData/ENCFF047WAI.tsv -b FirstExampleData/ENCFF937GNL.tsv -d FirstExampleData/bigWigs/ -g FirstExampleData/gencode.v26.annotation.gtf --annot FirstExampleData/metadata.tsv -n "+str(i)+" --out output"+str(i)+"TSS.txt --protCod")
	else:
		os.system("python binning.py -a FirstExampleData/ENCFF047WAI.tsv -b FirstExampleData/ENCFF937GNL.tsv -d FirstExampleData/bigWigs/ -g FirstExampleData/gencode.v26.annotation.gtf --annot FirstExampleData/metadata.tsv -n "+str(i)+" --out output"+str(i)+".txt --protCod")
	os.system("python classification.py -i output"+str(i)+".txt -l labels_mRNA.tsv -n -c 5 -o BinNumber.txt -a")

#get the calculated AUC values
aucs=[]
fileRF = open("BinNumber.txt")
for line in fileRF.readlines():
	lineSplit=line.split()
	aucs.append(list(map(float,lineSplit[2:])))

#Plot everything in a boxplot
plt.boxplot(aucs)
plt.xlabel("#Bins")
plt.ylabel("AUC score")
plt.title('AUC Score of Random Forest for different\ntotal number of bins')
plt.xticks(list(range(0,8)),[0,40,80,120,160,200,240])
plt.savefig('DifferentTotalNumberBins.png')
plt.show()
