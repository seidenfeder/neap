#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################
#
# This script tests how the performance is influenced by diffent bin sizes
#
####################################################################################

import os
import matplotlib.pyplot as plt

#run the binning file with for differnent bin sizes and run always the RF classifaction afterwards
for i in range(50,225,25):
	os.system("python binning.py -a FirstExampleData/ENCFF047WAI.tsv -b FirstExampleData/ENCFF937GNL.tsv -d FirstExampleData/bigWigs/ -g FirstExampleData/gencode.v26.annotation.gtf --annot FirstExampleData/metadata.tsv -s "+str(i)+" --out output"+str(i)+".txt --protCod")
	os.system("python classification.py -i output"+str(i)+".txt -l labels_mRNA.tsv -n -c 5 -o BinSize.txt -a")

#get the calculated auc scores
aucs=[]
fileRF = open("BinSize.txt")
for line in fileRF.readlines():
	lineSplit=line.split()
	aucs.append(list(map(float,lineSplit[2:])))

#plot the AUCs in a boxplot
plt.boxplot(aucs)
plt.xlabel("bin size")
plt.ylabel("AUC score")
plt.title('AUC Score of Random Forest for different bin sizes')
plt.xticks(list(range(0,8)),[0,50,75,100,125,150,175,200,225])
plt.savefig('DifferentBinSizes.png')
plt.show()
