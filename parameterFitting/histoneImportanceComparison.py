#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################################
#
# Method to compare performance when using only one histone modification with performance
# when using pairs of histne modifications
# (First the performance need to be calculate with the corresponding scripts, only the plotting part is done here)
#
####################################################################################################################

import numpy as np
#from optparse import OptionParser
import matplotlib.pyplot as plt

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--singleHist", type="string", dest="filenameSingle", help = "Performance file of single histone modifications")
parser.add_option("--pairHist", type="string", dest="filenameDouble", help = "Performance file of pairwise histone modifications")
parser.add_option("--method",help="Method to parse the results after")
parser.add_option("--dataset",help="Dataset to parse the results after")
(options, args) = parser.parse_args()

filenameSingle=options.filenameSingle
filenameDouble=options.filenameDouble
dataset=options.dataset
method=options.method

aucs=[]
modis=[]
colors=[]
#Read performance of the single modifications
fileRF = open(filenameSingle)
for line in fileRF.readlines():
    lineSplit=line.split()
    #Filter the dataset and the method
    if lineSplit[0] == dataset and lineSplit[1] == method:
        aucs.append(list(map(float,lineSplit[4:])))
        modis.append(lineSplit[3])
        if lineSplit[3]=='All':
            	colors.append('black')
        else:
            	colors.append('darkblue')

#Read performance of pairs of modifications    
fileRF = open(filenameDouble)
for line in fileRF.readlines():
    lineSplit=line.split()
    #Filter the dataset and the method
    if lineSplit[0] == dataset and lineSplit[1] == method:
        #Do not plot the performance of all bins twice
        if not lineSplit[3]=='None':
            	aucs.append(list(map(float,lineSplit[4:])))
            	modis.append(lineSplit[3])
            	colors.append('green')

#Remove all human notations in the string
modis=list(map(lambda x: x.replace("-human",""), modis))
modis=list(map(lambda x: x.replace("All","All histones"), modis))


#calculate the mean for bar plots
aucMean = np.mean(aucs, axis=1)
aucMean = aucMean.flatten()
aucMean= aucMean.tolist() 


#Sort the aucs (and corresponding the labels after size)
modis_sorted=[x for (y,x) in sorted(zip(aucMean,modis), reverse=True)]
colors_sorted=[x for (y,x) in sorted(zip(aucMean,colors), reverse=True)]
aucMean_sorted=sorted(aucMean,reverse=True)

# plot the mean as barplot
plt.figure(figsize=(12,5))
plt.bar(range(0,len(aucMean_sorted)),aucMean_sorted,width=0.6, align="center", color=colors_sorted)
plt.xlabel("Used Histone Modification")
plt.ylabel("Mean of AUC score")
plt.title("Performance of each possible pairwise histone combination")
plt.xticks(list(range(0,len(modis_sorted))),modis_sorted,fontsize=12,rotation=90)
plt.tight_layout()
plt.savefig('HistImportance_Comparison.png')
plt.show()
