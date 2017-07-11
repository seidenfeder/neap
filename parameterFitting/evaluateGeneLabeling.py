#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################################################
#
# Script to evaluate the classification performance of the different label methods in labelGenes.py
# Uses RF and 10-fold crossvalidation
#
####################################################################################################

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser

#this is necessary to get the parameters from the comand line
parser = OptionParser()
parser.add_option("-a", type="string", dest="fileRep1", help = "gene expression file (replicate 1)")
parser.add_option("-b", type="string", dest="fileRep2", help = "gene expression file (replicate 2)")
parser.add_option("-o", type="string", dest="labelFileName", help= "output label file",
                  default="")
parser.add_option("-g", type="string", dest="fileGencode", help = "gene annotation file")
parser.add_option("-i",dest="input", help="This gives the path to the file with the input data (the output of the binning)")

(options, args) = parser.parse_args()

fileRep1=options.fileRep1
fileRep2=options.fileRep2
labelFileName=options.labelFileName
gencode=options.fileGenecode
inputFile=options.input

#For each label method run the label script and the classification ones
methods=["median","mean","zero"]
classificationMethods=["RF","SVM"]
for m in methods:
    print(m)
    os.system("python preprocessing/labelGenes.py -a "+fileRep1+
              "-b "+fileRep2+" -m "+ m + " -o "+labelFileName+ 
              " --protCod -g "+genecode)
    
    for cM in classificationMethods:
        print(cM)
        os.system("python methods/classification.py -i "+inputFile+" -l "+labelFileName+ 
                  "-c 10 -a -o evalLabels_normalized.txt -n -m "+cM)

#Get the scores from the result file
#get the calculated score values from the file
scores=[]
fileLabel = open("evalLabels_normalized.txt")
for line in fileLabel.readlines():
	lineSplit=line.split()
	scores.append(list(map(float,lineSplit[2:])))
    
#Plot the results
plt.boxplot(scores)
labels=[a+" "+b for a in methods for b in classificationMethods]
plt.xticks(range(1,len(labels)+1),labels)
plt.xlabel("Labeling method")
plt.ylabel("AUC score")
plt.title('AUC Score of RF for different labeled sets')
plt.savefig('evalLabels_normalized.png')
#plt.show()
                                                                                         
