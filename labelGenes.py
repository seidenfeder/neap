#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################################################
#
# Script to assign the genes to different classes according to their expression value:
# Method 1 (Cheng et al): use the median value as cut-off (values bigger than the median get the label 1)
# Method 2 (Dong et al): use zero values (no expression) as cut-off (values bigger than 0 get the label 1)
#
##########################################################################################################
import numpy as np

#Command line parameters
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-a", type="string", dest="fileRep1", help = "gene expression file (replicate 1)")
parser.add_option("-b", type="string", dest="fileRep2", help = "gene expression file (replicate 2)")
parser.add_option("-m", type="string", dest="method", help = "cut-off method, values: median/zero",
                  default="median")
parser.add_option("-o", type="string", dest="labelFileName", help= "output label file",
                  default="")
parser.add_option("--protCod", action="store_true", dest="proteinCoding", default=False)
parser.add_option("-g", type="string", dest="fileGencode", help = "gene annotation file")

(options, args) = parser.parse_args()

fileRep1=options.fileRep1
fileRep2=options.fileRep2
method=options.method
labelFileName=options.labelFileName


#Parse the file of the first replicate
genesRep1=dict()
with open(fileRep1) as f:
	for line in f:
		lineSplit = line.strip().split()
		if(lineSplit[0].startswith("EN")):
			genesRep1[lineSplit[0]] = lineSplit[6]

       
#Parse the file of the second replicate
genesRep2=dict()
with open(fileRep2) as f:
	for line in f:
		lineSplit = line.strip().split()
		if(lineSplit[0].startswith("EN")):
			genesRep2[lineSplit[0]] = lineSplit[6]
  
#Both gene lists contain the same genes (checked by comparison)
if(not (genesRep1.keys()==genesRep2.keys())):
	print("Gene lists unequal!")
     
#Calculate average expression values over the two lists
genesAverage=dict()
for gene in genesRep1.keys():
	genesAverage[gene]=(float(genesRep1[gene])+float(genesRep2[gene]))/2

#Method 1: Median expression as cut-off
if(method=="median"):
    expVals=np.array(list(genesAverage.values()))
    cutOff=np.median(expVals)
#Method 2: Zero expression as cut-off
elif(method=="zero"):
    cutOff=0.0
else:
    print("The given method was not defined properly. Please enter either 'median' or 'zero' as proper method values.")
    exit()
 
#Create file with labels
labelFile = open(labelFileName,'w')

#Optionally save only labels for protein-coding genes
proteinCoding=options.proteinCoding
if proteinCoding:
    fileGencode = options.fileGencode        
    #Get all protein coding genes
    genesPC=dict()
    with open(fileGencode) as f:
        for line in f:
            if not line.startswith("##"):
                lineSplit = line.strip().split()
                if lineSplit[2]=='gene':
                    geneID=lineSplit[9][1:-2]
                    if lineSplit[11]=='"protein_coding";':
                        genesPC[geneID]=True
                    else:
                        genesPC[geneID]=False
    #Label protein coding genes
    for gene in genesAverage.keys():
            if gene in genesPC and genesPC[gene]:
                #Label each gene with one if the expression is above the threshold
                if(genesAverage[gene]>cutOff):
                    label=1
                else:
                    label=0
                labelFile.write(gene+"\t"+str(genesAverage[gene])+"\t"+str(label)+"\t"+"\n")
#Save labels for all genes                        
else:
    for gene in genesAverage.keys():
        #Label each gene with one if the expression is above the threshold
        if(genesAverage[gene]>cutOff):
            label=1
        else:
            label=0
        labelFile.write(gene+"\t"+str(genesAverage[gene])+"\t"+str(label)+"\t"+"\n")

labelFile.close()