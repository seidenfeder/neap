#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################################################
#
# Script to assign the genes to different classes according to their expression value:
# Method 1 (Cheng et al): use the median value as cut-off (values bigger than the median get the label 1)
# Method 2 (Dong et al): use zero values (no expression) as cut-off (values bigger than 0 get the label 1)
# Method 3: use the mean value as cut-off (values bigger than the mean get the label 1)
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
#Optionally save only labels for protein-coding genes
proteinCoding=options.proteinCoding
genesAverage=dict()
if proteinCoding:
    fileGencode = options.fileGencode        
    #Get all protein coding genes from the GENCODE annotation file
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
        
    for gene in genesRep1.keys():
        #Save only expression values for protein coding genes
        if(gene in genesPC and genesPC[gene]):
            	genesAverage[gene]=(float(genesRep1[gene])+float(genesRep2[gene]))/2
                        
else:
    for gene in genesRep1.keys():
        	genesAverage[gene]=(float(genesRep1[gene])+float(genesRep2[gene]))/2

#Method 1: Median expression as cut-off
if(method=="median"):
    expVals=np.array(list(genesAverage.values()))
    cutOff=np.median(expVals)
#Method 2: Zero expression as cut-off
elif(method=="zero"):
    cutOff=0.0
#Method 3: Mean expression as cut-off
elif(method=="mean"):
    expVals=np.array(list(genesAverage.values()))
    cutOff=np.mean(expVals)   
else:
    print("The given method was not defined properly. Please enter either 'median', 'mean' or 'zero' as proper method values.")
    exit()

#Print the used method and chosen threshold
print("Method: "+method)
print("Calculated threshold for classification: "+ str(cutOff))

#Save labels for each gene
geneLabels=dict()
#Count number of on and off-labels
posLabels=0
negLabels=0  
for gene in genesAverage.keys():
    #Label each gene with one if the expression is above the threshold
    if(genesAverage[gene]>cutOff):
        geneLabels[gene]=1
        posLabels=posLabels+1
    else:
        geneLabels[gene]=0
        negLabels=negLabels+1


#Print the number of genes with label 0 and with label 1
print("Number of genes with label 1: " + str(posLabels))
print("Number of genes with label 0: " + str(negLabels))

#Create file with labels
labelFile = open(labelFileName,'w')

#Save header
labelFile.write("## Method: "+method+"\n")
labelFile.write("## Calculated threshold for classification: "+ str(cutOff)+"\n")
labelFile.write("## Number of genes with label 1: " + str(posLabels)+"\n")
labelFile.write("## Number of genes with label 0: " + str(negLabels)+"\n")

#Save label for each file
for gene in genesAverage.keys():
    labelFile.write(gene+"\t"+str(genesAverage[gene])+"\t"+str(geneLabels[gene])+"\t"+"\n")
labelFile.close()
