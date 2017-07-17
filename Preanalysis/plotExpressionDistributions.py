#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from math import log
from scipy.stats.stats import pearsonr

#############################################################################################
#
# Analyze distribution of the expression values (of all protein coding genes):
# - Compare the two replicates
# - Show distribution of the average values between the replicates (also in log scale)
#
############################################################################################# 
     
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--rep1", type="string", dest="fileRep1", help = "Expression values of the first replicate (in Encode tsv format)")
parser.add_option("--rep2", type="string", dest="fileRep2", help = "Expression values of the second replicate (in Encode tsv format)")
parser.add_option("-g", type="string", dest="fileGencode", help = "Gencode annotation file to restrict the analysis to protein coding genes")

(options, args) = parser.parse_args()

fileRep1=options.fileRep1
fileRep2=options.fileRep2
fileGencode=options.fileGencode

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

#Correlation between replicates
valuesRep1=list(map(float,list(genesRep1.values())))
valuesRep2=list(map(float,list(genesRep2.values())))

plt.scatter(valuesRep1, valuesRep2)
corr = pearsonr(valuesRep1,valuesRep2)[0]
plt.title("Correlation between replicate 1 and 2 (r="+str(corr)+")")
plt.savefig("CorrelationExpressionReplicates.png")
plt.show()
    
#Total gene expression distribution
genesAverage=dict()   
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


values=list(genesAverage.values())
print("Maximal expression value: "+str(np.max(values)))
plt.hist(values, bins=list(range(0,300,10)))
plt.title("Histogram over expression levels (for values up to 300)")
plt.xlabel("Gene expression value")
plt.ylabel("Number of genes")
plt.savefig("ExpressionValueDistribution.png")
plt.show()

#Logarithmized distribution
#Natural logarithm
valuesLog=list(map(lambda x: log(x+0.0001),values))
#Log 10 logiarthm
#valuesLog=list(map(lambda x: np.log10(x+0.0001),values))
print("Maximal logarithmed expression value: "+str(np.max(valuesLog)))
plt.hist(valuesLog, bins='auto')
plt.title("Histogram over logarithmed expression levels")
plt.xlabel("Logarithmized gene expression value")
plt.ylabel("Number of genes")
plt.savefig("LogExpressionValueDistribution.png")
plt.show()

