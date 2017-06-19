#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#############################################################################################
#
# Variant of the script bins_adapted.py,
# where additionally a header for the result file is created out of a ENCODE metadata file
#
############################################################################################# 

import pyBigWig
import os


#first read the comandline arguments to know later how we should calculate the bins:
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-d", type="string", dest="bigWigDir", help = "directory containing bigwig files")
parser.add_option("-a", type="string", dest="fileRep1", help = "gene expression file (replicate 1)")
parser.add_option("-b", type="string", dest="fileRep2", help = "gene expression file (replicate 2)")
parser.add_option("--annot",type="string", dest="annotFilename", help = "Sample Annotation File of ENCODE")
parser.add_option("-g", type="string", dest="fileGencode", help = "gene annotation file")
parser.add_option("-n", type="int", dest="numberBins", help = "give the number of bins", default=160)
parser.add_option("-s", type="int", dest="sizeBins", help = "give the size of the bins",default=100)
parser.add_option("-o", dest="onlyTSS", help = "show bins only around TSS", default=False)
parser.add_option("--out", dest="output", help = "give the name of the output file", default="output.txt")

(options, args) = parser.parse_args()

fileRep1=options.fileRep1
fileRep2=options.fileRep2
#Parse the file of the first replicate
genesRep1=dict()
with open(fileRep1) as f:
	for line in f:
		lineSplit = line.strip().split()
		genesRep1[lineSplit[0]] = lineSplit[6]
fileRep1.close()
#Remove header line
del genesRep1['gene_id']
       
#Parse the file of the second replicategenesRep1=dict()
genesRep2=dict()
with open(fileRep2) as f:
	for line in f:
		lineSplit = line.strip().split()
		genesRep2[lineSplit[0]] = lineSplit[6]
fileRep2.close()
#Remove header line
del genesRep2['gene_id']
  
#Both gene lists contain the same genes (checked by comparison)
if(not (genesRep1.keys()==genesRep2.keys())):
	print("Gene lists unequal!")
     
#Calculate average score over the two lists
genesAverage=dict()
for gene in genesRep1.keys():
	genesAverage[gene]=(float(genesRep1[gene])+float(genesRep2[gene]))/2

fileGencode = options.fileGencode         
#Get transcription start site of each gene
geneTSS=dict()
geneTTS=dict()
geneChr=dict()
geneDirection=dict()
with open(fileGencode) as f:
	for line in f:
		if not line.startswith("##"):
			lineSplit = line.strip().split()
			if lineSplit[2]=='gene':
				gene_id=lineSplit[9][1:-2]
				if gene_id in genesAverage:
					geneChr[gene_id]=lineSplit[0]
					#save if it is on the + or on the minus strand 	
					if(lineSplit[6]=='+'):
						geneDirection[gene_id]="+"
						geneTSS[gene_id]=int(lineSplit[3])
						geneTTS[gene_id]=int(lineSplit[4])
					else:
						geneDirection[gene_id]="-"
						geneTSS[gene_id]=int(lineSplit[4])
						geneTTS[gene_id]=int(lineSplit[3])
fileGencode.close()


#Create file header
annotFilename=options.annotFilename
annotFile=open(annotFilename)
histonName=dict()
for line in annotFile.readlines():
    lineSplit=line.split("\t")
    histonName[lineSplit[0]]=lineSplit[16]
annotFile.close()
#Assuming that all samples are from the same cell line
cellLine=lineSplit[6]

#open all bigWig files in the bigWigs folder to calculate the bins
bigWigDir=options.bigWigDir
listFiles = os.listdir(bigWigDir)
histMods=[]
modificationNames=[]
for hist in listFiles:
	histMods.append(pyBigWig.open(bigWigDir+hist))
	accessID=hist.split(".")[0]
	modificationNames.append(histonName[accessID])

numberBins=options.numberBins 	#get how many bins we have
sizeBins=options.sizeBins	#get the size of each bin
onlyTSS = options.onlyTSS	#True if you only want to look at the TSS
typeBin = "mean"
inputFile = open(options.output,'w') #open a file where to write the input for the maschine leraning algorithms inside

#Write header
inputFile.write("##"+cellLine+"\n")
inputFile.write("##"+" ".join(modificationNames)+"\n")

#iterate over all genes
for gene in geneTSS:
	geneMatrix=[]
	 #iterate over all histone modifications   
	for bigWig in histMods:
		#fore the case we look at TSS and TTS
		if not onlyTSS:
			#start and end shows the range where the bins are for TSS and start1/end1 for TTS
			start = geneTSS[gene]-int(numberBins/4)*sizeBins
			end = geneTSS[gene]+(int(numberBins/4))*sizeBins-1
			start1 = geneTTS[gene]-int(numberBins/4)*sizeBins
			end1 = geneTTS[gene]+(int(numberBins/4))*sizeBins-1
			#Only calculate the bins if the start is not negative and the end is inside the chromosome
			if start>0 and start1>0 and end<=bigWig.chroms(geneChr[gene]) and end1<=bigWig.chroms(geneChr[gene]):
				#Calculate all bins between start and end
				binValues_TSS = bigWig.stats(geneChr[gene],start,end,nBins=int(numberBins/2),exact=True,type=typeBin)
				binValues_TTS = bigWig.stats(geneChr[gene],start1,end1,nBins=int(numberBins/2),exact=True,type=typeBin)	
				
				#Reverse the bins, if the gene is on the other strand
				if geneDirection[gene] == "-":
					binValues_TSS = list(reversed(binValues_TSS))
					binValues_TTS = list(reversed(binValues_TTS))

				binValues=binValues_TSS+binValues_TTS
				#save the values only if all bins have a value -> all bins are inside the chromosome
				if None not in binValues:
					geneMatrix.append(binValues)  
		 #If you only want to look at the TTS                 
		else:
			start = geneTSS[gene]-int(numberBins/2)*sizeBins
			end = geneTSS[gene]+(int(numberBins/2))*sizeBins-1
			#Only calculate the bins if the start is not negative
			if start>0 and end<=bigWig.chroms(geneChr[gene]):
				binValues = bigWig.stats(geneChr[gene],start,end,nBins=int(numberBins),exact=True,type=typeBin)
				#Reverse the bins, if the gene is on the other strand
				if geneDirection[gene] == "-":
					binValues = list(reversed(binValues))
				#save the values only if all bins have a value -> all bins are inside the chromosome
				if None not in binValues:
					geneMatrix.append(binValues)

	#Add only the gene to the file, if all modifications could be found
	if len(geneMatrix) == len(histMods):
		inputFile.write("#"+gene+" "+str(genesAverage[gene])+"\n")
		for row in geneMatrix:
			row=['{:.4f}'.format(x) for x in row]
			inputFile.write(','.join(row)+"\n")

#Close the file
inputFile.close()			
