# -*- coding: utf-8 -*-

#############################################################################################
#
# This method plots two heatmaps using the binned histone modification
# - a signal pattern matrix showing the distribution of each modification along the bins
# - a correlation matrix showing the correlation between the expression values and
#   the modification values in each bin
#
############################################################################################# 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-i", type="string", dest="filename", help = "Filename of binning file")
parser.add_option("-s", type="string", dest="signalPlot", help = "Signal plot", 
                  default='signalPattern.png')
parser.add_option("-c", type="string", dest="corrPlot", help = "Correlation plot", 
                  default='corrPattern.png')
parser.add_option("-S",dest="outFile", help="File to save the signal values", default ="")
parser.add_option("-C",dest="outFile2", help="File to save the correlation values", default ="")

(options, args) = parser.parse_args()

#Path to save the signal and the correlation plot
corrPlot=options.corrPlot
signalPlot=options.signalPlot
outFile =options.outFile
outFile2 =options.outFile2


#Read input file
filename=options.filename
inputFile = open(filename)


#Binning values per gene for each histone modification
genesModis=dict()
#Expresssion values per gene
geneExpression=dict()
#Name of the data set (from the header)
dataset=inputFile.readline().rstrip()[2:]
#All modifications
modifications=inputFile.readline().rstrip()[2:].split(" ")
#Parse the input file
for line in inputFile.readlines():
    #Parse gene header with ID and expression value
    if(line.startswith('#')):
        lineSplit=line.split(" ")
        geneID=lineSplit[0]
        #Remove the hashtag at the beginning of the line
        geneID=geneID[1:]
        genesModis[geneID]=[]
        geneExpression[geneID]=float(lineSplit[1])
    #Parse value line containing binned signal values
    else:
        valueList=line.split(",")
        valueList=list(map(float,valueList))
        genesModis[geneID].append(valueList)

#Close input file
inputFile.close()
 
print("Anzahl analysierte Gene:" + str(len(genesModis)))       
#Create signal distribution matrix
#Initialize entries
histonModis=dict()
firstGene=list(genesModis.keys())[1]

print("Anzahl Modifikationen " + str(len(genesModis[firstGene])))

#Assign array for each histone modification
for i in range(0,len(genesModis[firstGene])):
    histonModis[i]=[]

#Fill histone matrix
for gene in genesModis:
    histonRows=genesModis[gene]
    for i in range(0,len(histonRows)):
        histonModis[i].append(histonRows[i])
        
#Create calculate average signal for each histone modification and bin
averageSignal=[]
normalizedSignal=[]
for histonM in histonModis:
    summedHistons=[0]*len(histonModis[0][0])
    for i in range(0,len(histonModis[histonM])):
        summedHistons=[sum(x) for x in zip(histonModis[histonM][i],summedHistons)]
    average=[x/(i+1) for x in summedHistons]
    averageSignal.append(average)
    #Normalize row (according to Z scores)
    mean=np.mean(average)
    std=np.std(average)
    normalized=[(x-mean)/std for x in average]
    normalizedSignal.append(normalized)

  
#If interested, save results
#for histonM in histonModis:
    #...
    
#Plot heatmap with normalized signal pattern
heatmap=np.array(normalizedSignal)
numBins=int(len(heatmap[1])/2)
numBins2=int(len(heatmap[1])/4)
numBins4=int(len(heatmap[1])/8)

plt.figure()
ax = plt.gca()
im = ax.imshow(heatmap, interpolation='nearest', aspect='auto')
ax.vlines([numBins], *ax.get_ylim())
plt.title('Signal Pattern')
plt.xlabel('Bin')
plt.ylabel('Histone Modification')
plt.yticks(range(0,len(modifications)),modifications)
plt.xticks(list(range(0,numBins*2+1,numBins4)),
           [-numBins2,-numBins4,'TSS',numBins4,'',-numBins4,'TTS',numBins4,numBins2])

#Set size of colorbar according to the size of the plot
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

#Create colorbar
cbar=plt.colorbar(im, cax=cax)
cbar.set_label('Normalized signal p-value')
plt.tight_layout()
plt.savefig(signalPlot)
plt.show()

#Calculate correlation pattern
from scipy.stats.stats import spearmanr

expressionValues=list(geneExpression.values())

#Calculate the spearman correlation for each histon modification and bin
#Iterate over all modifications
corrMatrix=[]
for histonM in histonModis:
    valueMatrix=np.array(histonModis[histonM])
    #Iterate over all bins
    corrBin=[]
    for i in range(0,len(histonModis[histonM][0])):
        
        #Normalize values
        values=valueMatrix[:,i]
        mean=np.mean(values)
        std=np.std(values)
        normalizedValues=[(x-mean)/std for x in values]
        
        #Spearman correlation
        corrBin.append(spearmanr(normalizedValues,expressionValues)[0])
    corrMatrix.append(corrBin)

##Write now the signals and the correlation into the outputfile
signals=open(outFile,'w')
for i in range(0,len(normalizedSignal)):
    for j in range(0,len(normalizedSignal[0])):
        signals.write(str(normalizedSignal[i][j])+"\t")
    signals.write("\n")

corre=open(outFile2,'w')
for i in range(0,len(corrMatrix)):
    for j in range(0,len(corrMatrix[0])):
        corre.write(str(corrMatrix[i][j])+"\t")
    corre.write("\n")

#Plot heatmap with correlation matrix    
heatmap=np.array(corrMatrix)
    
plt.figure()
ax = plt.gca()
im = ax.imshow(heatmap, interpolation='nearest', aspect='auto')
ax.vlines([numBins], *ax.get_ylim())
plt.title('Correlation Pattern')
plt.xlabel('Bin')
plt.ylabel('Histone Modification')
plt.yticks(range(0,len(modifications)),modifications)
plt.xticks(list(range(0,numBins*2+1,numBins4)),
           [-numBins2,-numBins4,'TSS',numBins4,'',-numBins4,'TTS',numBins4,numBins2])

#Set size of colorbar according to the size of the plot
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

#Create colorbar
cbar=plt.colorbar(im, cax=cax)
cbar.set_label('Spearman correlation')
plt.tight_layout()
plt.savefig(corrPlot)
plt.show()

