# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

filename="/home/sch/schmidka/Dokumente/GeneExpressionPrediction/neap/InteractivePlots/PlotInput/evalBins.txt"
score="AUC"
plotname="ComparisonBins_Datasets.png"

aucs=dict()
fileRF = open(filename)
for line in fileRF.readlines():    
    lineSplit=line.split()
    if lineSplit[0] != 'keratinocyte' and lineSplit[1]=='RF':
        #If already the first value for the method was saved
        if lineSplit[0] in aucs:
            aucs[lineSplit[0]].append(np.mean(list(map(float,lineSplit[3:]))))
        else:
            aucs[lineSplit[0]] = [np.mean(list(map(float,lineSplit[3:])))]

#calculate the steps 
numBins=int(len(aucs['K562'])/2)
numBins2=int(len(aucs['K562'])/4)
numBins4=int(len(aucs['K562'])/8)

#make a plot with the mean values
plt.plot(range(0,160),aucs['K562'],label="K562")
plt.plot(range(0,160), aucs['Endo'],label="Endo")
plt.plot(range(0,160), aucs['K562_short'],label="K562_short")
#plt.plot(range(0,160), aucs['LR'],label="Linear Regression")
plt.xlabel("Bin")
plt.ylabel("Mean of "+score+" Score")
plt.axvline(x=80,color='black')
plt.axvline(x=40,color='r')
plt.axvline(x=120,color='r')
plt.xticks(list(range(0,numBins*2+1,numBins4)),[-numBins2,-numBins4,'TSS',numBins4,'',-numBins4,'TTS',numBins4,numBins2])
plt.title('Performance for each bin of different datasets with RF', fontsize=12)
plt.legend(loc=1, prop={'size': 11})
plt.savefig(plotname)

