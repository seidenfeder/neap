# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

filename="C:/Users/kathi/Documents/neap/InteractivePlots/PlotInput/evalBins.txt"
#filename="C:/Users/kathi/Documents/neap/InteractivePlots/PlotInput/evalBinsReg.txt"
dataset="K562"
score="AUC"
#score="R2"
plotname="ComparisonBins_"+dataset+"_Class.png"
#plotname="ComparisonBins_"+dataset+"_Reg.png"


aucs=dict()
fileRF = open(filename)
for line in fileRF.readlines():    
    lineSplit=line.split()
    if lineSplit[0] == dataset:
        #If already the first value for the method was saved
        if lineSplit[1] in aucs:
            aucs[lineSplit[1]].append(np.mean(list(map(float,lineSplit[3:]))))
        else:
            aucs[lineSplit[1]] = [np.mean(list(map(float,lineSplit[3:])))]

#Deep Learning data only for classification
#Einlesen der Deep-Learning data
filenameDL="C:/Users/kathi/Documents/neap/InteractivePlots/PlotInput/deepLearningBins.txt"

fileDL = open(filenameDL)
for line in fileDL.readlines():    
    lineSplit=line.split()
    if lineSplit[0] == dataset:
        #If already the first value for the method was saved
        if 'DL' in aucs:
            aucs['DL'].append(float(lineSplit[2]))
        else:
            aucs['DL'] = [float(lineSplit[2])]

#calculate the steps 
numBins=int(len(aucs['RF'])/2)
numBins2=int(len(aucs['RF'])/4)
numBins4=int(len(aucs['RF'])/8)

#make a plot with the mean values
plt.plot(range(0,160),aucs['RF'],label="Random Forest")
plt.plot(range(0,160), aucs['SVM'],label="Support Vector Machine")
plt.plot(range(0,160), aucs['DL'],label="Deep Learning")
#plt.plot(range(0,160), aucs['LR'],label="Linear Regression")
plt.xlabel("Bin")
plt.ylabel("Mean of "+score+" Score")
plt.axvline(x=80,color='black')
plt.axvline(x=40,color='r')
plt.axvline(x=120,color='r')
plt.xticks(list(range(0,numBins*2+1,numBins4)),[-numBins2,-numBins4,'TSS',numBins4,'',-numBins4,'TTS',numBins4,numBins2])
plt.title('Performance of '+dataset+' for each bin with different classification methods', fontsize=10)
plt.legend(loc=1, prop={'size': 6})
plt.savefig(plotname)

