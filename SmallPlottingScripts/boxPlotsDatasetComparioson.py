# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

#fileName="C:/Users/kathi/Documents/neap/InteractivePlots/PlotInput/performanceDatasets.txt"
fileName="C:/Users/kathi/Documents/neap/InteractivePlots/PlotInput/performanceDatasetsRegression.txt"
method="LR" #"LR","RF"
scoreName="R2 Score" #"AUC score"

score=[]
dataset=[]
resultFile=open(fileName)
for line in resultFile.readlines():
    line=line.rstrip()
    results=line.split('\t')
    if results[1] == method:
        score.append(list(map(float,results[3:])))
        if(results[0] == "gastrocnemius medialis"):
            dataset.append("gastrocnemius\nmedialis")
        else:
            dataset.append(results[0])

        
plt.figure(0)
plt.boxplot(score)
plt.xticks(range(1,len(dataset)+1),dataset, rotation='vertical')
plt.xlabel("Data set")
plt.ylabel(scoreName)
plt.title('Cross-validation performance of different datasets with ' + method)
plt.tight_layout()
plt.savefig('CrossVal_DataPerformance_'+method+'.png')
