#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-l", type="string", dest="labelFilename", help = "Label file")
parser.add_option("-f", type="string", dest="featureFilename", help = "Feature file with bins")

(options, args) = parser.parse_args()

#Files
labelFilename=options.labelFilename
featureFilename=options.featureFilename

#Read labels
labelFile = open(labelFilename)
labelDict = dict()
for line in labelFile.readlines():
    lineSplit=line.split()
    labelDict[lineSplit[0]]=int(lineSplit[2])
    

#Read features
featureFile=open(featureFilename)
#Name of the data set (from the header)
dataset=featureFile.readline().rstrip()[2:]
#All modifications
modifications=featureFile.readline().rstrip()[2:].split(" ")
#All modifications per gene
genesModis=dict()
for line in featureFile.readlines():
    line=line.rstrip()
    if(line.startswith('#')):
        lineSplit=line.split(" ")
        geneID=lineSplit[0]
        #Remove the hashtag at the beginning of the line
        geneID=geneID[1:]
        genesModis[geneID]=[]
    else:
        valueList=line.split(",")
        valueList=list(map(float,valueList))
        genesModis[geneID].append(valueList)

#Create huge feature vector X for the PCA and values y for labeling
y=[]
X=[]       
for geneID in genesModis:
    y.append(labelDict[geneID])
    valueMatrix=np.array(genesModis[geneID])
    X.append(valueMatrix.flatten())
X=np.array(X)
       
#PCA
pca = PCA(n_components=20)
pca.fit(X.transpose())

#Print explained variance
print("Explained variance ratio")
print(pca.explained_variance_ratio_)

#2D scatterplot
plt.figure()
plt.scatter(pca.components_[0], pca.components_[1], c=y, alpha=0.5)
plt.xlabel('First component')
plt.ylabel('Second component')
plt.title('PCA of ' + dataset)
plt.tight_layout()
plt.savefig('PCA_TwoComponents.png')
plt.show()

#3D scatterplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca.components_[0], pca.components_[1], pca.components_[2], c=y, alpha=0.5)
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
ax.set_zlabel('Third component')
ax.set_title('PCA of ' + dataset)
plt.tight_layout()
fig.show()
plt.savefig('PCA_ThreeComponents.png')
