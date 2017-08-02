library(ggplot2)
        
#################
# Specify the parameter
#setwd("C:/Users/kathi/Documents/neap/InteractivePlots/")
setwd("/home/sch/schmidka/Dokumente/GeneExpressionPrediction/neap/InteractivePlots/")
type <- "c" #c or r
method <- "SVM" #Possible SVM, RF, LR
plotName <- paste0("dataComparison_",method,".png")

if(type=="c"){
  filename = "PlotInput/dataMatrix.txt"
  titleString = "AUC Score"
} else{
  filename = "PlotInput/dataMatrixReg.txt"
  titleString = "R2 Score"
}

#Read input data
data<-read.csv(filename,sep="\t",header=F)

#Filter data according to the selected methods
matches <- grepl(paste(method,collapse="|"), data$V1)
plottedData<-data[matches,]

# Show all data sets
# #Filter data according to the selected data sets
# matches <- grepl(paste(input$datasets_2,collapse="|"), plottedData$V2)
# plottedData<-plottedData[matches,]
# 
# matches <- grepl(paste(input$datasets_2,collapse="|"), plottedData$V3)
# plottedData<-plottedData[matches,]

#Rename variables
colnames(plottedData)<-c("Method","Trainset","Testset","Score")
#Create heatmap
p<-ggplot(data = plottedData, aes(x = Trainset, y = Testset)) +
  geom_tile(aes(fill = Score))+
  scale_fill_gradient2(low = "white",mid="yellow", high = "red",midpoint=0.5, limits=c(0.0,1.0), name=paste0(titleString,"\n"))+
  ggtitle("Predicting on a different data set")+
  labs(x="Training Set",y="Test Set")

ggsave(plotName, p)
