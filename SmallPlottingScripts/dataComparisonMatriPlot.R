library(ggplot2)
        
#################
# Specify the parameter
setwd("/home/sch/schmidka/Dokumente/GeneExpressionPrediction/neap/InteractivePlots/")
type <- "c" #c or r
method <- "RF" #Possible SVM, RF, LR
plotName <- paste0("dataComparison_",method,"_",type,".png")

if(type=="c"){
  filename = "PlotInput/dataMatrix.txt"
  titleString = "AUC Score"
} else{
  filename = "PlotInput/dataMatrixReg.txt"
  titleString = "R2 Score"
}

#Read input data
data<-read.csv(filename,sep="\t",header=F)

#Set the gastro label better visible
data$V2<-as.character(data$V2)
data$V2[data$V2=="gastrocnemius medialis"]="Gastro"
data$V2[data$V2=="thyroid gland"]="thyroid\ngland"
data$V2[data$V2=="keratinocyte"]="keratino-\ncyte"

data$V3<-as.character(data$V3)
data$V3[data$V3=="gastrocnemius medialis"]="Gastro"

#Filter data according to the selected methods
matches <- grepl(paste(method,collapse="|"), data$V1)
plottedData<-data[matches,]

#Rename variables
colnames(plottedData)<-c("Method","Trainset","Testset","Score")
#Create heatmap
p<-ggplot(data = plottedData, aes(x = Trainset, y = Testset)) +
  geom_tile(aes(fill = Score))+
  scale_fill_gradient2(low = "white",mid="yellow", high = "red",midpoint=0.5, limits=c(0.0,1.0), name=paste0(titleString,"\n"))+
  ggtitle("Predicting on a different data set")+
  labs(x="Training Set",y="Test Set")+
  theme(axis.text=element_text(size=18),
        axis.title=element_text(size=20,face="bold"),
        plot.title = element_text(size=24),
        legend.text=element_text(size=18),
        legend.title=element_text(size=20))
p
#ggsave(plotName, p)
