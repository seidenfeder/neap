library(shiny)
library(plotly)
#Fuer Melt
library(reshape2)
library(ggplot2)

options(warn =-1)

shinyServer(
  function(input, output, session) {
    
    #Dynamically change avaliable methods (in the second panel Model development)
    observe({
      x <- input$type
      
      if(x == "c"){
        updateCheckboxGroupInput(session, "method", label="Methods", 
                                 choices = c("Random Forest" = "RF", 
                                             "Support Vector Machine" = "SVM"),
                                 selected = "RF")
      }
      else{
        updateCheckboxGroupInput(session, "method", label="Methods", 
                                 choices = c("Linear Regression" = "LR",
                                             "RF Regression" = "RF", 
                                             "SVM Regression" = "SVM"),
                                 selected = 1)
      }
    })
    
    #Dynamically change avaliable methods (in the third panel Dataset comparison)
    observe({
      x <- input$type_2
      
      if(x == "c"){
        updateCheckboxGroupInput(session, "method_2", label="Methods", 
                                 choices = c("Random Forest" = "RF", 
                                             "Support Vector Machine" = "SVM"),
                                 selected = "RF")
      }
      else{
        updateCheckboxGroupInput(session, "method_2", label="Methods", 
                                 choices = c("Linear Regression" = "LR",
                                             "RF Regression" = "RF", 
                                             "SVM Regression" = "SVM"),
                                 selected = 1)
      }
    })
    
    ####################################################################################
    # Plots for the model development tab
    
    #Create the plot of the label evaluation
    output$labelPlot<-renderPlotly({
      #Display the plot only for the classification task and if at least one method is selected
      if(input$type=="c" & ! is.null(input$method)){
        #Read input data
        data<-read.csv("PlotInput/evalLabels_normalized.txt",sep="\t",header=F)
        
        #Reformat data for the box plots
        reshapedData<-melt(data, id=c("V1","V2"))
        
        #Filter data according to the selected methods
        matches <- grepl(paste(input$method,collapse="|"), reshapedData$V1)
        plottedData<-reshapedData[matches,]
        
        
        #Set levels of plotted Data new to get a right scaling of the axis
        plottedData<-droplevels(plottedData)
        
        #Create interactive box plots
        plot_ly(y = plottedData$value, 
                x = plottedData$V1, 
                type="box")%>%
          layout(title = paste('Evaluation of different labeling methods'),
                 xaxis = list(
                   title = "Labeling method"),
                 yaxis = list(
                   title = "AUC Score"
                 )
          )
      }
      else{
        return(NULL)
      }
    })
    
    #Display an explanation text with the label plot
    output$labelText<-renderText({
      if(input$type=="c" & ! is.null(input$method)){
        return(paste("We tested three different labeling methods to separate the gene set in two classes.",
                     "We splitted either at the median gene expression value, the mean gene expression value",
                     "or at a expression value of zero. The method \"median\", which splits the genes in two equal sets",
                     "seems to work best."))
      }
      else{
        return(NULL)
      }
    })
    
    #Create the plot of plot for the bins
    output$binsPlot<-renderPlotly({
      #Display the plot only for the classification task and if at least one method is selected
      if(input$type=="c" & ! is.null(input$method)){
        #Read input data
        dataBinsC<-read.csv("PlotInput/evalBins.txt", sep="\t", header=F)
        
        #Filter data according to the selected methods
        matchesBins<- grepl(paste(input$method,collapse="|"), dataBinsC$V1)
        plottedData<-dataBinsC[matchesBins,]
        
        #Set levels of plotted Data new to get a right scaling of the axis
        plottedData<-droplevels(plottedData)
        
        #Create interactive line plots
        color1<-c("blue","red")
        plot_ly(y = rowMeans(plottedData[,3:ncol(plottedData)]),
                x = plottedData$V2, type="scatter", 
                color=plottedData$V1,
                colors = color1,
                
                mode="lines")%>%
          layout(title = paste('AUC score for each bin'),
                 xaxis = list(
                   title = "Bin"),
                 yaxis = list(
                   title = "AUC Score"
                 )
          )
      }
      else if(! is.null(input$method)){
        #Read input data
        dataBinsC<-read.csv("PlotInput/evalBinsReg.txt", sep="\t", header=F)
        
        #Filter data according to the selected methods
        matchesBins<- grepl(paste(input$method,collapse="|"), dataBinsC$V1)
        plottedData<-dataBinsC[matchesBins,]
        
        #Set levels of plotted Data new to get a right scaling of the axis
        plottedData<-droplevels(plottedData)
        
        #Create interactive line plots
        color1<-c("blue","red")
        plot_ly(y = rowMeans(plottedData[,3:ncol(plottedData)]),
                x = plottedData$V2, type="scatter", 
                color=plottedData$V1,
                colors = color1,
                
                mode="lines")%>%
          layout(title = paste('R2 score for each bin'),
                 xaxis = list(
                   title = "Bin"),
                 yaxis = list(
                   title = "R2 Score"
                 )
          )
      }
      else{
        return(NULL)
      }
    })
    
    ####################################################################################
    # Plots for the dataset comparison tab
    
    #Create plot for the different data sets
    output$cvData<-renderPlotly({
      if(! is.null(input$method_2)){
        
        #Read different files for classification and for regression
        #TODO: add right data files (after creating them ;) )
        if(input$type=="c"){
          filename = "PlotInput/performanceDatasets.txt"
          titleString = "AUC Score"
        }
        else{
          filename = "PlotInput/performanceDatasets.txt"
          titleString = "R2 Score"
        }
        #Read input data
        data<-read.csv(filename,sep="\t",header=F)
        
        #Reformat data for the box plots
        reshapedData<-melt(data, id=c("V1","V2", "V3"))
        
        #Filter data according to the selected methods
        matches <- grepl(paste(input$method_2,collapse="|"), reshapedData$V2)
        plottedData<-reshapedData[matches,]
        
        #Filter data acoording to the selected data sets
        matches <- grepl(paste(input$datasets_2,collapse="|"), plottedData$V1)
        plottedData<-plottedData[matches,]
        
        #Set levels of plotted Data new to get a right scaling of the axis
        plottedData<-droplevels(plottedData)
        
        #Create interactive box plots
        plot_ly(y = plottedData$value, 
                x = paste(plottedData$V1, " - ", plottedData$V2), 
                type="box")%>%
          layout(title = paste('Cross evaluation of different data sets using different methods'),
                 xaxis = list(
                   title = "Data set - method"),
                 yaxis = list(
                   title = titleString
                 )
          )
      }
    })
    
    #Comparison matrix of the different data sets
    output$dataMatrix<-renderPlotly({
      if(! is.null(input$method_2)){
        
        #Read different files for classification and for regression
        #TODO: add right data files (after creating them ;) )
        if(input$type=="c"){
          filename = "PlotInput/dataMatrix.txt"
          titleString = "AUC Score"
        }
        else{
          filename = "PlotInput/dataMatrix.txt"
          titleString = "R2 Score"
        }
        #Read input data
        data<-read.csv(filename,sep="\t",header=F)
        
        #Filter data according to the selected methods
        matches <- grepl(paste(input$method_2,collapse="|"), data$V1)
        plottedData<-data[matches,]
        
        #Filter data according to the selected data sets
        matches <- grepl(paste(input$datasets_2,collapse="|"), plottedData$V2)
        plottedData<-plottedData[matches,]
        
        matches <- grepl(paste(input$datasets_2,collapse="|"), plottedData$V3)
        plottedData<-plottedData[matches,]
        
        #If more than one method is selected, calculate the mean of all methods
        
        #Rename variables
        colnames(plottedData)<-c("Method","Dataset1","Dataset2","Score")
        #Create heatmap
        p<-ggplot(data = plottedData, aes(x = Dataset1, y = Dataset2)) +
          geom_tile(aes(fill = Score))+
          scale_fill_gradient(low = "white", high = "red")+
          ggtitle("Training on set 1, prediction set 2")+
          labs(x="Data set 1",y="Data set 2")
        
        ggplotly(p)
        
      }
    })
    
  })