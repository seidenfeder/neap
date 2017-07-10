library(shiny)
library(plotly)
#Fuer Melt
library(reshape2)

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
                                 selected = "RF")
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
                                 selected = "RF")
      }
    })
    
    
    #Create the plot of the label evaluation
    output$labelPlot<-renderPlotly({
      #Display the plot only for the classification task and if at least one method is selected
      if(input$type=="c" & ! is.null(input$method)&!is.null(input$datasets)){
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
    
    #Create the plot of plot for the bins
    output$binsPlot<-renderPlotly({
      #Display the plot only for the classification task and if at least one method is selected
      if(input$type=="c" & ! is.null(input$method)&!is.null(input$datasets)){
        #Read input data
        dataBinsC<-read.csv("PlotInput/evalBins.txt", sep="\t", header=F)
        
        #Filter data according to the selected methods
        matchesBins<- grepl(paste(input$method,collapse="|"), dataBinsC$V2)
        plottedData<-dataBinsC[matchesBins,]
        
        #Filter data according to the selected cell lines
        matchesBinsCell<- grepl(paste(input$datasets,collapse="|"), plottedData$V1)
        plottedDataCell<-plottedData[matchesBinsCell,]
        
        #Get different labels and colors for differnt datasets and methods
        NeededColors <- paste(plottedDataCell$V1,plottedDataCell$V2,sep=" - ")
        
        #Set levels of plotted Data new to get a right scaling of the axis
        plottedDataCell<-droplevels(plottedDataCell)
        
        #Create interactive line plots
        color1<-c("blue","red")
        plot_ly(y = rowMeans(plottedDataCell[,4:ncol(plottedDataCell)]),
                x = plottedDataCell$V3, type="scatter", 
                color=NeededColors,
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
      else if(! is.null(input$method)&!is.null(input$datasets)){
        #Read input data
        dataBinsC<-read.csv("PlotInput/evalBinsReg.txt", sep="\t", header=F)
        
        #Filter data according to the selected methods
        matchesBins<- grepl(paste(input$method,collapse="|"), dataBinsC$V2)
        plottedData<-dataBinsC[matchesBins,]
        
        #Filter data according to the selected cell lines
        matchesBinsCell<- grepl(paste(input$datasets,collapse="|"), plottedData$V1)
        plottedDataCell<-plottedData[matchesBinsCell,]
        
        #Get different labels and colors for differnt datasets and methods
        NeededColors <- paste(plottedDataCell$V1,plottedDataCell$V2,sep=" - ")
        
        #Set levels of plotted Data new to get a right scaling of the axis
        plottedDataCell<-droplevels(plottedDataCell)
        
        #Create interactive line plots
        color1<-c("blue","red")
        plot_ly(y = rowMeans(plottedDataCell[,4:ncol(plottedDataCell)]),
                x = plottedDataCell$V3, type="scatter", 
                color=NeededColors,
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
    
    
    
  })