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
      if(input$type == "c"){
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
      if(input$type_2 == "c"){
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
    
    output$dynamic <- renderUI({
      
      if (input$type == "c") { 
        tabsetPanel(
          tabPanel("Gene labels",
                   br(),
                   plotlyOutput("labelPlot"),
                   br(),
                   textOutput("labelText")
          ),
          tabPanel("Bin importance per bin",
                   br(),
                   plotlyOutput("binsPlot")
          ),
          tabPanel("Normalization",
                   br(),
                   plotlyOutput("normPlot")
          )
        )
      } else {
        tabsetPanel(
          tabPanel("Bin importance per bin",
                   br(),
                   plotlyOutput("binsPlot")
          ),
          tabPanel("Normalization",
                   br(),
                   plotlyOutput("normPlot")
          )
        )
      }
      })
      
    ####################################################################################
    # Plots for the model development tab
    
    #Create the plot of the label evaluation
    output$labelPlot<-renderPlotly({
      #Display the plot only for the classification task and if at least one method is selected
      if(input$type=="c" & ! is.null(input$method)&!is.null(input$datasets)){
        #Read input data
        data<-read.csv("PlotInput/evalLabels_normalized.txt",sep="\t",header=F)
        #produce the right labels
        data$names<-paste(data$V1,data$V2,data$V3,sep=" - ")
        
        #Reformat data for the box plots
        reshapedData<-melt(data, id=c("V1","V2","V3","V4","names"))
        
        #Filter data according to the selected methods
        matches <- grepl(paste(input$method,collapse="|"), reshapedData$V2)
        plottedData<-reshapedData[matches,]
        
        #Filter data according to the selected cell lines
        matchesBinsCell<- grepl(paste(input$datasets,"$",collapse="|",sep=""), plottedData$V1)
        plottedDataCell<-plottedData[matchesBinsCell,]
        
        
        #Set levels of plotted Data new to get a right scaling of the axis
        plottedDataCell<-droplevels(plottedDataCell)
        
        #Create interactive box plots
        plot_ly(y = plottedDataCell$value, 
                x = plottedDataCell$names, 
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
      if(!is.null(input$method)&!is.null(input$datasets)){
        #Read input data
        if(input$type=="c"){
          dataBinsC<-read.csv("PlotInput/evalBins.txt", sep="\t", header=F)
          yAxisTitle<-"AUC Score"
        }
        else{
          dataBinsC<-read.csv("PlotInput/evalBinsReg.txt", sep="\t", header=F)
          yAxisTitle<-"R2 Score"
        }
        
        #Filter data according to the selected methods
        matchesBins<- grepl(paste(input$method,collapse="|"), dataBinsC$V2)
        plottedData<-dataBinsC[matchesBins,]
        
        #Filter data according to the selected cell lines
        matchesBinsCell<- grepl(paste(input$datasets,"$",collapse="|",sep=""), plottedData$V1)
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
          layout(title = paste('Performance for each bin'),
                 xaxis = list(
                   title = "Bin"),
                 yaxis = list(
                   title = yAxisTitle
                 )
          )
      }
      else{
        return(NULL)
      }
    })
    
    #Create the plot of the normalization evaluation
    output$normPlot<-renderPlotly({
      #Display the plot only for the classification task and if at least one method is selected
      if(! is.null(input$method)&!is.null(input$datasets)){
        #Read input data
        if(input$type=="c"){
          data<-read.csv("PlotInput/evalNormalisation.txt",sep="\t",header=F)
          yaxisTitle<-"AUC Score"
        }
        else{
          data<-read.csv("PlotInput/evalNormalisationReg.txt",sep="\t",header=F)
          yaxisTitle<-"R2 Score"
        }
        #produce the right labels
        data$names<-paste(data$V1,data$V2,data$V3,sep=" - ")
        
        #Filter data according to the selected datasets
        data<-data[data$V1 %in% input$datasets,]
        
        #Reformat data for the box plots
        reshapedData<-melt(data, id=c("V1","V2","V3","V4","names"))
        
        #Filter data according to the selected method
        matches <- grepl(paste(input$method,collapse="|"), reshapedData$V2)
        plottedData<-reshapedData[matches,]
                
        #Set levels of plotted Data new to get a right scaling of the axis
        plottedData<-droplevels(plottedData)
        
        #Create interactive box plots
        plot_ly(y = plottedData$value, 
                x = plottedData$names, 
                type="box")%>%
          layout(title = paste('Evaluation of different normalization methods'),
                 xaxis = list(
                   title = "Normalization method"),
                 yaxis = list(
                   title = yaxisTitle
                 )
          )
      }
      else{
        return(NULL)
      }
    })
    
  
    ####################################################################################
    # Plots for the regression plot tab
    
    output$regressionScatterplot<-renderPlotly({
      if(! is.null(input$method_reg)){
        #Read input data
        filename=paste("PlotInput/Regression_",input$datasets_reg,".txt", sep="")
        data<-read.csv(filename,sep="\t",header=F)
        #produce the right labels
        #data$names<-paste(data$V1,data$V2,sep=" - ")
        ## Use densCols() output to get density at each point
        x <- densCols(data$V3,data$V4, colramp=colorRampPalette(c("black", "white")))
        data$dens <- col2rgb(x)[1,] + 1L
        
        #Filter data according to the selected methods
        matches <- grepl(paste(input$method,collapse="|"), data$V2)
        plottedData<-data[matches,]        
        
        plot_ly(y = plottedData$V4, 
                x = plottedData$V3,
                color=~plottedData$dens
        )%>%
          layout(title = paste('Regression with',input$method,sep=" "),
                 xaxis = list(
                   title = "Measured"),
                 yaxis = list(
                   title = "Predicted"
                 )
          )
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
    
    ####################################################################################
    # Plots for the run prediction tab
    
    output$comparePredicton<-renderPlotly({
      
      dummyData=data.frame(dataset=c("dataset1","dataset2","dataset3"),scores=c(0.8,0.7,0.9))
      plot_ly(
        x = dummyData$dataset,
        y = dummyData$scores,
        name = "Boxplot",
        type = "bar")%>%
        layout(title='Comparison of the current prediction with prediction on other datasets',
               xaxis = list(
                 title = "Data set"),
               yaxis = list(
                 title = "AUC score"
               )
        )
    })
    
#     score<-system("/home/sch/schmidka/anaconda3/bin/python  ~/Dokumente/GeneExpressionPrediction/neap/methods/classification_withStoredModel -i 
#            ~/Desktop/InputFiles/input_mRNA_normalized.txt -l ~/Desktop/InputFiles/testLabels_median.txt -m ~/Desktop/model.pkl -a -n")
  })
