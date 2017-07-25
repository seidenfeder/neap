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
    #Dynamical change for spatial information tap
    observe({
      if(input$type_spa == "c"){
        updateCheckboxGroupInput(session, "method_spa", label="Methods", 
                                 choices = c("Random Forest" = "RF", 
                                             "Support Vector Machine" = "SVM"),
                                 selected = "RF")
      }
      else{
        updateCheckboxGroupInput(session, "method_spa", label="Methods", 
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
    
    #Dynamically change avaliable methods (in the panel histone importance)
    observe({
      if(input$type_histone == "c"){
        updateRadioButtons(session, "method_histone", label="Methods", 
                                 choices = c("Random Forest" = "RFC", 
                                             "Support Vector Machine" = "SVC"),
                                 selected = "RFC")
      }
      else{
        updateRadioButtons(session, "method_histone", label="Methods", 
                                 choices = c("Linear Regression" = "LR",
                                             "RF Regression" = "RFR", 
                                             "SVM Regression" = "SVR"),
                                 selected = "RFR")
      }
    })
    
    #Dynamically change avaliable methods (in the panel histone importance)
    observe({
      if(input$type_2 == "c"){
        updateRadioButtons(session, "method_2_comp", label="Method shown in the comparison matrix",
                           choices = c("Random Forest" = "RF", 
                                       "Support Vector Machine" = "SVM"),
                           selected = "RF")
      }
      else{
        updateRadioButtons(session, "method_2_comp", label="Method shown in the comparison matrix",
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
                 ),
                 margin = list(b = 100, r=50)
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
                   title = "Bin",
                   tickvals = c(20,40,60,100,120,140),
                   ticktext = c("-20","TSS","+20","-20","TTS","+20")),
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
                 ),
                 margin = list(b = 100, r=50)
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
    
    output$regressionScatterplotZeros<-renderPlotly({
      if(! is.null(input$method_reg)){
        #Read input data
        filename=paste("PlotInput/Regression_",input$datasets_reg,"_zeros.txt", sep="")
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
    #Spatial Information Tab
    
    output$signalPattern<-renderPlotly({
      
      #Read input data
      filename=paste("PlotInput/signal",input$dataset_spatial,".txt", sep="")
      data<-read.csv(filename,sep="\t",header=F)
      data$V162<-NULL
      colnames(data)<-c("names",1:160)
      reshapedData<-melt(data, id=c("names"))
      normalized_p_Value<-reshapedData$value
      
      p<-ggplot(data = reshapedData, aes(x = reshapedData$variable, y = reshapedData$names)) +
        geom_tile(aes(fill = normalized_p_Value))+
        scale_fill_gradient2(low = "white", high = "steelblue")+
        ggtitle("Signal Pattern")+
        labs(x="Bins",y="Histone")+
        scale_x_discrete(breaks=c(20,41,60,100,121,140),
                         labels=c("-20","TSS", "+20","-20","TTS","+20"))
      
      ggplotly(p)%>%
        layout(margin = list(l = 110))
        
    })
    
    output$corrPattern<-renderPlotly({
      
      #Read input data
      filename=paste("PlotInput/corr",input$dataset_spatial,".txt", sep="")
      data<-read.csv(filename,sep="\t",header=F)
      data$V162<-NULL
      colnames(data)<-c("names",1:160)
      reshapedData<-melt(data, id=c("names"))
      Spearman<-reshapedData$value
      
      p<-ggplot(data = reshapedData, aes(x = reshapedData$variable, y = reshapedData$names)) +
        geom_tile(aes(fill = Spearman))+
        scale_fill_gradient2(low = "white",mid="yellow", high = "red")+
        ggtitle("Correlation Pattern")+
        labs(x="Bins",y="Histone")+
        scale_x_discrete(breaks=c(20,41,60,100,121,140),
                           labels=c("-20","TSS", "+20","-20","TTS","+20"))
      
      ggplotly(p)%>%
        layout(margin = list(l = 110))
      
    })
    
    output$binsPlot2<-renderPlotly({
      #Display the plot only for the classification task and if at least one method is selected
      if(!is.null(input$method_spa)&!is.null(input$dataset_spatial)){
        #Read input data
        if(input$type_spa=="c"){
          dataBinsC<-read.csv("PlotInput/evalBins.txt", sep="\t", header=F)
          yAxisTitle<-"AUC Score"
        }
        else{
          dataBinsC<-read.csv("PlotInput/evalBinsReg.txt", sep="\t", header=F)
          yAxisTitle<-"R2 Score"
        }
        
        #Filter data according to the selected methods
        matchesBins<- grepl(paste(input$method_spa,collapse="|"), dataBinsC$V2)
        plottedData<-dataBinsC[matchesBins,]
        
        #Filter data according to the selected cell lines
        matchesBinsCell<- grepl(paste(input$dataset_spatial,"$",collapse="|",sep=""), plottedData$V1)
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
                   title = "Bin",
                   tickvals = c(20,40,60,100,120,140),
                   ticktext = c("-20","TSS","+20","-20","TTS","+20")
                   ),
                 yaxis = list(
                   title = yAxisTitle
                 )
          )
      }
      else{
        return(NULL)
      }
    })
    
    ####################################################################################
    # Plots for the histone importance tab
    singleHistons <- reactive({
      data<-read.csv("PlotInput/histoneImportance_Single.txt",sep="\t",header=F)
      
      #Calculate the mean over the values of the cross-validation
      dataNew<-data.frame(dataset=data$V1,
                          method=data$V2,
                          histone=gsub("-human","",data$V4),
                          performanceMean=rowMeans(data[5:dim(data)[2]]),
                          type="Single")
      
      dataNew$type <- factor(dataNew$type, levels=c(levels(dataNew$type), 'All'))
      dataNew$type[dataNew$histone=='All']<-'All'
      
      return(dataNew)
    })

    pairsHistons <- reactive({
      data<-read.csv("PlotInput/histoneImportance_Pairs.txt",sep="\t",header=F)
      
      #Calculate the mean over the values of the cross-validation
      dataNew<-data.frame(dataset=data$V1,
                          method=data$V2,
                          histone=gsub("-human","",data$V4),
                          performanceMean=rowMeans(data[5:dim(data)[2]]),
                          type="Pair")
      return(dataNew)
    })
    
    output$histonePlot<-renderPlotly({
      
      dataSingle<-singleHistons()
      dataPairs<-pairsHistons()
      
      dataAll<-rbind(dataSingle, dataPairs)
      
      #Filter data according to the selected datasets
      dataAll<-dataAll[dataAll$dataset==input$dataset_histone,]
      
      #Filter data according to the selected methods
      dataAll<-dataAll[dataAll$method==input$method_histone,]
      
      #Sort according to the size
      dataAll<-dataAll[order(dataAll$performanceMean, decreasing=TRUE),]
      dataAll$histone <- factor(dataAll$histone, levels = dataAll$histone[order(dataAll$performanceMean, decreasing=TRUE)])
      
      #Show only the first data columns
      shownColumns<-round(nrow(dataAll)*input$perc_histone/100)
      dataAll<-dataAll[1:shownColumns,]
      
      plot_ly(y = dataAll$performanceMean, 
              x = dataAll$histone,
              color=dataAll$type,
              type="bar")%>%
        layout(title = paste('Performance of single or pairs of histone modifications'),
               xaxis = list(
                 title = "Histone modification(s)"),
               yaxis = list(
                 title = "AUC Score",
                 tickangle = 90
               ),
               margin=list(b=230)
        )
    })
    
    output$histoneComparison<-renderTable({
      x<-data.frame(a=c(1,2,3),b=c(4,5,6))
      
      dataSingle<-singleHistons()
      dataPairs<-pairsHistons()
      
      dataAll<-rbind(dataSingle, dataPairs)
      
      #Filter data according to the selected methods
      matches <- grepl(paste(input$methods_comp_histone,collapse="|"), dataAll$method)
      dataAll<-dataAll[matches,]
      
      #Filter data according to the selected datasets
      matches<- grepl(paste(input$datasets_comp_histone,"$",collapse="|",sep=""), dataAll$dataset)
      dataAll<-dataAll[matches,]
      
      #Fuer jede Histone Modifikation
      histons <- unique(dataAll$histone[dataAll$type=="Single" & dataAll$histone != "All"])
      
      #Loop ueber jede Gruppe (Datensatz und Methode)
      pairs<-expand.grid(input$datasets_comp_histone,input$methods_comp_histone)
      pairName<-paste(pairs[,1],pairs[,2],sep="-")
      
      results<-data.frame(histons=histons)
      for(i in 1:nrow(pairs)){
        #Filter dataset to the group
        dataShort<-dataAll[dataAll$dataset==as.character(pairs[i,1]) & dataAll$method==as.character(pairs[i,2]),]
        
        
        #Shown histone modifications
        dataShort<-dataShort[order(dataShort$performanceMean, decreasing=TRUE),]
        shownColumns<-round(nrow(dataShort)*input$perc_histone/100)
        dataShort<-dataShort[1:shownColumns,]
        
        #Occurence in general
        occAll<-sapply(histons, function(x) sum(grepl(x,dataShort$histone)))
        occSingle<-sapply(histons, function(x) sum(grepl(x,dataShort$histone[dataShort$type=="Single"])))
        occurences<-paste0(occAll," (",occSingle,")")
        results<-cbind(results,occurences)
      }
      colnames(results)<-c("Histone",pairName)
      
      results
    })
    
    ####################################################################################
    # Plots for the dataset comparison tab
    
    #Create plot for the different data sets
    output$cvData<-renderPlotly({
      if(! is.null(input$method_2)){
        
        #Read different files for classification and for regression
        #TODO: add right data files (after creating them ;) )
        if(input$type_2=="c"){
          filename = "PlotInput/performanceDatasets.txt"
          titleString = "AUC Score"
        }
        else{
          filename = "PlotInput/performanceDatasetsRegression.txt"
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
        #Read different files for classification and for regression
        if(input$type_2=="c"){
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
        matches <- grepl(paste(input$method_2_comp,collapse="|"), data$V1)
        plottedData<-data[matches,]
        
        #Filter data according to the selected data sets
        matches <- grepl(paste(input$datasets_2,collapse="|"), plottedData$V2)
        plottedData<-plottedData[matches,]
        
        matches <- grepl(paste(input$datasets_2,collapse="|"), plottedData$V3)
        plottedData<-plottedData[matches,]
        
        #Rename variables
        colnames(plottedData)<-c("Method","Dataset1","Dataset2","Score")
        #Create heatmap
        p<-ggplot(data = plottedData, aes(x = Dataset1, y = Dataset2)) +
          geom_tile(aes(fill = Score))+
          scale_fill_gradient(low = "white", high = "red")+
          ggtitle("Training on set 1, prediction set 2")+
          labs(x="Data set 1",y="Data set 2")
        
        ggplotly(p)

    })

    ####################################################################################
    # Plots for the deep learning tab
    
    output$dl_learningRates<-renderPlotly({
      data<-read.csv("PlotInput/deepLearning_learningrates.txt",sep="\t",header=F)
      
      plot_ly(y = as.numeric(data$V4),
              x = as.numeric(data$V3),
              color = paste(data$V1,data$V2,sep="-"),
              type="scatter",
              mode="lines")%>%
      layout(title = 'Influence of the learning rate',
               xaxis = list(
                 title = "Step"),
               yaxis = list(
                 title = "Auc score"
               )
        )
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
