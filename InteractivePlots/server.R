library(shiny)
library(plotly)
#Fuer Melt
library(reshape2)
library(ggplot2)

options(warn =-1, shiny.maxRequestSize=1000*1024^2)

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

    #Dynamically change avaliable methods (in the panel histone importance)
    observe({
      if(input$type_3 == "c"){
        updateRadioButtons(session, "method_3", label="Method",
                           choices = c("Random Forest" = "RF", 
                                       "Support Vector Machine" = "SVM"),
                           selected = "RF")
      }
      else{
        updateRadioButtons(session, "method_3", label="Method",
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
                   plotlyOutput("binsPlot"),
                   br(),
                   textOutput("binsText")
          ),
          tabPanel("Normalization",
                   br(),
                   plotlyOutput("normPlot"),
                   br(),
                   textOutput("normText")
          )
        )
      } else {
        tabsetPanel(
          tabPanel("Bin importance per bin",
                   br(),
                   plotlyOutput("binsPlot"),
                   br(),
                   textOutput("binsText")
          ),
          tabPanel("Normalization",
                   br(),
                   plotlyOutput("normPlot"),
                   br(),
                   textOutput("normText")
          )
        )
      }
      })
    
    output$inputFiles<-renderUI({
      if (input$type_3 == "c") {
        fileInput("labelFile", label = "Label file")
      } else {
        NULL
      }
    })
    
    output$flexibelSetOptionsDL<-renderUI({
      if (input$DLtab == "Learning rates" | input$DLtab == "Graph layout") {
        checkboxGroupInput("sets_DL", label="Shown curves",
                           choices = c("Training set" = "train", 
                                       "Test set" = "test"),
                           selected = c("train"))
      } else {
        NULL
      }
      
    })
    
    output$flexibelCheckDL<-renderUI({
      if (input$DLtab == "Learning rates") {
        checkboxGroupInput("learnrate_DL", label="Learning rates",
                           choices = c("0.05", 
                                       "0.005",
                                       "0.0005"),
                           selected = c("0.005"))
      } else if (input$DLtab == "Graph layout") {
        checkboxGroupInput("graph_DL", label="Graph layout",
                           choices = c("1 Convolution layer" = "C1", 
                                       "2 Convolution layers" = "C2",
                                       "5 Convolution layers" = "C5"),
                           selected = c("C2"))
      } else {
        NULL
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
                 margin = list(b = 100, r=50, t=30)
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
                     "seems to work best, independent of the chosen data set and machine learning method."))
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
                 ),
                 margin = list(t=30)
          )
      }
      else{
        return(NULL)
      }
    })
    
    #Display an explanation text with the bin plot
    output$binsText<-renderText({
      if(! is.null(input$method)&!is.null(input$datasets)){
        return(paste("Running the method always at only one bin, different performances can be observed.",
                     "So the histone signal corresponding to the gene expression level is not always the same in the region around the TSS,",
                     "the gene body and the TTS, but shows always the strongest signal at the region of the TSS.",
                     "Still, differences between the data sets can be observed. For example, the dataset K562 contains more histone modifications than",
                     "K562_short and with H4K79me2 also one additional with a strong correlation to the expression in the gene body.",
                     "So in this dataset the performance of the bins in the gene body are better and there is not so a huge drop after the TSS."))
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
                 margin = list(b = 100, r=50, t=30)
          )
      }
      else{
        return(NULL)
      }
    })
    
  
    output$normText<-renderText({
      if(! is.null(input$method)&!is.null(input$datasets)){
        return(paste("Also the normalization influences the performance of the data. We compared the performance ",
                     "after no normalization, scaling of the data and normalization of the data.",
                     "The scaling ...Say what the scaling does."))
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
        scale_fill_gradient2(low = "white",mid="yellow", high = "red", midpoint=0.0)+
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
        scale_fill_gradient2(low = "white",mid="yellow", high = "red", midpoint=0.0)+
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
          filename = "PlotInput/dataMatrixReg.txt"
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
        colnames(plottedData)<-c("Method","Trainset","Testset","Score")
        #Create heatmap
        p<-ggplot(data = plottedData, aes(x = Trainset, y = Testset)) +
          geom_tile(aes(fill = Score))+
          scale_fill_gradient2(low = "white",mid="yellow", high = "red",midpoint=0.5, limits=c(0.0,1.0))+
          ggtitle("Training on set 1, prediction set 2")+
          labs(x="Training Set",y="Test Set")

        
        ggplotly(p)

    })

    ####################################################################################
    # Plots for the deep learning tab

    output$dl_Layout<-renderPlotly({
      
      data<-read.csv("PlotInput/deepLearning_graphLayout.txt",sep="\t",header=F)
      
      #Filter data set
      matches <- grepl(paste(input$datasets_DL,collapse="|"), data$V1)
      plottedData<-data[matches,]
      
      #Filter train - test set
      matches <- grepl(paste(input$sets_DL,collapse="|"), plottedData$V3)
      plottedData<-plottedData[matches,]
      
      #Filter the learning rate
      matches <- grepl(paste0(input$graph_DL,collapse="|"), plottedData$V2)
      plottedData<-plottedData[matches,]
      
      plot_ly(y = as.numeric(plottedData$V5),
              x = as.numeric(plottedData$V4),
              color = paste(plottedData$V1,plottedData$V2,plottedData$V3,sep="-"),
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
    
    
    output$dl_learningRates<-renderPlotly({
      
      data<-read.csv("PlotInput/deepLearning_learningrates.txt",sep="\t",header=F)
      
      #Filter data set
      matches <- grepl(paste(input$datasets_DL,collapse="|"), data$V1)
      plottedData<-data[matches,]
      
      #Filter train - test set
      matches <- grepl(paste(input$sets_DL,collapse="|"), plottedData$V3)
      plottedData<-plottedData[matches,]
      
      #Display for digits of the numbers without exponentaion notation
      options(scipen=4)
      
      #Filter the learning rate
      matches <- grepl(paste0(input$learnrate_DL,collapse="|"), plottedData$V2)
      plottedData<-plottedData[matches,]
      
      plot_ly(y = as.numeric(plottedData$V5),
              x = as.numeric(plottedData$V4),
              color = paste(plottedData$V1,plottedData$V2,plottedData$V3,sep="-"),
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
    
    output$binImp<-renderPlotly({
      data<-read.csv("PlotInput/deepLearningBins.txt",sep="\t",header=F)
      matches <- grepl(paste(input$datasets_DL,collapse="|"), data$V1)
      plottedData<-data[matches,]
      
      #Create interactive line plots
      color1<-c("blue","red")
      plot_ly(y = plottedData$V3,
              x = plottedData$V2, type="scatter", 
              color=plottedData$V1,
              colors = color1,
              mode="lines")%>%
        layout(title = paste('Performance for each bin'),
               xaxis = list(
                 title = "Bin",
                 tickvals = c(20,40,60,100,120,140),
                 ticktext = c("-20","TSS","+20","-20","TTS","+20")
               ),
               yaxis = list(
                 title = "AUC Score"
               )
        )
      
      
    })
    
    ####################################################################################
    # Plots for the run prediction tab
    
    #Calculate the auc score for the loaded data
    calculateScore <- eventReactive(input$action, {

      #Check if all necessary files are uploaded
      if (input$type_3 == "c" & (is.null(input$binningFile) | is.null(input$labelFile))){
        showModal(modalDialog(
          title = "Error!",
          "Please upload a feature and a lable file!"
        ))
        
        return(NULL)
      }
      
      if (input$type_3 == "r" & is.null(input$binningFile)){
        showModal(modalDialog(
          title = "Error!",
          "Please upload a feature file!"
        ))
        
        return(NULL)
      }
      
      #Chose the right dataset
      if(input$datasetTrain == "gastrocnemius medialis"){
        datasetName<-"gastrocnemius_medialis"
      }
      else if(input$datasetTrain == "thyroid gland"){
        datasetName<-"thyroid_gland"
      }
      else{
        datasetName<-input$datasetTrain
      }
      model<-paste0("PredictionModels/model_",input$type_3,input$method_3,"_",datasetName,".pkl")


      #Choose the right script - either classification or regression
      if(input$type_3=="c"){
        systemCommand<-paste(input$pythonPath,"PredictionModels/classification_Interactive.py",
                             "-i",input$binningFile$datapath,"-l", input$labelFile$datapath,"-m", model,"-a -n")
      }
      else{
        systemCommand<-paste(input$pythonPath,"PredictionModels/regression_Interactive.py",
                             "-i",input$binningFile$datapath,"-m", model,"-a -n")
      }

      score<-withProgress(message="Prediction",value=0.5,expr={system(systemCommand, intern=T)})
      return(as.numeric(score))
    })
    
    output$comparePredicton<-renderPlotly({
      
      newScore<-calculateScore()

      if(is.numeric(newScore)){
        
        #Read different files for classification and for regression
        if(input$type_3=="c"){
          filename = "PlotInput/dataMatrix.txt"
          titleString = "AUC Score"
        }
        else{
          filename = "PlotInput/dataMatrixReg.txt"
          titleString = "R2 Score"
        }
        
        #Read input data
        data<-read.csv(filename,sep="\t",header=F)
        
        #Filter data according to the selected methods
        matches <- grepl(paste(input$method_3,collapse="|"), data$V1)
        plottedData<-data[matches,]
        
        #Filter data according to the trained data
        matches <- grepl(paste(input$datasetTrain,collapse="|"), plottedData$V2)
        plottedData<-plottedData[matches,]
        
        #Filter data according to the other data
        matches <- grepl(paste(input$datasets_3,collapse="|"), plottedData$V3)
        plottedData<-plottedData[matches,]
        
        #Add the new prediction
        plottedData$V3 <- factor(plottedData$V3 , levels=c(levels(plottedData$V3), 'Your data'))
        plottedData<-rbind(c(input$method_3, input$datasetTrain, "Your data", newScore),plottedData)
        
        #Set levels of plotted Data new to get a right scaling of the axis
        plottedData<-droplevels(plottedData)
        
        #Add colors
        plottedData$V5<-"oldData"
        plottedData$V5[plottedData$V3=='Your data']<-"newdata"
        
        plot_ly(
          x = plottedData$V3,
          y = as.numeric(plottedData$V4),
          color = plottedData$V5,
          name = "Boxplot",
          type = "bar")%>%
          layout(title='Comparison of the current prediction with prediction on other datasets',
                 xaxis = list(
                   title = "Data set"),
                 yaxis = list(
                   title = titleString
                 ),
                 margin(l=100),
                 showlegend = FALSE
          )
      }
      else{
        NULL
      }
    })
    

  })
