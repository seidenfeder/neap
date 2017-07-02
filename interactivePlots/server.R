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
    
    
  
})