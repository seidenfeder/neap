library(shiny)
library(shinythemes)
library(plotly)


# Define UI for the application
shinyUI(
  navbarPage("Gene expression predidiction",theme = shinytheme("sandstone"),
             tabPanel("Project description",
                      fluidPage(
                        h1("NEAP Group 1 - Gene expression prediction using histone modifications"),
                        h4("Nicola Palandt, Katharina Schmid"),
                        h2("Section"),
                        p("Text ...")
                      )),
             tabPanel("Model development",
                      sidebarLayout(
                        sidebarPanel(
                          radioButtons("type", label="Machine learning task",
                                       c("Classification"="c", "Regression"="r")
                          ),
                          checkboxGroupInput("method", label="Methods", 
                                             choices = c("Random Forest" = "RF", 
                                                         "Support Vector Machine" = "SVM"),
                                             selected = "RF"),
                          checkboxGroupInput("datasets", label="Data sets", 
                                             choices = c("K562_short"="K562_short",
                                                         "K562" = "K562", 
                                                         "Endothelial cell of umbilical vein" = "Endo",
                                                         "Keratinocyte"="keratinocyte"),
                                             selected = "K562")
                        ),
                        mainPanel(
                          uiOutput("dynamic")
                        )
                      )
             ),
             tabPanel("Regression plots",
                      sidebarLayout(
                        sidebarPanel(
                          radioButtons("method_reg", label="Methods", 
                                             choices = c("Linear Regression" = "LR",
                                                         "RF Regression" = "RF", 
                                                         "SVM Regression" = "SVM"),
                                             selected = "RF"),
                          radioButtons("datasets_reg", label="Data sets",
                                       choices = c("K562_short"="K562_short",
                                                   "K562" = "K562", 
                                                   "Endothelial cell of umbilical vein" = "Endo",
                                                   "Keratinocyte"="keratinocyte"),
                                       selected = "K562")
                          
                        ),
                        mainPanel(
                          plotlyOutput("regressionScatterplot"),
                          br(),
                          plotlyOutput("regressionScatterplotZeros")
                        )
                      )
             ),
             tabPanel("Spatial information",
                      sidebarLayout(
                        sidebarPanel(
                          radioButtons("dataset_spatial", label="Data sets",
                                       choices = c("K562_short"="K562_short",
                                                   "K562" = "K562", 
                                                   "Endothelial cell of umbilical vein" = "Endo",
                                                   "Keratinocyte"="keratinocyte"),
                                       selected = "K562")
                          ),
                          mainPanel(
                            tabsetPanel(
                              tabPanel("Signal Pattern",
                                       br()
                                       plotlyOutput("signalPattern"),
                                       br()),
                              tabPanel("Correlation Pattern",
                                       br(),
                                       plotlyOutput("corrPattern")
                              )
                            )
                              
                            
                          )
                        )
             ),
             tabPanel("Histone modifications",
                      sidebarLayout(
                        sidebarPanel(
                          radioButtons("type_histone", label="Machine learning task",
                                       c("Classification"="c", "Regression"="r")
                          ),
                          radioButtons("method_histone", label="Methods", 
                                       choices = c("Random Forest" = "RFC", 
                                                   "Support Vector Machine" = "SVC"),
                                       selected = "RFC"),
                          radioButtons("dataset_histone", label="Data sets",
                                       choices = c("K562" = "K562", 
                                                   "Endothelial cell of umbilical vein" = "endo",
                                                   "Keratinocyte"="keratinocyte"),
                                       selected = "K562"),
                          sliderInput("perc_histone", label="Percentage of best histone modifications to compare", 
                                      0, 100, 10),
                          checkboxGroupInput("datasets_comp_histone", label="Data sets for comparison",
                                             choices = c("K562" = "K562", 
                                                         "Endothelial cell of umbilical vein" = "Endo",
                                                         "Keratinocyte"="keratinocyte"),
                                             selected = "K562"),
                          checkboxGroupInput("methods_comp_histone", label="Methods for comparison",
                                             choices = c("Random Forest" = "RFC", 
                                                         "Support Vector Machine" = "SVC",
                                                         "Linear Regression" = "LR",
                                                         "RF Regression" = "RFR", 
                                                         "SVM Regression" = "SVR"),
                                             selected = "K562")

                        ),
                        mainPanel(
                          plotlyOutput("histonePlot"),
                          tableOutput('histoneComparison')
                        )
                      )
             ),
             tabPanel("Dataset comparison",
                      sidebarLayout(
                        sidebarPanel(
                          radioButtons("type_2", label="Machine learning task",
                                       c("Classification"="c", "Regression"="r")),
                          checkboxGroupInput("method_2", label="Methods",
                                             choices = c("Random Forest" = "RF",
                                                         "Support Vector Machine" = "SVM"),
                                             selected = "RF"),
                          checkboxGroupInput("datasets_2", label="Data sets",
                                             choices = c("dataset1" = "dataset1", 
                                                         "dataset2" = "dataset2",
                                                         "dataset3" = "dataset3",
                                                         "K562" = "K562", 
                                                         "Endothelial cell of umbilical vein" = "Endo",
                                                         "Keratinocyte"="keratinocyte",
                                                         "Gastrocnemius medialis"="gastrocnemius medialis",
                                                         "SK-N-SH"="SK-N-SH",
                                                         "Thyroid gland"="thyroid gland"),
                                             selected = c("dataset1"))
                          
                        ),
                        mainPanel(
                          plotlyOutput("cvData"),
                          br(),
                          p(paste0("Choosing the optimal parameters, which were detected during the model development, ",
                                   "different data sets were tested using all possible classification and regression ",
                                   "methods in a 10-fold cross validation. The best determined parameters were thereby ",
                                   "a normalized data set, usisng all bins and all histone modifications, which were ",
                                   "in common for all data sets, and for classification the labeling method median/zero(?).")
                          ),
                          plotlyOutput("dataMatrix"),
                          br(),
                          p("Test ...")
                        )
                      )
                      
             ),
             tabPanel("Run prediction",
                      sidebarLayout(
                        sidebarPanel(
                          radioButtons("type_3", label="Machine learning task",
                                       c("Classification"="c", "Regression"="r")),
                          checkboxGroupInput("method_3", label="Methods",
                                             choices = c("Random Forest" = "RF",
                                                         "Support Vector Machine" = "SVM"),
                                             selected = "RF"),
                          radioButtons("datasetTrain", label="Training set",
                                             choices = c("dataset1" = "dataset1", 
                                                         "dataset2" = "dataset2",
                                                         "dataset3"="dataset3")),
                          fileInput("binningFile", label = "Feature file with bins"),
                          fileInput("labelFile", label = "Label file"),
                          actionButton("action", label = "Run prediction"),
                          checkboxGroupInput("datasets_3", label="Data sets to compare to",
                                             choices = c("dataset1" = "dataset1", 
                                                         "dataset2" = "dataset2",
                                                         "dataset3"="dataset3"),
                                             selected = c("dataset1"))                          
                          
                        ),
                        mainPanel(
                          p("In this tab you are able to run your own predictions. Just insert the data in the corresponding format. ... "),
                          plotlyOutput("comparePredicton")

                        )
                      )
            )
  )
)
