library(shiny)
library(shinythemes)
library(plotly)


# Define UI for the application
shinyUI(
  navbarPage("Gene expression predidiction",theme = shinytheme("sandstone"),
             tabPanel("Project description",
                      fluidPage(
                        fluidRow(
                          column(2,
                                 " "
                          ),
                          column(8,
                                 h1("NEAP Group 1 - Gene expression prediction using histone modifications"),
                                 h4("Nicola Palandt, Katharina Schmid"),
                                 h2("Introduction"),
                                 p(paste("On this website we present the interactive plots, created during our research project on ",
                                         "'Gene Expression Prediction using Histone Modifications'. We analysed three different cell lines from the ENCODE database ",
                                         "with a bigger set of histone modifications. The two primary cell lines: the endothelial cell of umbilical vein and the keratinocyte as well as the immortalized cell line K562, ",
                                         "which is a leukemia cell. For detailed information about the project,",
                                         "read our report. The most important results are presented shortly in the following tabs.")),
                                 h2("Model development"),
                                 p(paste("Here we tried to improve our performance by trying different label method and different normalization methods, ",
                                         "we also explored which bin the most important bin is.")),
                                 h2("Regression "),
                                 p(paste("In this tab you will find the results of the regresssion for different datasets, you will see two scatter plots ",
                                         "that show the measured and the predicted data once with the expression level zero and once without it.")),
                                 h2("Spatial Information"),
                                 p(paste("Here we show the normalized signal pattern of the histone modifications over the different bins, as well as the Spearman correlation ",
                                         "between the different bins and the gen expression values. We compare these pattern with the performance of each single bin.")),
                                 h2("Histone Modification"),
                                 p(paste("Different histone modifications play a differnent role for the prediction. In this tab we compare the importance ",
                                         "of the different histone modifications.")),
                                 h2("Dataset Comparision"),
                                 p(paste("We compared six different datasets with each other, two immortalized cell lines, two tissues and two primary cell lines. ",
                                         "We run all the datasets with our implemented methods and tested the model with the other datasets")),
                                 h2("Deep Learning"),
                                 p(paste("Deep Learning is a method that is very differnet from the other implemented ones, this tab is to show all our result that we got ",
                                         "from deep learning. Showing the learning curve and ....")),
                                 h2("Run Prediction"),
                                 p(paste("In this tab you are able to compare your own dataset with our datasets. You are able to predict the gene expression ",
                                         "by using our trained models and compare the performance with the performance of our models.
                                         "))
                                 
                          ),
                          column(2,
                                 " ")
                        )
                        
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
                          radioButtons("type_spa", label="Machine learning task",
                                       c("Classification"="c", "Regression"="r")
                          ),
                          checkboxGroupInput("method_spa", label="Methods", 
                                             choices = c("Random Forest" = "RF", 
                                                         "Support Vector Machine" = "SVM"),
                                             selected = "RF"),
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
                                        br(),
                                        plotlyOutput("signalPattern")
                                        ),
                               tabPanel("Correlation Pattern",
                                        br(),
                                        plotlyOutput("corrPattern")  
                              )
                            ),
                            br(),
                            br(),
                            plotlyOutput("binsPlot2")
                          )
                        )
             ),
             tabPanel("Histone modifications",
                      sidebarLayout(
                        sidebarPanel(
                          radioButtons("type_histone", label="Machine learning task",
                                       c("Classification"="c", "Regression"="r"),
                                       selected="c"
                          ),
                          radioButtons("method_histone", label="Methods", 
                                       choices = c("Random Forest" = "RFC", 
                                                   "Support Vector Machine" = "SVC"),
                                       selected = "RFC"),
                          radioButtons("dataset_histone", label="Data sets",
                                       choices = c("K562" = "K562", 
                                                   "Endothelial cell of umbilical vein" = "Endo",
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
                                             selected = "RFC")

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
                                             choices = c("K562" = "K562", 
                                                         "Endothelial cell of umbilical vein" = "Endo",
                                                         "Keratinocyte"="keratinocyte",
                                                         "Gastrocnemius medialis"="gastrocnemius medialis",
                                                         "SK-N-SH"="SK-N-SH",
                                                         "Thyroid gland"="thyroid gland"),
                                             selected = c("K562","Endo","keratinocyte")),
                          radioButtons("method_2_comp", label="Method shown in the comparison matrix",
                                       c("Random Forest" = "RF",
                                         "Support Vector Machine" = "SVM"))
                          
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
                          p("The matrix above ")
                        )
                      )
                      
             ),
             tabPanel("Deep Learning",
                      tabsetPanel(
                        tabPanel("Graph layout",
                                 br(),
                                 plotlyOutput("dl_Layout")
                        ),
                        tabPanel("Learning rates",
                                 br(),
                                 plotlyOutput("dl_learningRates")
                        ),
                        tabPanel("Data sets",
                                 br(),
                                 plotlyOutput("dl_datasets")
                        ),
                        tabPanel("Bin Importance",
                                 sidebarLayout(
                                   sidebarPanel(
                                     radioButtons("dataset_deep", label="Dataset",
                                                  choices = c("K562" = "K562", 
                                                              "Endothelial cell of umbilical vein" = "Endo",
                                                              "Keratinocyte"="keratinocyte"),
                                                  selected = "K562")
                                     ),
                                   mainPanel(
                                     br(),
                                     plotlyOutput("binImp")
                                     )
                                   )
                                 
                                 ))
             ),
             tabPanel("Run prediction",
                      sidebarLayout(
                        sidebarPanel(
                          radioButtons("type_3", label="Machine learning task",
                                       c("Classification"="c", "Regression"="r")),
                          radioButtons("method_3", label="Method",
                                             choices = c("Random Forest" = "RF",
                                                         "Support Vector Machine" = "SVM"),
                                             selected = "RF"),
                          radioButtons("datasetTrain", label="Training set",
                                             choices = c("K562" = "K562", 
                                                         "Endothelial cell of umbilical vein" = "Endo",
                                                         "Keratinocyte"="keratinocyte",
                                                         "Gastrocnemius medialis"="gastrocnemius medialis",
                                                         "SK-N-SH"="SK-N-SH",
                                                         "Thyroid gland"="thyroid gland"),
                                              selected="K562"),
                          fileInput("binningFile", label = "Feature file with bins"),
                          fileInput("labelFile", label = "Label file"),
                          textInput("pythonPath", "Python path", "python"),
                          actionButton("action", label = "Run prediction"),
                          br(),
                          checkboxGroupInput("datasets_3", label="Data sets to compare to",
                                             choices = c("K562" = "K562", 
                                                         "Endothelial cell of umbilical vein" = "Endo",
                                                         "Keratinocyte"="keratinocyte",
                                                         "Gastrocnemius medialis"="gastrocnemius medialis",
                                                         "SK-N-SH"="SK-N-SH",
                                                         "Thyroid gland"="thyroid gland"),
                                             selected = c("K562","Endo","keratinocyte"))                          
                          
                        ),
                        mainPanel(
                          p("In this tab you are able to run your own predictions. Just insert the data in the corresponding format. ... "),
                          plotlyOutput("comparePredicton")

                        )
                      )
            )
  )
)
