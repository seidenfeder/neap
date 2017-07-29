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
                                         "'Gene Expression Prediction using Histone Modifications'. First we did a detailed analysis of factors, influencing the performance",
                                         "of our prediction models, shown in the tabs Model development, Regression plots, Spatial Information and Histone Modification.",
                                         "Therefore, we use datasets of three different cell lines from the ENCODE database ",
                                         "with 7 histone modifications and histone variants (H3K27ac, H3K27me3, H2AFZ, H3K4me3, H3K79me2, H3K9me3, H3K36me3).",
                                         "The two primary cell lines: the endothelial cell of umbilical vein (vein present during fetal development, connecting placenta and fetus)",
                                         "and the keratinocyte (cell type in the epidermis, the outer skin) as well as the immortalized cell line K562, ",
                                         "which is a leukemia cell. But because there are only very few data sets for which so many histone modifications were measured",
                                         "and we wanted also to compare a bigger set of data sets to evaluate differences between different cell types",
                                         "we used in the Data Comparison tab a smaller set of 4/5 histone modifications (? -> which?), but three additional data sets,",
                                         "two tissues, Gastrocnemius medialis (muscle of the lower leg) and thyroid gland, and a second immortalized cell line, SK-N-SH, a neuroblastoma cell line.",
                                         "For detailed information about the project,",
                                         "read our report. The most important results are presented shortly in the following tabs.")),
                                 h2("Model development"),
                                 p(paste("Here we tried to improve our performance by trying different label methods (only for the classification of course possible)",
                                         " and different normalization methods, ",
                                         "we also explored which bin the most important bin is.")),
                                 h2("Regression plots"),
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
                                         ")),
                                 h5("Data Formats:"),
                                 p("To run your prediction you need a binning file and if you use a classification also a label file as input. 
                                   These files follow the following format:"),
                                 strong("Binning File:"),
                                 div("##name of the dataset"),
                                 div("##used histone modifications"),
                                 div("#Genename1"),
                                 div("Tab separated scores for histone modification 1 for all bins"),
                                 div("Tab separated scores for histone modification 2 for all bins"),
                                 div("..."),
                                 div("#Genename2"),
                                 div("..."),
                                 strong("Labels File:"),
                                 div("## Additional Inforamtion"),
                                 div("Genename   Expression_Value    Class"),
                                 br()
                                 
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
                          plotlyOutput("regressionScatterplotZeros"),
                          div("This tab shows the predicted gen expression value for every single gene. Top the genes with a expression value of 0 were deleted.
                              Down we have the prediction for all genes also the ones with a expression value of 0. To get better inside where many points are plotted 
                              we used a density scatter plot. You can clearly see that many points are around a logarithmed expression value of 3")
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
                                                         "Support Vector Machine" = "SVM",
                                                         "Deep Learning" = "DL"),
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
                                        plotlyOutput("signalPattern"),
                                        br(),
                                        textOutput("signalPatternText")
                                        ),
                               tabPanel("Correlation Pattern",
                                        br(),
                                        plotlyOutput("corrPattern"),
                                        br(),
                                        textOutput("corrPatternText")  
                              )
                            ),
                            br(),
                            p(paste("When comparing the performance of the individual bins, it is also interesting to see",
                                    "how the general signal and correlation pattern looks like. Because comparing these pattern of",
                                    "different datasets can explain differences in the bin plot, especially for K562 and K562_short.",
                                    "The performance in the gene body is better for K562 than for K562_short, but the performance around at the TSS",
                                    "nearly identical. This can be explained, as in K562 there is additionally the histone modification H3K79me2,",
                                    "which has a strong signal in the gene body.")),
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
                                                   "Support Vector Machine" = "SVC",
                                                   "Deep Learning" = "DL"),
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
                                                         "Deep Learning" = "DL",
                                                         "Linear Regression" = "LR",
                                                         "RF Regression" = "RFR", 
                                                         "SVM Regression" = "SVR"),
                                             selected = "RFC")

                        ),
                        mainPanel(
                          plotlyOutput("histonePlot"),
                          p(paste("To test the influence of each histone modification on the complete performance, we run subsets of the data",
                                  "containing always only one or two histone modifications. The barplot above shows the results for one dataset and",
                                  "one method, where the performance results are ordered descending. It can be seen cleary that with only two histone",
                                  "modifications nearly the same performance than with the complete data set can be achieved. The performance of multiple",
                                  "pairs is nearly identical, so it is not possible to see if there is a or a few histone modifications which is clearly more",
                                  "important. Furthermore, the most important histone modification differ more between the methods than between the data sets.",
                                  "So the information between the histone modifications seems to be quite redundant.")),
                          tableOutput('histoneComparison'),
                          p(paste("The table shows the number of occurrences of each histone modification in the top x%",
                                  "of the run subsets of one and two histone modifications, when ordering them according to their performance",
                                  "(see barplot at the top of the line). The factor x is regulated via the slider on the right.",
                                  "All occurences of each histone modification are counted - in subsets with one or two histone modifications - ",
                                  "how often it occurs in subsets of one histone modification, is shown in brakes behind the first number.",
                                  "Different datasets and methods can be compared, shown in the header in the format <dataset> - <method>."))
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
                                                         "Thyroid gland"="thyroid gland",
                                                         "Merged big data set" = "BigData"),
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
                                   "a scaled data set, using all bins and all histone modifications, which were ",
                                   "in common for all data sets, and for classification the labeling method median.")
                          ),
                          plotlyOutput("dataMatrix"),
                          br(),
                          p(paste("The matrix above shows how the methods perform, if they were trained on one data set and tested",
                                  "on another data set. The performance in the diagonal, when training and predicting on the same",
                                  "data set, is clearly the best, of course. Off the diagonal, the performs varys between the methods clearly.",
                                  "The results for SVM stay good, which would indicate that the histone signal connected with gene expression",
                                  "is very similar across different cell types",
                                  "in human, while for Random Forest the performs drops significantly."))
                        )
                      )
                      
             ),
             tabPanel("Deep Learning",
                      sidebarLayout(
                        sidebarPanel(
                          checkboxGroupInput("datasets_DL", label="Data sets",
                                             choices = c("K562" = "K562", 
                                                         "Endothelial cell of umbilical vein" = "Endo",
                                                         "Keratinocyte"="Kera"),
                                             selected = c("K562")),
                          checkboxGroupInput("sets_DL", label="Shown curves",
                                             choices = c("Training set" = "train", 
                                                         "Test set" = "test"),
                                             selected = c("train")),
                          uiOutput("flexibelCheckDL")
                          ),
                          mainPanel(
                            tabsetPanel(id="DLtab",
                              tabPanel("Graph layout",
                                       br(),
                                       plotlyOutput("dl_Layout"),
                                       br(),
                                       p(paste("The layout of the graph influences the performance of the method and the learning curve.",
                                                "We tested three different layouts, each with a different number of convolution layers.",
                                                "The network with one convolution layer (with filter size 10 and 50 output channels) had",
                                                "also one maxpooling layer (with a maxpool parameter of 2), the network with 2 convolution layers",
                                                "(each filter size 10 and output channels 20 and 50) 2 maxpooling layers after each convolution layer (each with parameter 2)",
                                                "while the network with 5 convolution layers (each filter size 10 and output channels 20, 30, 40, 50 and 60) contained also",
                                                "after each convolution a maxpooling layer (each with parameter 2).",
                                                "General, no big differences can be observed between the layouts."))
                              ),
                              tabPanel("Learning rates",
                                       br(),
                                       plotlyOutput("dl_learningRates"),
                                       br(),
                                       p(paste("Of course, also the learning rate influences the learning curve. Choosing the graph layout with 2 convolution layers, we tested",
                                                "different learning rates. The rate of 0.05 is clearly to huge, as the AUC values of the training set become also much worse again,",
                                                "between the rate of 0.05 and 0.005 no big differences can be observed."))
                              )
                          )
                      )
                    )
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
                          uiOutput("inputFiles"),
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
                          p("In this tab you are able to run your own predictions. Just insert the data in the corresponding format. 
                            You find a description of the needed format under the project description.  "),
                          br(),
                          plotlyOutput("comparePredicton"),
                          br(),
                          textOutput("labelTextRun")

                        )
                      )#,
                      #tags$footer("My footer", align = "center", style = "
#               position:absolute;
#               bottom:0;
#               width:100%;
#               height:50px;   /* Height of the footer */
#               color: white;
#               padding: 10px;
#               background-color: black;
#               z-index: 1000;")
            )
  )
)
