library(shiny)
library(plotly)


# Define UI for the application
shinyUI(
  navbarPage("Gene expression predidiction",
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
                                             choices = c("K562" = "K562", 
                                                         "Endothelial cell of umbilical vein" = "Endo",
                                                         "Keratinocyte"="keratinocyte"),
                                             selected = "K562")
                        ),
                        mainPanel(
                          plotlyOutput("labelPlot"),
                          br(),
                          textOutput("labelText"),
                          br(),
                          plotlyOutput("binsPlot"),
                          br(),
                          plotlyOutput("normPlot")
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
                                                         "dataset3"="dataset3"),
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
                          checkboxGroupInput("datasets_3", label="Training set",
                                             choices = c("dataset1" = "dataset1", 
                                                         "dataset2" = "dataset2",
                                                         "dataset3"="dataset3"),
                                             selected = c("dataset1"))
                          
                        ),
                        mainPanel(
                          p("In this tab you are able to run your own predictions. Just insert the data in the corresponding format. ... ")
                        )
                      )
            )
  )
)
