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
                                                         "..."=3),
                                             selected = "K562")
                        ),
                        mainPanel(
                          plotlyOutput("labelPlot"),
                          br(),
                          textOutput("labelText"),
                          br(),
                          plotlyOutput("binsPlot")
                        )
                      )
             ),
             tabPanel("Dataset comparison",
                      sidebarLayout(
                        sidebarPanel(
                          radioButtons("type_2", label="Machine learning task",
                                       c("Classification"="c", "Regression"="r")
                          ),
                          checkboxGroupInput("method_2", label="Methods", 
                                             choices = c("Random Forest" = "RF", 
                                                         "Support Vector Machine" = "SVM"),
                                             selected = "RF"),
                          checkboxGroupInput("datasets", label="Data sets", 
                                             choices = c("K562" = "K562", 
                                                         "Endothelial cell of umbilical vein" = "Endo",
                                                         "..."=3),
                                             selected = "K562")
                        ),
                        mainPanel(
                          plotOutput("plot")
                        )
                      )),
             tabPanel("Run prediction")
  )
)