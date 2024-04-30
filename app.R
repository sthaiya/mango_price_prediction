install_missing_packages <- function(pkg){
  missing_package <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(missing_package))
    install.packages(missing_package, dependencies = TRUE, repos='http://cran.rstudio.com/')
  ignore <- sapply(pkg, require, character.only = TRUE) # Load the Library
}

packages = c("tidyverse", "shiny", "shinythemes", "plotly")

# Respond with "Yes" if prompted on the console
install_missing_packages(packages)

################################################################################
############################## LOAD THE DATASET ################################
################################################################################
script_path <- function(){
  this_file = gsub("--file=", "", commandArgs()[grepl("--file", commandArgs())])
  ifelse (length(this_file) > 0, 
          paste(head(strsplit(this_file, '[/|\\]')[[1]], -1), collapse = .Platform$file.sep),
          dirname(rstudioapi::getSourceEditorContext()$path)
  )
}

model_path = paste0(script_path(), .Platform$file.sep, "rf_model.rds")
meta_data_path = paste0(script_path(), .Platform$file.sep, "meta_data.csv")

if (file.exists(model_path) && file.exists(meta_data_path)) {
  model <- readRDS(model_path) # Read in the model
  meta_data = read.csv(meta_data_path) # Read in the meta data
} else {
  stop("Requisite files do not exist. Run the project script first")
}

states <- meta_data %>% filter(type == "state") %>% with(., split(value, key))
districts <- meta_data %>% filter(type == "district") %>% with(., split(value, key))
markets <- meta_data %>% filter(type == "market") %>% with(., split(value, key))
varieties <- meta_data %>% filter(type == "variety") %>% with(., split(value, key))
          
################################################################################
############################### USER INTERFACE #################################
################################################################################
ui <- fluidPage(theme = shinytheme("journal"),
                
    # Page header
    headerPanel('Mango Price Prediction'),
    
    # Input values
    sidebarPanel(
      selectInput("state", label = "State:", choices = states),
      selectInput("district", label = "District:", choices = districts),
      selectInput("market", label = "Market:", choices = markets),
      selectInput("variety", label = "Variety:", choices = varieties),
      dateInput('arrival_date', label = 'Arrival Date', value = Sys.Date()),
      actionButton("btnSubmit", "Run Prediction", class = "btn btn-primary")
    ),
    
    mainPanel(
      tags$label(h3('Results Area')),
      verbatimTextOutput('contents'),
      br(),
      tabsetPanel(
        id = 'tabs',
        tabPanel(title =  "Monthly Trend", plotlyOutput("plot")),
        tabPanel(title =  "Monthly Stats", tableOutput('summary'))
      )
    )
)

################################################################################
############################### SERVER CODE ####################################
################################################################################
server <- function(input, output) {
  
  userInput <- eventReactive(input$btnSubmit, {  
    df <- data.frame(
      input_key = c("state", "district", "market", "variety", "modal_price"),
      input_value = as.numeric(c(input$state, input$district, input$market,
                             input$variety, 0)))
    
    # transpose the inputs
    df <- setNames(data.frame(t(df[,-1])), df[,1])
    
    # create month data
    arrival_date = as.Date(input$arrival_date, format = "%Y-%m-%d")
    sel_day <- day(arrival_date)
    sel_month <- month(arrival_date)
    start_month <- ymd(format(arrival_date, "%Y-%m-01"))
    n <- days_in_month(arrival_date)
    df <- df[rep(seq_len(nrow(df)), n), ]
    
    df <- df %>%
      mutate(arr_date = seq(from=start_month, by = 'day', length.out = n)) %>%
      mutate(month = month(arr_date), day = day(arr_date)) %>%
      select(-arr_date)
    
    pred_df <- predict(model, df)
    pred_msg <- paste0("Predicted Price for date '", arrival_date, "' is ", round(pred_df[sel_day], 3))
    df <- df %>% mutate(pred_price = pred_df) %>% select(c('day', 'pred_price'))
    
    combo <- list(results = df, msg = pred_msg, sel_day = sel_day, sel_month = sel_month)
    combo
  })
  
  # Status Box
  output$contents <- renderText({
    if (input$btnSubmit > 0) {
      showTab(inputId = "tabs", target = "Monthly Trend", select = TRUE, session = getDefaultReactiveDomain())
      showTab(inputId = "tabs", target = "Monthly Stats", select = FALSE, session = getDefaultReactiveDomain())
      isolate(userInput()$msg)
    } else {
      hideTab("tabs", target = "Monthly Trend", session = getDefaultReactiveDomain())
      hideTab("tabs", target = "Monthly Stats", session = getDefaultReactiveDomain())
      return("Status: We are ready for Predictions")
    }
  })
  
  # render plot
  output$plot <- renderPlotly({
    if (input$btnSubmit > 0) {
      month <- format(as.Date(input$arrival_date), format = "%B")
      
      ggplot(userInput()$results, aes(x = day, y=pred_price)) +
        geom_line(col = "red") +
        geom_point(col = "red", shape = 1, size = 2) +
        geom_vline(xintercept = userInput()$sel_day, col = "DarkGray") +
        labs(title=paste(month, " Predictions"), x ="Day of Month", y = "Predicted Price") +
        theme_minimal()
    }
  })
  
  # render plot
  output$summary <- renderTable({
    res <- userInput()$results
    df <- data.frame(res)
    isolate(summary(df))
  })
}

################################################################################
############################### CREATE THE SHINY APP ###########################
################################################################################
shinyApp(ui = ui, server = server)
