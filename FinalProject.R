################################################################################
###################### Install ML and visualization packages ###################
################################################################################
install_missing_packages <- function(pkg){
  missing_package <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(missing_package))
    install.packages(missing_package, dependencies = TRUE, repos='http://cran.rstudio.com/')
  ignore <- sapply(pkg, require, character.only = TRUE) # Load the Library
}

packages =c("tidyverse", "caret", "skimr", "corrplot", "googledrive", "janitor", "naniar", 
            "data.validator", "CatEncoders", "randomForest", "xgboost")

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

data_file = "data_file.csv"
data_file_path = paste0(script_path(), .Platform$file.sep, data_file)

if (!file.exists(data_file_path)) {
  drive_deauth()
  folder_id = drive_get(as_id("https://drive.google.com/drive/folders/10ze7uyKYysH5tCfWb2gam-5SvkFS7iz2"))
  
  #find files in folder
  files = drive_ls(folder_id)
  
  #mkdir
  tmp_dir = paste0(script_path(), .Platform$file.sep, "tmp", .Platform$file.sep)
  unlink(tmp_dir, recursive = TRUE)
  dir.create(tmp_dir)
  
  #loop dirs and download files inside them
  for (i in seq_along(files$name)) {
    try({
      if (files$name[i] == "Mango_2023.csv") { ## Remove this check to load all data ~80,000 records
        drive_download(as_id(files$id[i]), path = str_c(tmp_dir,files$name[i]))
      }
    })
  }
  
  mango <- list.files(path=tmp_dir, full.names = TRUE) %>% 
    lapply(read_csv) %>% 
    bind_rows 
  
  write.csv(mango, file = data_file_path, row.names = FALSE)
  unlink(tmp_dir, recursive = TRUE)
} else {
  mango = read.csv(data_file_path)
}

################################################################################
######################## DATA LEARNING AND CLEANING ############################
################################################################################
dim(mango) # the shape of our dataset
head(mango) # first-k rows
print(skim(mango)) # view a pre-data-cleaning summary
gg_miss_var(mango, show_pct = TRUE) # Identify and address missing data

# Step 1: Some cleaning
mango <- mango %>% 
  remove_empty(which = c("rows", "cols")) %>% # Remove completely empty rows or columns
  tidyr::drop_na() # Missing data is < 0.1% so can drop the missing entries

# filter(variety != "Other") %>% # remove where variety is other

# Step 2: Identify and drop duplicates
get_dupes(mango) %>% head(5) # show duplicates
mango <- mango %>% distinct(.keep_all = TRUE) # remove duplicates

# Step 3: show columns with only one value
mango %>% purrr::keep(~length(unique(.x)) == 1) %>% head(5) # Show first 5 rows
mango <- mango %>% remove_constant() #remove columns with only constants

# Step 4: validate dates and numerical data
mango_validator <- function() {
  report <- data_validation_report()
  data.validator::validate(mango, name = "Mango Dataset") %>%
    validate_if(min_price >= 0, description = "We cant have negative Min Price") %>%
    validate_if(max_price >= 0, description = "We cant have negative Max Price") %>%
    validate_if(modal_price >= 0, description = "We cant have negative Modal Price") %>%
    validate_if(lubridate::is.Date(arrival_date), description = "Arrival date has all Date values") %>%
    add_results(report)
  
  print(report)
  
  # save_report(report, success = TRUE)
  # browseURL("validation_report.html")
}
# Check validations
mango_validator()

# VISUALISATIONS
ggplot(mango, aes(x=variety)) + geom_histogram(stat="count", colour="darkblue", fill="darkblue") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# one hot encoding 
cols_to_hold = c("arrival_date", "update_date", "min_price", "max_price", "modal_price")
non_encoded_cols <- mango %>% select(any_of(cols_to_hold))
to_encode <- mango %>% select(-any_of(cols_to_hold))
dmy <- dummyVars(~ ., data = to_encode)
encoded_cols <- data.frame(predict(dmy, newdata = to_encode))
mango <- cbind(non_encoded_cols, encoded_cols)

# Some mutations before computations: Encode character data into numeric and extract month
mango <- mango %>% 
  mutate(
    month = month(as.Date(arrival_date, format = "%d/%m/%Y")),
    day = day(as.Date(arrival_date, format = "%d/%m/%Y"))
  )

# Recheck validation
mango_validator()

# We get p-value matrix and confidence intervals matrix using cor.mtest()
# testRes = cor.mtest(mango%>%select_if(is.numeric), conf.level = 0.95)

# co-relation matrix
# corrplot(cor(mango%>%select_if(is.numeric)), method = "color", addCoef.col = "black", number.cex = 0.75,
#          p.mat = testRes$p, insig = 'blank') # shows min_price and max_price with correlation to our output variable we thus drop them

# view a after cleaning summary
cols_to_drop = c("arrival_date", "update_date", "min_price", "max_price")
mango <- mango %>% select(-any_of(cols_to_drop))

# How to deal with outliers
# Hampel Filter Median Absolute Deviations (MAD) / Percentiles
# lower_bound <- median(mango$modal_price) - 3 * mad(mango$modal_price, constant = 1)
# lower_bound
# 
# upper_bound <- median(mango$modal_price) + 3 * mad(mango$modal_price, constant = 1)
# upper_bound
# 
# outlier_ind <- which(mango$modal_price < lower_bound | mango$modal_price > upper_bound)
# length(outlier_ind)

################################################################################
############################### MODEL TRAINING #################################
################################################################################

# RF SVM XGBoost

# Use caret's createDataPartition() to split dataset into a training and testing sets
train_ind = createDataPartition(mango$modal_price, p = .7, list = F)
train = mango[train_ind, ]
test = mango[-train_ind, ]

set.seed(1)
tr_ctrl <- trainControl(
  method = "cv",  # Cross-validation
  number = 5,     # 5-fold cross-validation
)

# Random Forest
rf_model <- randomForest(modal_price ~ ., data = train,  trControl = tr_ctrl, ntree = 250)
rf_pred = predict(rf_model, newdata = test)

# SVM
# training a Support Vector Machine Regression model using svmLinear
svm_model = train(modal_price ~ ., data = train, method = "svmLinear", trControl = tr_ctrl)
print(svm_model)
svm_pred = predict(svm_model, newdata = test)

# XGBoost
tr_input_x <- as.matrix(select(train, -modal_price))
tr_input_y <- train$modal_price
ts_input_x <- as.matrix(select(test, -modal_price))
ts_input_y <- test$modal_price

d_train_data = xgb.DMatrix(data=tr_input_x, label = tr_input_y)
d_test_data = xgb.DMatrix(data=ts_input_x, label = ts_input_y)

watchlist = list(train=d_train_data, test=d_test_data)

# model = xgb.train(data = d_train_data, max.depth = 6, watchlist=watchlist, nrounds = 450)
# From the output, the RMSE just keeps dropping as the number of rounds is increased, so let's use MAE as the stopping criterion rather than RMSE.
# MAE is less sensitive to outliers compared to RMSE because MAE does not square the errors. If your dataset contains significant outliers that you do not want to heavily penalize, MAE might be a better choice.
# Will perform cross-validation to validate the results from the XGBoost model
# Specifying evaluation metrics and displaying training and testing data at each of the 100 rounds (using MAE)
# model <- xgb.train(data = xgb_train, max.depth = 3, watchlist = watchlist, nrounds = 100,
#                    eval_metric = "mae")  # Using MAE as the evaluation metric
#From the output we can see that the minimum testing MAE is achieved at the 72nd round. Beyond this point, the test MSE fluctuates instead of being always decreasing.
#To avoid overfitting the data, we will fit the final XGBoost model with 72 rounds.

xgb_model = xgboost(data=d_train_data, max.depth=6, nrounds = 100, verbose = FALSE)
xgb_pred <- predict(xgb_model, newdata = d_test_data)




grid <- expand.grid(mtry = c(2, 3, 4),        # Number of variables to sample as candidates at each split
                    ntree = c(500, 1000, 1500))

rf_tuned <- train(Species ~ .,                   # Formula for the model
                  data = data,                   # Training data
                  method = "rf",                 # Random Forest method
                  trControl = ctrl,              # Training control settings
                  tuneGrid = grid)


grid <- expand.grid(mtry = c(2, 3, 4),        # Number of variables to sample as candidates at each split
                    ntree = c(500, 1000, 1500))

rf_tuned <- train(modal_price ~ .,                   # Formula for the model
                  data = mango,                   # Training data
                  method = "rf",                 # Random Forest method
                  trControl = tr_ctrl,              # Training control settings
                  tuneGrid = grid)


# XGBoost
input_x <- as.matrix(select(train, -modal_price))
input_y <- train$modal_price

grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

train_control <- trainControl(
  method = "cv",
  number = 5, # with n folds 
  verboseIter = FALSE, # no training log
  allowParallel = FALSE # FALSE for reproducible results 
)

xgb_base <- caret::train(
  x = input_x,
  y = input_y,
  trControl = train_control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verbose = FALSE
)

xgb_base

# We start with nrounds from 200 as lower rounds gives very big errors
tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = 1000, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 5, # with n folds 
  verboseIter = FALSE, # no training log
  allowParallel = FALSE # FALSE for reproducible results 
)

xgb_tune <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = FALSE
)

tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}

tuneplot(xgb_tune)

# We get the best tuning parameters
xgb_tune$bestTune

# Final results
df_res = data.frame(
  model = c("Random Forest", "SVR", "XGBoost"),
  RMSE = c(
    RMSE(rf_pred, test$modal_price),
    RMSE(svm_pred, test$modal_price),
    RMSE(xgb_pred, test$modal_price)
  ),
  R2 = c(
    R2(rf_pred, test$modal_price),
    R2(svm_pred, test$modal_price),
    R2(xgb_pred, test$modal_price)
  ),
  MAE = c(
    MAE(rf_pred, test$modal_price),
    MAE(svm_pred, test$modal_price),
    MAE(xgb_pred, test$modal_price)
  )
)
df_res




# Save model to RDS file
saveRDS(xgb_tune, paste0(script_path(), .Platform$file.sep, "model.rds"))
