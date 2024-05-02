################################################################################
###################### Install ML and visualization packages ###################
################################################################################
install_missing_packages <- function(pkg){
  missing_package <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(missing_package))
    install.packages(missing_package, dependencies = TRUE, repos='http://cran.rstudio.com/')
  ignore <- sapply(pkg, require, character.only = TRUE) # Load the Library
}

packages =c("tidyverse", "caret", "skimr", "corrplot", "googledrive", "janitor", 
            "naniar", "pROC", "dplyr", "data.table", "data.validator", 
            "randomForest", "xgboost", "ggplot2")

# Respond with "Yes" if prompted on the console
install_missing_packages(packages)

############################## LOAD THE DATASET ################################
mango = read.csv(url("https://raw.githubusercontent.com/sthaiya/mango_price_prediction/main/Mango_2023.csv"))

################################################################################
######################## DATA LEARNING AND CLEANING ############################
################################################################################
dim(mango) # the shape of our dataset
head(mango) # first-k rows
print(skim(mango)) # view a pre-data-cleaning summary

#Exclude the "min_price" and the "max_price" from the dataset
mango<-mango %>% select(-4,-7,-8,-10)
sum(is.na(mango))
#No missing values in any relevant column/variable. (Pairwise deletion applied in lieu of listwise deletion)

#Bivariate Analysis - One-Way ANOVA tests; Computed the MAE, MSE, RMSE, R-Squared on the  
#GeneratedVariance Importance Plot for the XGBoost Model and the Random Forest Model
#Only if you find including Bivariable Analysis in the report is necessary

#Perform Bivariable Analysis: conduct one-way anova between each predictor and the outcome "modal_price"
#Between "state" and "modal_price"
anova_state<-aov(modal_price ~ state, data=mango)
summary(anova_state)
#Between "district" and "modal_price"
anova_district<-aov(modal_price ~ district, data=mango)
summary(anova_district)
#Between "market" and "modal_price"
anova_market<-aov(modal_price ~ market, data=mango)
summary(anova_market)
#Between "variety" and "modal_price"
anova_variety<-aov(modal_price ~ variety, data=mango)
summary(anova_variety)
#Between "arrival_date" and "modal_price"
anova_arrival_date<-aov(modal_price ~ arrival_date, data=mango)
summary(anova_arrival_date)
#Significant: State, District, Market, Variety, Arrival_Date
#A statistically significant difference in mean modal price of mangoes across the states, districts, between the markets, varieties, and arrival dates.


#View the structure of the data
str(mango)
summary(mango)

set.seed(2)

# Change the data type of the "arrival_date" column in the mango dataset
mango$arrival_date<-as.Date(mango$arrival_date)

# Create dummy variables
# Specify the columns you want to dummy encode
dummy <- dummyVars(" ~ state + district + market + variety", data=mango)

# use one hot encoding for the categorical variables
new_mango <- data.frame(predict(dummy, newdata = mango))

new_mango$modal_price<-mango$modal_price
new_mango$arrival_date<-mango$arrival_date

################################################################################
############################### MODEL TRAINING #################################
################################################################################

# Models to test: RF, SVM and XGBoost
#Split the data into training (70%) and testing (30%)
parts = createDataPartition(new_mango$modal_price, p = .7, list = F)
train = new_mango[parts, ]
test = new_mango[-parts, ]

tr_ctrl <- trainControl(
  method = "cv",  # Cross-validation
  number = 5,     # 5-fold cross-validation
)

# Random Forest
rf_model <- randomForest(modal_price ~ ., data = train,  trControl = tr_ctrl, ntree = 250, mtry=3)
rf_pred = predict(rf_model, newdata = test)

# SVR through Caret
svm_model = train(modal_price ~ ., data = train, method = "svmRadial", trControl = tr_ctrl)
svm_pred = predict(svm_model, newdata = test)

## Fitting XGBoost Model
# Define predictors and response variable in the training set
train_x = data.matrix(train[, !names(train) %in% "modal_price"])
train_y = train[,"modal_price"]

# Define predictors and response variable in the testing set
test_x = data.matrix(test[, !names(test) %in% "modal_price"])
test_y = test[,"modal_price"]

# Define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

# Define watchlist
watchlist = list(train=xgb_train, test=xgb_test)

#Fitting the XGBoost model with the default parameters for model selection, first attempt (trial) fix nrounds = 10
xgb_model = xgboost(data = xgb_train, nrounds = 100, verbose = 0)
xgb_pred <- predict(xgb_model, xgb_test)

# Test results
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


# Tuning the XGBoost Model
#Perform the cross-validations based on minimized RMSE
# Define the parameters
params <- list(booster = "gbtree", objective = "reg:squarederror", eta=0.3, gamma=0, subsample=1, colsample_bytree=1)

# To perform a grid search
# Define a grid of nrounds and max_depth values to try
#nrounds_values <- c(1, 5, 10, 15, 20, 25)
#max_depth_values <- c(1, 2, 3, 4, 5, 6)

#Testing nrounds = 1, max_depth = 1
cv_1<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 1, max_depth = 1, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 5, max_depth = 1
cv_2<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 5, max_depth = 1, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 10, max_depth = 1
cv_3<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 10, max_depth = 1, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 15, max_depth = 1
cv_4<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 15, max_depth = 1, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 20, max_depth = 1
cv_5<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 20, max_depth = 1, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 25, max_depth = 1
cv_6<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 25, max_depth = 1, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 1, max_depth = 2
cv_7<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 1, max_depth = 2, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 5, max_depth = 2
cv_8<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 5, max_depth = 2, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 10, max_depth = 2
cv_9<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 10, max_depth = 2, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 15, max_depth = 2
cv_10<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 15, max_depth = 2, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 20, max_depth = 2
cv_11<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 20, max_depth = 2, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 25, max_depth = 2
cv_12<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 25, max_depth = 2, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 1, max_depth = 3
cv_13<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 1, max_depth = 3, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 5, max_depth = 3
cv_14<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 5, max_depth = 3, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 10, max_depth = 3
cv_15<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 10, max_depth = 3, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 15, max_depth = 3
cv_16<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 15, max_depth = 3, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 20, max_depth = 3
cv_17<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 20, max_depth = 3, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 25, max_depth = 3
cv_18<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 25, max_depth = 3, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 1, max_depth = 4
cv_19<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 1, max_depth = 4, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 5, max_depth = 4
cv_20<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 5, max_depth = 4, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 10, max_depth = 4
cv_21<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 10, max_depth = 4, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 15, max_depth = 4
cv_22<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 15, max_depth = 4, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 20, max_depth = 4
cv_23<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 20, max_depth = 4, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 25, max_depth = 4
cv_24<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 25, max_depth = 4, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 1, max_depth = 5
cv_25<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 1, max_depth = 5, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 5, max_depth = 5
cv_26<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 5, max_depth = 5, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 10, max_depth = 5
cv_27<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 10, max_depth = 5, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 15, max_depth = 5
cv_28<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 15, max_depth = 5, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 20, max_depth = 5
cv_29<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 20, max_depth = 5, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 25, max_depth = 5; unable to run due to high RAM
cv_30<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 25, max_depth = 5, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 1, max_depth = 6
cv_31<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 1, max_depth = 6, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 5, max_depth = 6
cv_32<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 5, max_depth = 6, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 10, max_depth = 6
cv_33<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 10, max_depth = 6, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 15, max_depth = 6
cv_34<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 15, max_depth = 6, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 20, max_depth = 6
cv_35<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 20, max_depth = 6, objective = "reg:squarederror", metrics = list("rmse","mae"))

#Testing nrounds = 25, max_depth = 6
cv_36<-xgb.cv(data = xgb_train, nfold = 5, nrounds = 25, max_depth = 6, objective = "reg:squarederror", metrics = list("rmse","mae"))

# Create a list of all cv objects
cv_list <- list(cv_1, cv_2, cv_3, cv_4, cv_5, cv_6, cv_7, cv_8, cv_9, cv_10, cv_11, cv_12, cv_13, cv_14, cv_15, cv_16, cv_17, cv_18, cv_19, cv_20, cv_21, cv_22, cv_23, cv_24, cv_25, cv_26, cv_27, cv_28, cv_29, cv_30, cv_31, cv_32, cv_33, cv_34, cv_35, cv_36)

# Initialize variables to store the minimum RMSE and MAE
min_rmse <- Inf
min_mae <- Inf

# Loop over the list and update the minimum RMSE and MAE
for (i in seq_along(cv_list)) {
  cv <- cv_list[[i]]
  if (min(cv$evaluation_log$test_rmse_mean) < min_rmse) {
    min_rmse <- min(cv$evaluation_log$test_rmse_mean)
    min_rmse_cv <- paste("cv_", i, sep = "")
  }
  if (min(cv$evaluation_log$test_mae_mean) < min_mae) {
    min_mae <- min(cv$evaluation_log$test_mae_mean)
    min_mae_cv <- paste("cv_", i, sep = "")
  }
}

# Print the cv_ that gives the lowest RMSE and MAE
print(paste("The model with the lowest RMSE is: ", min_rmse_cv))
print(paste("The model with the lowest MAE is: ", min_mae_cv))


#The model with the lowest RMSE was "cv_22"; the model with the lowest MAE was "cv_36".
#Try fitting an XGBoost Model with nrounds=15, max_depth=4 ("cv_22") and nrounds=25, max_depth=6 ("cv_36")
final_22 = xgboost(data = xgb_train, nrounds = 15, max_depth = 4, verbose = 0)
final_36 = xgboost(data = xgb_train, nrounds = 25, max_depth = 6, verbose = 0)


# Generate predictions on the test data with two potential best XGBoost models
pred_y_22 <- predict(final_22, xgb_test)  # Make predictions on test data
pred_y_36 <- predict(final_36, xgb_test)

#To calculate the testing MSE, MAE, RMSE, and R-Squared associated with the XGBoost prediction model with the lowest RMSE ("cv_22")
mean((test_y - pred_y_22)^2) #mse
MAE(test_y, pred_y_22) #mae
RMSE(test_y, pred_y_22) #rmse
R2(pred_y_22, test_y) #R-Squared

#To calculate the testing MSE, MAE, RMSE, and R-Squared associated with the XGBoost prediction model with the lowest MAE ("cv_36")
mean((test_y - pred_y_36)^2) #mse
MAE(test_y, pred_y_36) #mae
RMSE(test_y, pred_y_36) #rmse
R2(pred_y_36, test_y) #R-Squared

#To calculate the testing MSE, MAE, RMSE, and R-Squared associated with the XGBoost prediction model with the lowest RMSE ("cv_22")
#  > mean((test_y - pred_y_22)^2) #mse
#[1] 38572180 ***
#> MAE(test_y, pred_y_22) #mae
#[1] 1613.641 v
#> RMSE(test_y, pred_y_22) #rmse
#[1] 6210.651 ***
#> R2(pred_y_22, test_y) #R-Squared
#[1] 0.3107045 ***
#The root mean squared error turns out to be $6210.651. This represents the average difference between the prediction made for the modal mango price and the actual modal mango prices in the test set.
#MAE = $1613.641: The MAE of $1613.641 signifies that, on average, the absolute difference between the model's predicted modal mango prices and modal mango prices in the test set is approximately $1613.641. 

#(In this context, MAE is the average of the absolute differences between predicted and observed values. MAE is directly interpretable and represents the average magnitude of errors without considering the direction. A lower MAE indicates better performance in terms of the absolute error.)



#RMSE = $6210.651: An RMSE of $6210.651 means that, on average, the model's predicted modal mango prices differs from the actual modal mango price values by approximately $6210.651. 

#(The RMSE represents the square root of the average of the squared differences between predicted and observed values. RMSE penalizes larger errors more heavily than MAE due to the squaring operation.)


#R-Squared = 0.3107045:  An R-squared value of 31.07045% implies that the XGBoost model explains about 2.56% of the variance in the target variable, i.e., the modal mango price. This very low R-squared value indicates that the model's predicted modal mango prices does not align with the actual modal mango prices at all and captures minimal portion of the variability in the data.

#(R-squared measures the proportion of variance explained by the model.)


#  > #To calculate the testing MSE, MAE, RMSE, and R-Squared associated with the XGBoost prediction model with the lowest MAE ("cv_36")
#  > mean((test_y - pred_y_36)^2) #mse
#[1] 41214078
#> MAE(test_y, pred_y_36) #mae
#[1] 1440.424
#> RMSE(test_y, pred_y_36) #rmse
#[1] 6419.819
#> R2(pred_y_36, test_y) #R-Squared
#[1] 0.2722947 

##To produce the Variable Importance Plot for the XGBoost model
# Get feature importance
importance_matrix <- xgb.importance(feature_names = colnames(xgb_train), model = final_22)

# Plot the Variable Importance Plot (bar graph format available only)
xgb.plot.importance(importance_matrix, top_n = 10)

# Save the tuned model to RDS file
saveRDS(final_22, paste0(script_path(), .Platform$file.sep, "model.rds"))
