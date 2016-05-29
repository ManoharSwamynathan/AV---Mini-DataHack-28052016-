########################
# Using forecast package
########################

# load libraries
library("xts")
library("forecast")

# Read train and test data
train <- read.csv('../input/Train_KQyJ5eh.csv', sep=',', header=T)
test <- read.csv('../input/Test_HmLwURQ.csv', sep=',', header=T)

# Clean the date variable
train$Date <- as.Date(train$Date, format="%d-%b-%y")
test$Date <- as.Date(test$Date, format="%d-%b-%y")

# Creating time-series object
# This function takes a numeric vector, the start time and the frequency of measurement. 
# In our case these values are, 2007 (the year for which the measurements begin) and a frequency of 365 (days in a year)
ts_sales = ts(train$Number_SKU_Sold, start=2007, frequency=365)

plot(ts_sales)

# Model 1: Exponential State Smoothing
m_ets <- ets(ts_sales)
fore_ets <- forecast(m_ets, h=365) # forecast for next 365 days
plot(fore_ets) # Plot

# Model 2: ARIMA
# The auto.arima() function automatically searches for the best model and optimizes the parameters
m_aa = auto.arima(ts_sales)
fore_aa = forecast(m_aa, h=365)
plot(fore_aa)

# Model 3: TBATS
# TBATS is designed for use when there are multiple cyclic patterns (e.g. daily, weekly and yearly patterns) in a single time series. 
m_tbats = tbats(ts_sales)
fore_tbats = forecast(m_tbats, h=365)
plot(fore_tbats)

# Model comparison
# The model with the smallest AIC is the best fitting model.
barplot(c(ETS=m_ets$aic, ARIMA=m_aa$aic, TBATS=m_tbats$AIC), col="light blue", ylab="AIC")

# print performance matrices
accuracy(fore_ets)
accuracy(fore_aa)
accuracy(fore_tbats)

# Create submission file
test <- cbind(test, fore_tbats)
test <- test[,1:2]
names(test) <- c('Date','Number_SKU_Sold')

write.csv(test, file = '../output/submission.csv', row.names=F)

##############
# RandomForest
##############

# Clear all objects
rm(list=ls())

# load libraries
library(randomForest)

# Read train and test data
train <- read.csv('../input/Train_KQyJ5eh.csv', sep=',', header=T)
test <- read.csv('../input/Test_HmLwURQ.csv', sep=',', header=T)

# Clean the date variable
train$Date <- as.Date(train$Date, format="%d-%b-%y")
test$Date <- as.Date(test$Date, format="%d-%b-%y")

# Make features for train
train$year = as.numeric(substr(train$Date,1,4))
train$month = as.numeric(substr(train$Date,6,7))
train$day = as.numeric(substr(train$Date,9,10))
train$days =sapply(train$Date, function(x) as.numeric(difftimeDate(timeDate(x),timeDate(paste(substr(x,1,4),"-01-01",sep="")),"days")))

train$logsales = log(train$Number_SKU_Sold)
# weight certain features more by duplication, not sure if helpful?
train$tDays = 360*(train$year-2007) + (train$month-1)*30 + train$day
train$days30 = (train$month-1)*30 + train$day

# Make features for train
test$year = as.numeric(substr(test$Date,1,4))
test$month = as.numeric(substr(test$Date,6,7))
test$day = as.numeric(substr(test$Date,9,10))
test$days =sapply(test$Date, function(x) as.numeric(difftimeDate(timeDate(x),timeDate(paste(substr(x,1,4),"-01-01",sep="")),"days")))

# weight certain features more by duplication, not sure if helpful?
test$tDays = 360*(test$year-2007) + (test$month-1)*30 + test$day
test$days30 = (test$month-1)*30 + test$day

# Run model
Model =  randomForest(logsales ~ year + month + day + days + tDays + days30, ntree=4800, replace=TRUE, mtry=4, data=train)

train_pred = exp(predict(Model,train))
# Calculate RMSE
sqrt(mean((train$Number_SKU_Sold-train_pred)^2))

# Predict on test data
test_pred = exp(predict(Model,test))

# Create submission file
test <- cbind(test, test_pred)
test <- test[,c('Date','test_pred')]
names(test) <- c('Date','Number_SKU_Sold')

write.csv(test, file = "../output/Submission_RF.csv", row.names=F)

# scored 10th position on public leaderboard
