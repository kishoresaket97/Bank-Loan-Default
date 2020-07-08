rm(list=ls(all=T))
setwd("C:/Users/levi0/Downloads")
getwd()

#run this line of code to install all the packages 
install.packages(c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
                   "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees'))

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

lapply(x, require, character.only = TRUE)
rm(x)

loan_default = read.csv("bank-loan.csv", header = T, na.strings = c(" ", "", "NA"))

missing_val = data.frame(apply(loan_default,2,function(x){sum(is.na(x))}))

loan_default = na.omit(loan_default)

cnames = c("age", "employ", "address", "income", "debtinc", "creddebt", "othdebt")
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "default"), data = subset(loan_default))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="default")+
           ggtitle(paste("Box plot of responded for",cnames[i])))
}

## Plotting boxplots
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,ncol=2)

#Removing outliers
for(i in cnames)
{
  val = loan_default[,i][loan_default[,i] %in% boxplot.stats(loan_default[,i])$out]
  print (i)
  print(length(val))
  loan_default[,i][loan_default[,i] %in% val] = NA
}

#Imputing outliers with KNN method
loan_default = knnImputation(loan_default, k = 3)

#correlation-analysis
#Feature selection
library(corrplot)

M <- cor(loan_default)

corrplot::corrplot(M, method = 'number')

#removing income variable
cnames = c("age", "employ", "address", "debtinc", "creddebt", "othdebt")
loan_default <- subset(loan_default, select = c("age", "ed", "employ", "address", "debtinc", "creddebt", "othdebt", "default"))

#Normality check
qqnorm(loan_default$age)
hist(loan_default$age)

#Normalisation

for (i in cnames)
{print(i)
  loan_default[,i] = (loan_default[,i] - min(loan_default[,i]))/
    (max(loan_default[,i] - min(loan_default[,i])))
}

#Clean the environment
library(DataCombine)
rmExcept("loan_default")

#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(loan_default$default, p = .80, list = FALSE)
train = loan_default[ train.index,]
test  = loan_default[-train.index,]

#Logistic Regression
logit_model = glm(default ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions  = predict(logit_model, newdata = test, type = "response")

#install.packages("ROCR")
library(ROCR)

pred<-ROCR::prediction(logit_Predictions, test$default)
ROCRperf = performance(pred, "tpr", "fpr")

plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.3,1.6))

#convert probabilty 
logit_Predictions = ifelse(logit_Predictions > 0.4, 1, 0)

##Evaluate the performance of classification model
ConfMatrix_logit = table(test$default, logit_Predictions)
confusionMatrix(ConfMatrix_logit)

#Accuracy : 71%
#FNR : 42.5%

#Decison Tree
train$default<-as.factor(train$default)
str(train$default)

C50_model<-C5.0(train[-8], train$default)

C50_Predictions = predict(C50_model, test[,-8], type = "prob")

C50_Predictions=C50_Predictions[,2]

predc50<-ROCR::prediction(C50_Predictions, test$default)

ROCRperfc50 = performance(predc50, "tpr", "fpr")

plot(ROCRperfc50, colorize=TRUE, text.adj=c(-0.3,1.6))

C50_Predictions = ifelse(C50_Predictions > 0.5, 1, 0)

##Evaluate the performance of classification model
ConfMatrix_c50 = table(test$default, C50_Predictions)
confusionMatrix(ConfMatrix_c50)

#Accuracy : 70%
#FNR : 0.65


#RandomForest

RF_model = randomForest(default ~ ., train, importance = TRUE, ntree = 500)
RF_Predictions = predict(RF_model, test[,-8], type='prob')
RF_Predictions=RF_Predictions[,2]

predRF<-ROCR::prediction(RF_Predictions, test$default)
perfRF<-ROCR::performance(predRF,"tpr","fpr")

plot(perfRF, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.3,1.6))

#convert prob
RF_Predictions = ifelse(RF_Predictions > 0.4, 1, 0)

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$default, RF_Predictions)
confusionMatrix(ConfMatrix_RF)
#Accuracy : 60%
#FNR : 35%

##KNN Implementation
library(class)

#Predict test data
KNN_Predictions = knn(train[, 1:7], test[, 1:7], train$default, k = 17)

#Confusion matrix
ConfMatrix_KNN = table(KNN_Predictions, test$default)
confusionMatrix(ConfMatrix_KNN)
#Accuracy : 72%
#FNR : 46%

#Naive's Baye 
NB_model = naiveBayes(default ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:7], type = 'raw')

NB_Predictions=NB_Predictions[,2]

predNB<-ROCR::prediction(NB_Predictions, test$default)

perfNB<-ROCR::performance(predNB,"tpr","fpr")

plot(perfNB, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.2))

NB_Predictions = ifelse(NB_Predictions > 0.4, 1, 0)

ConfMatrix_NB = table(test$default, NB_Predictions)
confusionMatrix(ConfMatrix_NB)
#Accuracy : 74%
#FNR : 42.5%

#Oversampling 
library(DMwR)
loan_default$default = as.factor(loan_default$default)
smoted_data <- SMOTE(default~., loan_default, perc.over=100)

#To check there are equal number of 0s and 1s
table(smoted_data$default)

#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(smoted_data$default, p = .80, list = FALSE)
train = smoted_data[ train.index,]
test  = smoted_data[-train.index,]

#Logistic Regression
logit_model = glm(default ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions  = predict(logit_model, newdata = test, type = "response")

pred<-ROCR::prediction(logit_Predictions, test$default)

ROCRperf = performance(pred, "tpr", "fpr")

plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.3,1.6))

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.4, 1, 0)

##Evaluate the performance of classification model
ConfMatrix_logit = table(test$default, logit_Predictions)
confusionMatrix(ConfMatrix_logit)
#Accuracy : 69%
#FNR : 16%

#Decison Tree
train$default<-as.factor(train$default)
str(train$default)

C50_model<-C5.0(train[-8], train$default)

C50_Predictions = predict(C50_model, test[,-8], type = "prob")

C50_Predictions=C50_Predictions[,2]

predc50<-ROCR::prediction(C50_Predictions, test$default)

ROCRperfc50 = performance(predc50, "tpr", "fpr")

plot(ROCRperfc50, colorize=TRUE, text.adj=c(-0.3,1.6))

C50_Predictions = ifelse(C50_Predictions > 0.5, 1, 0)

##Evaluate the performance of classification model
ConfMatrix_c50 = table(test$default, C50_Predictions)
confusionMatrix(ConfMatrix_c50)
#Accuracy : 76%
#FNR : 23%


#RandomForest

RF_model = randomForest(default ~ ., train, importance = TRUE, ntree = 500)
RF_Predictions = predict(RF_model, test[,-8], type='prob')
RF_Predictions=RF_Predictions[,2]

predRF<-ROCR::prediction(RF_Predictions, test$default)
perfRF<-ROCR::performance(predRF,"tpr","fpr")

plot(perfRF, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.3,1.6))

#convert prob
RF_Predictions = ifelse(RF_Predictions > 0.4, 1, 0)

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$default, RF_Predictions)
confusionMatrix(ConfMatrix_RF)
#Accuracy : 83%
#FNR :6%

##KNN Implementation
library(class)

#Predict test data
KNN_Predictions = knn(train[, 1:7], test[, 1:7], train$default, k = 17)

#Confusion matrix
ConfMatrix_KNN = table(KNN_Predictions, (test$default))
confusionMatrix(ConfMatrix_KNN)
#Accuracy : 71% 
#FNR : 31%

#Naive's Baye 
NB_model = naiveBayes(default ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:7], type = 'raw')

NB_Predictions=NB_Predictions[,2]

predNB<-ROCR::prediction(NB_Predictions, test$default)

perfNB<-ROCR::performance(predNB,"tpr","fpr")

plot(perfNB, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.2))

NB_Predictions = ifelse(NB_Predictions > 0.4, 1, 0)

ConfMatrix_NB = table(test$default, NB_Predictions)
confusionMatrix(ConfMatrix_NB)
#Accuracy : 71%
#FNR : 15% 

#RandomForest with 500 trees and a threshold of 0.4 proved to be the best model after oversampling
#RandomForest produced the best output with an Accuracy of 83% and false negative rate of 6%

#The Naive's Baye proved better here than KNN probably because converting the output of 
#KNN method to probabilities and back manually with a threshold is not a straightforward task 
#like it is in python/jupyter notebook and hence was skipped 
