#Problem:- https://inclass.kaggle.com/c/15-071x-the-analytics-edge-summer-2015
setwd("/Users/akshaykulkarni/Documents/Kaggle/Analytics_Edge_Summer_2015")
library(tm)
library(SnowballC)
library(caTools)
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(ROCR)
library(ggplot2)
set.seed(3000)

base_dir = "/Users/akshaykulkarni/Documents/Kaggle/Analytics_Edge_Summer_2015/"
eBayTrain = read.csv("eBayiPadTrain.csv", stringsAsFactors=FALSE)
eBayTest = read.csv("eBayiPadTest.csv", stringsAsFactors=FALSE)
eBayTrain$train = 1
eBayTest$train = 0
eBayTest$sold = 0

#Data Pre-Processing
data = rbind(eBayTrain,eBayTest)
data$biddable = as.factor(data$biddable)
data$condition = as.factor(data$condition)
data$cellular = as.factor(data$cellular)
data$carrier = as.factor(data$carrier)
data$color = as.factor(data$color)
data$storage = as.factor(data$storage)
data$productline = as.factor(data$productline)
data$sold = as.factor(data$sold)
data$startprice = scale(data$startprice)

#Converting text corpus to sparse Document term matrix
corpus = Corpus(VectorSource(data$description))
corpus = tm_map(corpus,content_transformer(tolower),lazy=TRUE)
corpus = tm_map(corpus,PlainTextDocument,lazy=TRUE)
corpus = tm_map(corpus,removePunctuation,lazy=TRUE)
corpus = tm_map(corpus,removeWords, stopwords("english"),lazy=TRUE)
corpus = tm_map(corpus,stemDocument,lazy=TRUE)
frequencies = DocumentTermMatrix(corpus)
sparse = removeSparseTerms(frequencies,0.995)
description_parse = as.data.frame(as.matrix(sparse))
colnames(description_parse) = make.names(colnames(description_parse))
data = cbind(data,description_parse)
data$description = NULL

train_data = subset(data,data$train==1)
test_data = subset(data,data$train==0)
test_data$sold = NULL
train_data$UniqueID = NULL

#Creating training and cross-validation set
split = sample.split(train_data$sold,SplitRatio=0.7)
train = subset(train_data,split==TRUE)
cv = subset(train_data,split==FALSE)

#Method 1 - Using CART Treess with cross-validations and cp
#how to select best minbucket size for CART tress.Using cross-validation to fix cp paramter
numFolds = trainControl(method="cv",number=10)
cpGrid = expand.grid(.cp=seq(0.01,0.5,0.01))
train(sold ~.,data = train,method="rpart",trControl=numFolds,tuneGrid=cpGrid)
cart_model_cv = rpart(sold ~ .,data=train,method="class",cp=0.01)
result_cart = get_auc_values(cart_model_cv,cv,cv$sold)
auc_cart = result_cart[1]
prob_cart = result_cart[-1]

#Method 2 - Now using random forests to make prediction
rf_model = randomForest(sold~.,data=train_data)
result_rf = get_auc_values(rf_model,cv,cv$sold)
auc_rf = result_rf[1]
prob_rf = result_rf[-1]

#Using Method-3 Using Logistic Regression and using only biddable, startprice, 
#condition, cellular, storage and productline as predictors 
model_lr = glm(sold ~ biddable + startprice + condition + cellular + storage + productline,data=train,family="binomial")
prob_lr = predict(model,newdata=cv,type="response")
pred = prediction(prob_lr,cv$sold)
auc_lr = performance(pred, "auc")@y.values

#Comparing auc and picking the best model

best_prob = predict(cart_model_cv,newdata=test_data,type="prob")[,2]
best_auc = auc_cart

if(auc_rf > best_auc)
{
  best_auc = auc_rf
  best_prob = predict(rf_model,newdata=test_data,type="prob")[,2]
}

if(auc_lr > best_auc)
{
  best_auc = auc_lr
  best_prob = predict(model_lr,newdata=test_data,type="response")
}

MySubmission = data.frame(UniqueID = test_data$UniqueID, Probability1 = best_prob)
write.csv(MySubmission, paste(base_dir,"SubmissionSimpleLog.csv"), row.names=FALSE)

get_auc_values <- function(model,data,output)
{
  predictROC = predict(model,newdata=data,type="prob")
  pred = prediction(predictROC[,2],output)
  c(as.numeric(performance(pred, "auc")@y.values),c(predictROC[,2]))
}
