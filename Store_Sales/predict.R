setwd("/Users/akshaykulkarni/Documents/Kaggle/Store_Sales")
#Approach:-
# Splitting and grouping data by StoreId
# More each store generating a different model
# Using Linear regression, CART tress and randomForests to get best accuracy

library(caTools)
library(rpart)
library(randomForest)

get_best_training_model = function (data_set)
{
  split = sample.split(data_set,SplitRatio=0.90)
  tr = subset(data_set,split==TRUE)
  te = subset(data_set,split==FALSE)
  
  model_lm = lm(Sales~Open+Promo+DayOfWeek+SchoolHoliday,data=tr)
  answer_lm = predict(model_lm,newdata=te)
  rss_lm = sqrt(sum((answer_lm-te$Sales)^2)/nrow(te))
  
  model_cart = rpart(Sales~Open+Promo+DayOfWeek+SchoolHoliday,data=tr)
  answer_cart = predict(model_cart,newdata=te)
  rss_cart = sqrt(sum((answer_cart-te$Sales)^2)/nrow(te))
  
  model_rf = randomForest(Sales~Open+Promo+DayOfWeek+SchoolHoliday,data=tr)
  answer_rf = predict(model_rf,newdata=te)
  rss_rf = sqrt(sum((answer_rf-te$Sales)^2)/nrow(te))
  
  best_error = rss_cart
  best_model = model_cart
  
  if(best_error > rss_cart)
  {
    best_error = rss_cart;
    best_model = model_cart
  }
  
  if(best_error > rss_rf)
  {
    best_error = rss_rf;
    best_model = model_rf
  }
  
  return(best_model)
}

get_predicted_value = function(model)
{
  return(predict(model,newdata=test))
}

sales = read.csv("data/train.csv",header=TRUE)
store = read.csv("data/store.csv",header=TRUE)
test = read.csv("data/test.csv",header=TRUE)

#Merging two data-frames so that all relevant data is in 1 frame
train = merge(sales,store,by="Store",all.x=TRUE)
#Splitting training data by Stores
train_store = split(train,f=train$Store)
training_models = lapply(train_store,get_best_training_model)

final_sales = c()
sales_idx = c()
for(i in 1:nrow(test))
{
  final_sales = append(final_sales,ceiling(predict(training_models[[test[i,]$Store]],
                                           newdata=as.data.frame(test[i,]))))
  sales_idx = append(sales_idx,test[i,]$Id)
}

result_df = data.frame(sales_idx,final_sales)
names(result_df) = c("Id","Sales")
write.csv(result_df,file="data/output.csv",row.names=FALSE)
