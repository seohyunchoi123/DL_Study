rm(list=ls())
setwd("C:\\Users\\CSH\\Desktop\\투빅스 깃헙용 정리")
### KNN 가중치를 주는 함수 : 유사도 = 1/거리 사용해서 
### & 최적의 k 찾기 (5-fold cv)로 구현하기


wdbc <- read.csv('wisc_bc_data.csv', stringsAsFactors = F)
wdbc <- wdbc[-1]
wdbc$diagnosis <- factor(wdbc$diagnosis, level=c("B","M"))

set.seed(1)
idx <- sample(1:nrow(wdbc), 0.7*nrow(wdbc))
wdbc_train <- wdbc[idx,]
wdbc_test <- wdbc[-idx,]



Weighted_knn <- function (train, test, k){
  pred=c()
  for(i in 1:nrow(test)){
    dist_vec=c()
    for( j in 1:nrow(train)){
      dist = dist(rbind(train[j,-1], test[i,-1]), method = "euclidean") # let me try the function dist in different way 
      dist_vec = c(dist_vec, dist)
    }
    names(dist_vec) = train[,1]
    distance = head(sort(dist_vec), k)
    weight = distance^-1/sum(distance^-1)
    t = tapply(weight, INDEX =  names(weight), sum)
    class_predicted = names(which.max(t))
    pred = c(pred, class_predicted)
  }
  return(pred)
}


CrossValidation <- function(train){
  k = seq(3,7, 2)
  mean_acc=c()
  for(i in k){
    acc=c()
    for( j in 1:5){
      idx= (floor(nrow(train)/5*(j-1))+1) : (floor(nrow(train)/5*j))
      pred = Weighted_knn(train[-idx,], train[idx,], i)
      table = table(pred, train[idx,1])
      one_acc = sum(diag(table))/sum(table)
      acc = c(acc, one_acc)
    }
    mean_acc = c(mean_acc, mean(acc))
  }
  names(mean_acc) = k
  best_k = names(which.min(mean_acc))
  return(best_k)
}
 
k <- CrossValidation(wdbc_train)
pred <- Weighted_knn(wdbc_train, wdbc_test, k)
library(caret)
confusionMatrix(pred, wdbc_test$diagnosis) # ACCURACY 0.9064 

