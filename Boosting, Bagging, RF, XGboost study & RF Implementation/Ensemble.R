rm(list=ls())

dat = read.csv("dat.csv")
head(dat)
str(dat)
dat$is_cancer = as.factor(dat$is_cancer)
set.seed(1)
idx = sample(nrow(dat), nrow(dat)*0.7)
train = dat[idx,]
test = dat[-idx,]


# Random Forest
library(caret)
library(randomForest)
?randomForest
fit = randomForest(is_cancer ~ .,importance=T ,data=train)
summary(fit)
pred = predict(fit, newdata = test[,-32])
confusionMatrix(pred, test[,32])

fit$importance # 변수 중요도 !
fit$err.rate[,1] # oob 확인, 각 나무마다 out of bag으로 테스트했을 시에 생겼던 에러율을 나타냄

# XG Boost
library(xgboost)
?xgboost
fit2 = xgboost(data.matrix(train[,-32]), data.matrix(train[,32]), nrounds = 10, eta = 0.5)
pred = predict(fit2, data.matrix(test[,-32]))
confusionMatrix(round(pred), test[,32])

xgb.importance(feature_names = colnames(train), model = fit2) # 변수 중요도 !
# 학습시킬때 설명변수와 종속변수를 구분해서 넣어줘야함 
# 인풋 데이터는 매트릭스 형태로 바꿔줘야함 ! 
# eta = learning rate를 의미함  (min:0 max:1 default:0.3)
# nround = 학습할때 반복 횟수 default 값 없음. 꼭 설정해줘야함 
# objective = loss를 최적화할 목적 함수를 어떤 형식으로 정의할건지 결정, default는 선형회귀로 설정돼있음 ! 
#     "reg:linear" 선형 회귀 “reg:logistic” 로지스틱 회귀
#     “binary:logistic” 이항 로지스틱회귀
#     “binary:logitraw”, “count:poisson” ,“multi:softmax”
#     “multi:softprob” ,“rank:pairwise”,“reg:gamma”,“reg:tweedie” 



# Bagging
library(adabag)
?bagging
fit3 = bagging(is_cancer ~., data=train) # 여기서 bagging 함수는 의사결정나무 모델을 이용한 함수다. 
confusionMatrix(pred$class, test[,32])

fit3$importance # 변수중요도
fit3$votes # 나무들의 투표 


# Boosting
?boosting
fit4 = boosting(is_cancer~., data=train) # 여기서 boosting 함수는 강의 ppt에서 배웠던 ADA Boost가 다중분류, 회귀까지 되도록 업그레이드 된 버전 
pred = predict(fit4, newdata = test[,-32])
confusionMatrix(pred$class, test[,32])

fit4$weights # 나무별 가중치
fit4$votes # 나무들의 투표 
fit4$importance # 변수 중요도 
