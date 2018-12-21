################################################################################
## 0. 시행해 주세요
rm(list=ls())
work1 <- read.table('work1.txt', header = TRUE)
work2 <- read.table('work2.txt', header = TRUE)
work3 <- read.table('work3.txt', header = TRUE)
test <- read.table('test.txt', header = TRUE)

converting <- function(mydata) {
  for(i in c(1, 3, 4, 6, 7)) mydata[,i] <- factor(mydata[,i])
  return(mydata)
}

work1 <- converting(work1)
work2 <- converting(work2)
work3 <- converting(work3)
test <- converting(test)


#################################################################################
## 1. 초기화
PARAMS <- list(wife_edu_C1 = vector(mode = 'numeric', 4),
               husband_edu_C1 = vector(mode = 'numeric', 4),
               husband_job_C1 = vector(mode = 'numeric', 4),
               living_index_C1 = vector(mode = 'numeric', 4),
               wife_age_C1 = c(0, 8.2272),
               children_C1 = c(0, 2.3585),
               
               wife_edu_C2 = vector(mode = 'numeric', 4),
               husband_edu_C2 = vector(mode = 'numeric', 4),
               husband_job_C2 = vector(mode = 'numeric', 4),
               living_index_C2 = vector(mode = 'numeric', 4),
               wife_age_C2 = c(0, 8.2272),
               children_C2 = c(0, 2.3585),
               
               prior = vector(mode = 'numeric', 2),
               prev_cnt = c(0, 0))

##################################################################################
## 2. 모수 업데이트 함수
ParmasUpdate <- function(D, PARAMS) {
  
  d1 = subset(D, D$wife_work==0)
  d2 = subset(D, D$wife_work==1)
  present_cnt = c(nrow(d1), nrow(d2))
  prev_cnt = PARAMS$prev_cnt

  #연속형변수 업뎃
  PARAMS$wife_age_C1 = (PARAMS$wife_age_C1*prev_cnt[1] + c(mean(d1$wife_age), sd(d1$wife_age))*present_cnt[1]) / (present_cnt[1] + prev_cnt[1])
  PARAMS$children_C1 = (PARAMS$children_C1*prev_cnt[1] + c(mean(d1$children), sd(d1$children))*present_cnt[1]) / (present_cnt[1] + prev_cnt[1])
  
  PARAMS$wife_age_C2 = (PARAMS$wife_age_C2*prev_cnt[2] + c(mean(d2$wife_age), sd(d2$wife_age))*present_cnt[2]) / (present_cnt[2] + prev_cnt[2])
  PARAMS$children_C2 = (PARAMS$children_C2*prev_cnt[2] + c(mean(d2$children), sd(d2$children))*present_cnt[2]) / (present_cnt[2] + prev_cnt[2])
  
  #범주형변수 업뎃
  PARAMS$wife_edu_C1 = (PARAMS$wife_edu_C1*prev_cnt[1] + table(d1$wife_edu)) / (prev_cnt[1] + present_cnt[1]) 
  PARAMS$husband_edu_C1 = (PARAMS$husband_edu_C1*prev_cnt[1] + table(d1$husband_edu)) / (prev_cnt[1] + present_cnt[1]) 
  PARAMS$husband_job_C1 = (PARAMS$husband_job_C1*prev_cnt[1] + table(d1$husband_job)) / (prev_cnt[1] + present_cnt[1]) 
  PARAMS$living_index_C1 = (PARAMS$living_index_C1*prev_cnt[1] + table(d1$living_index)) / (prev_cnt[1] + present_cnt[1]) 
  
  PARAMS$wife_edu_C2 = (PARAMS$wife_edu_C2*prev_cnt[2] + table(d2$wife_edu)) / (prev_cnt[2] + present_cnt[2]) 
  PARAMS$husband_edu_C2 = (PARAMS$husband_edu_C2*prev_cnt[2] + table(d2$husband_edu)) / (prev_cnt[2] + present_cnt[2]) 
  PARAMS$husband_job_C2 = (PARAMS$husband_job_C2*prev_cnt[2] + table(d2$husband_job)) / (prev_cnt[2] + present_cnt[2]) 
  PARAMS$living_index_C2 = (PARAMS$living_index_C2*prev_cnt[2] + table(d2$living_index)) / (prev_cnt[2] + present_cnt[2]) 
    
  ##나머지 파라미터 업뎃
  PARAMS$prev_cnt = prev_cnt + present_cnt
  PARAMS$prior = present_cnt / sum(present_cnt)
  return(PARAMS)
}

# note

##################################################################################
## 3. predict 함수
predict.1 <- function(newdata, params) {
  result=c()
  for(i in 1:nrow(newdata)){
    #수치형
    x1_1 = dnorm(newdata[i,]$wife_age, params$wife_age_C1[1], params$wife_age_C1[2])
    x2_1 = dnorm(newdata[i,]$children, params$children_C1[1], params$children_C1[2])
    
    #범주형
    x3_1 = params$wife_edu_C1[newdata[i,]$wife_edu]
    x4_1 = params$husband_edu_C1[newdata[i,]$husband_edu]
    x5_1 = params$husband_job_C1[newdata[i,]$husband_job]
    x6_1 = params$living_index_C1[newdata[i,]$living_index]
    
    #c1 = log(x1_1)+log(x2_1)+log(x3_1)+log(x4_1)+log(x5_1)+log(x6_1)
    c1 = x1_1 * x2_1 * x3_1 * x4_1 * x5_1 * x6_1 * params$prior[1]
    
    #수치형
    x1_2 = dnorm(newdata[i,]$wife_age, params$wife_age_C2[1], params$wife_age_C2[2])
    x2_2 = dnorm(newdata[i,]$children, params$children_C2[1], params$children_C2[2])
    
    #범주형
    x3_2 = params$wife_edu_C2[newdata[i,]$wife_edu]
    x4_2 = params$husband_edu_C2[newdata[i,]$husband_edu]
    x5_2 = params$husband_job_C2[newdata[i,]$husband_job]
    x6_2 = params$living_index_C2[newdata[i,]$living_index]
    
    #c2 = log(x1_2)+log(x2_2)+log(x3_2)+log(x4_2)+log(x5_2)+log(x6_2)
    c2 = x1_2 * x2_2 * x3_2 * x4_2 * x5_2 * x6_2* params$prior[2]
    if(c1 > c2){
      result = c(result, 0)
    }
    else{
      result = c(result, 1)
    }
  }
  return(result)
}

###################################################################################
## 4. 결과 확인
library(naivebayes)

# 모수 추정이 같은가?
PARAMS_1 <- ParmasUpdate(work1, PARAMS)
model.1 <- naive_bayes(work1[,-1], work1[,1], laplace = F) # ok!

# 파라3, 모델3 만들기
PARAMS_2 <- ParmasUpdate(work2, PARAMS_1)
PARAMS_3 <- ParmasUpdate(work3, PARAMS_2)

work123 <- rbind(work1, work2, work3)
model.3 <- naive_bayes(work123[,-1], work123[,1])


## 파라3, 모델3 예측값 비교 
predict.1(test[,-1], PARAMS_3) # My prediction
predict(model.3, newdata=test[,-1])
 

#파라1, 모델1 예측값 비교 
table(predict(model.1, test[,-1]), predict.1(test[,-1], PARAMS_1)) # 완전히일치 ! 
