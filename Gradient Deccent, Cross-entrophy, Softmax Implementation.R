##################### 1.Gradient Descent 구현 과제입니다. ##########################
# 시각화용 코드입니다.
smoothing <- function(vec)
{
  vec1 <- c(vec[-1], vec[length(vec)])
  vec2 <- c(vec[1], vec[-length(vec)])
  return((vec1 + vec2 + vec) / 3)
}

visualize_loss <- function(loss_log)
{
  for(i in 1:100)
  {
    loss_log <- smoothing(loss_log)
    plot(loss_log)
    Sys.sleep(0.01)
  }
}
# 여기까지 그냥 실행시켜 주세요!

##############################################################################################################
#                                                                                                            #
#   이번 과제는 gradient descent를 이용한, 선형 회귀 구현 입니다. 아래에 비어있는 식을 채워주시면 됩니다!    #
#                                                                                                            #
##############################################################################################################
# 단순회귀 구현
x <- rnorm(1000, 0)
y <- 2 * x + 1
w <- 0.001
b <- 0.001
lr <- 0.01
loss_log <- c()
for(i in 1:length(x))
{
  ###                                         ###
  #            여기를 채워 주세요!              #
  loss =(  y[i] - x[i]*w - b)^2
  w = 
  ###                                         ###
  loss_log[i] <- loss
}
visualize_loss(loss_log)
if(max(abs(w-2), abs(b-1)) < 0.1)
{
  print("정답입니다!")
}else{
  print("모델을 수정하거나, 초기값, 파라미터를 수정해보세요!")
}

#다중회귀 구현(변수 11개)
x <- as.data.frame(matrix(rnorm(5000,0), nrow = 500, ncol = 10))
y <- x$V1 * 1 + x$V2 * 2 + x$V3 * 3 + x$V4 * 4 + x$V5 * 5 + x$V6 * 6 + x$V7 * 7 + x$V8 * 8 + x$V9 * 9 + x$V10 * 10 + 11
w <- rnorm(10,0)
b <- rnorm(1,0)
lr <- 0.01
loss_log <- c()
for(i in 1:nrow(x))
{
  ###                                         ###
  #            여기를 채워 주세요!              #
  loss = (y[i] - sum(w*x[i,])-b)^2
  w = w - lr * (y[i] - sum(w*x[i,]) - b) * -x[i,]
  b = b - lr * (y[i]  - sum(w*x[i,]) - b) * -1
  ###                                         ###
  loss_log[i] <- loss
}

visualize_loss(loss_log)
if(max(abs(w-1:10), abs(b-11)) < 0.5)
{
  print("정답입니다!")
}else{
  print("모델을 수정하거나, 초기값, 파라미터를 수정해보세요!")
}
a <- matrix(c(1,2,3,4), ncol = 2)
b <- c(1,2)
c <- rep(b,2)
a*c

#다중회귀 구현(변수 n개)
linear_regression <- function(n)
{
  x <- as.data.frame(matrix(rnorm(100*n*n,0), nrow = 100*n, ncol = n))
  y <- rep(0, 100*n)
  for(i in 1:(100*n))
  {
    y[i] <- sum(x[i,]*(1:n)) + (n+1)
  }
  w <- rnorm(n,0)
  b <- rnorm(1,0)
  lr <- 0.01
  loss_log <- c()
  for(i in 1:nrow(x))
  {
    ###                                         ###
    #            여기를 채워 주세요!              #
    loss = sqrt((y[i] - sum(w*x[i,])-b)^2)
    w = w - lr * (y[i] - sum(w*x[i,]) - b) * -x[i,]
    b = b - lr * (y[i]  - sum(w*x[i,]) - b) * -1
    
    ###                                         ###
    loss_log[i] <- loss
  }
  visualize_loss(loss_log)
  if(max(abs(w-1:n), abs(b-n-1)) < 0.5)
  {
    print("정답입니다!")
  }else{
    print("모델을 수정하거나, 초기값, 파라미터를 수정해보세요!")
  }
  return(list(w = w, b = b))
}
linear_regression(4)

linear_regression(10)
linear_regression(15)
linear_regression(20)

############# 2.Multinomial logistic에서 배운 softmax 와 cross_entropy 를 함수로 구현하세요 ###############
### 결과는 list(table, beta 계수) 반환하도록 해주세요 (GD 사용하실 경우, learning_rate 유의하세요)
## iris data에 한정적인 함수도 괜찮고, 일반화함수도 좋습니다.
# cross_entropy 함수로 beta를 구하고, softmax 함수에서 cross_entropy 함수를 받아들이면 됩니다.


rm(list=ls())
data("iris")
str(iris)
x<-iris[,-5]
y<-iris[,5]

set.seed(1234)
index <- sort(sample(1:length(x[,1]),length(x[,1])*0.8,replace = F))
train_x <- x[index,]
train_y <- y[index]  
test_x <- x[-index,]
test_y <- y[-index]

# cross entrophy try 1
w = matrix(runif(15, 0.5,2), 3, 5)
lr=0.005
loss=c()
cross_entrophy=0
type = unique(y)
train_x = cbind(train_x, 1)

for(k in 1:3){
  for(i in 1:nrow(train_x)){
    for ( j in 1:3){
      if (train_y[i] == type[j]){
        cross_entrophy = -log( exp(sum(w[j,]*train_x[i,]))  /
                                 (exp(sum(w[1,]*train_x[i,])) + exp(sum(w[2,]*train_x[i,])) + exp(sum(w[3,]*train_x[i,]))))
        w[j,] = w[j,] - lr * - train_x[i,] * 
          (1 - exp(sum(train_x[i,] * w[j,])) / (exp(sum(w[1,]*train_x[i,])) + exp(sum(w[2,]*train_x[i,])) + exp(sum(w[3,]*train_x[i,])))) 
        w = matrix(unlist(w), 3)
      }
    }
    loss = c(loss, cross_entrophy)
  }
}
plot(loss)

# softmax 
set.seed(1234)
index <- sort(sample(1:length(x[,1]),length(x[,1])*0.8,replace = F))
train_x <- x[index,]
train_y <- y[index]  
test_x <- x[-index,]
test_y <- y[-index]

test_x = cbind(test_x,1)
result_matrix = matrix(0, 1, 3)
result_matrix =as.matrix(test_x) %*% t(w) # 3번째가 제일 크게나옴 ㅜㅜ

result=c()
for( i in 1:nrow(result_matrix)){
  result = c(result, type[which.max(result_matrix[i,])])
}
result # 위 로스는 줄어드나 결과가 잘못됐음. cost를 class마다 계산하지 말고 모든 class에서의 cost를 loss함수로 정의하고 해보자 


############# try 2 ##############

#### cross entrophy 구현 (loss를 전체로 놓고 !) ###

rm(list=ls())
data("iris")
str(iris)
x<-iris[,-5]
y<-iris[,5]

set.seed(1)
index <- sort(sample(1:length(x[,1]),length(x[,1])*0.8,replace = F))
train_x <- x[index,]
train_y <- y[index]  
test_x <- x[-index,]
test_y <- y[-index]

lr = 0.0005 # lr을 0.005로 하니까 계속안됏다. 0.0005로해야됨 !!!
type = unique(train_y)
new_x = cbind(train_x, 1)
new_x = as.matrix(new_x)
w = matrix(runif(15, 0.2, 0.25), 3, 5) 
y_onehot = matrix(0, nrow(train_x), length(type))
for(i in 1:length(train_y)){
  y_onehot[i,train_y[i]] = 1
}
d = matrix(0,3,5)
loss_log = c()
i=1
for( k in 1:1000){  
  t = exp(new_x%*%t(w))/rowSums(exp(new_x%*%t(w)))
  cost = -sum(y_onehot * log(t))
  for(i in 1:length(type)){
    idx = which(y_onehot[,i] ==1)
    d[i,] = colSums(-new_x[idx,]* (1-t[idx,i]))
  }
  w = w - lr * d
  loss_log[k] = cost
}
plot(loss_log)

test_x = cbind(test_x,1)
output = as.matrix(test_x) %*% as.matrix(t(w))
result_list = apply(output, 1, which.max)
table = table(result_list, test_y)
sum(diag(table))/sum(table) # ACCURACY = 0.9333
