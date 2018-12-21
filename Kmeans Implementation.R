rm(list=ls())

my_kmeans=function(dat, k){
  cluster= rep(1, nrow(dat))
  idx = sample(nrow(dat), k)
  centroids = dat[idx,] # 중심점 임의지정
  
  while(1){ # 반복 시작 
    previous_cluster = cluster
    cluster=c()
    for(i in 1:nrow(dat)){
      dist_to_cnt=c()
      for(j in 1:k){
        t = dist(rbind(centroids[j,], dat[i,]), method = "euclidian")
        dist_to_cnt = c(dist_to_cnt, t)
      }
      names(dist_to_cnt) = 1:k
      cluster = c(cluster, names(which.min(dist_to_cnt)))
    }
    for(j in 1:k){
      centroids[j,] = colMeans(dat[cluster==j,])
    }
    t = previous_cluster == cluster
    if(sum(!t)<1) break
  }
  return(cluster)
}

data(iris)
dat = iris[,-5]
pred = my_kmeans(dat, 3)
pred
table(pred, iris$Species) # 150개중 134개가 제대로 분류됐음을 알 수 있다.

real = kmeans(dat, 3)$cluster
table(real, iris$Species) # 실제 kmeans 함수는 약 129개가 제대로 분류됐다. 구현함수와 성능이 비슷함을 알 수 있다. 
