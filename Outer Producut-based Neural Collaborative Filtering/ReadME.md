 # Outer Product-based Neural Collaborative Filtering (ONCF)
* 임베딩한 유저 벡터와 아이템 벡터를 CNN 구조로 분석하는 모델을 제안하고 있다. CNN을 통해 MLP보다 훨씬 적은 파라미터를 이용하여 더욱 좋은 성능을 낼 수 있다고 주장하고 있다.

## 실험 테스트 결과

**학습 환경 - 브런치 8d)**
>**ALS**: D=20, iter=10, num_proc=4, alpha=128.0,reg_u=1.0, reg_i=1.0

>**BPR**: D= 50, iter = 100, num_proc=12, reg_b = 0.01, reg_u = 0.01, reg_i = 0.01, reg_j = 0.01, decay = 0.95 (D가 50이상부터 성능이 비슷하다는 율의 이전 실험결과를 참고하여 D=50으로 설정. 나머지는 모두 기존 세팅 그대로 유지)

>**ONCF**: Embedding size = 64, iter = 10, optimizer = Adagrad, lr = 5e-02

**실험 환경 )**
>**실험 데이터**: 브런치 8d(전처리 후 총 유저수 1,940,636명)

>**테스트셋 구성**: 각 유저마다 정답 아이템 1개를 포함한 총 51개의 후보군 아이템 생성 (위와 동일)


**1. 브런치 8d 실험 결과 (HR, NDCG 순)**

![브런치_oncf_hr](https://user-images.githubusercontent.com/36473249/63907704-6cedcc80-ca57-11e9-8c57-7f659dbea52c.png)

![브런치_oncf_ndcg](https://user-images.githubusercontent.com/36473249/63907705-6cedcc80-ca57-11e9-946e-4d4e7c95d002.png)



## 도출된 결론

### **_" MLP(Multi-Layer Perceptron)구조의 뉴럴네트워크를 이용하면 단순한 내적연산보다 더욱 의미 있는 Collaborative Filtering이 가능하다 "_**

#### 근거)
1. 선형 내적과 뉴럴 네트워크가 앙상블로 합쳐진 구조인 Neural Collaborative Filtering 모델은 선형 내적뿐인 ALS보다 더 높은 성능을 보인다. (NCF 폴더 참고)
2. Neural Collaborative Filtering 모델에서 레이어를 추가할수록 성능이 높아지는 경향을 볼 수 있다.
3. 뉴럴네트워크 중에서도 ONCF의 CNN은 성능이 좋지 않았던 반면에 NCF의 MLP는 가장 높은 성능을 보이고 있다.


## 향후 과제들

1. ONCF의 Regularization Coefficient를 더욱 다양하게 실험해보기
