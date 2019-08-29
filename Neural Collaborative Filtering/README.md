### Neural Collaborative Filtering (NCF)
* 임베딩한 유저 벡터와 아이템 벡터를 내적 + 뉴럴네트워크(MLP) 구조로 분석하는 모델을 제안하고 있다. 내적이라는 선형적 방법 외에도 뉴럴네트워크라는 비선형적 연산까지 앙상블로 이용하여 ALS를 능가할 수 있다고 주장하는 논문이다.

## 실험 테스트 결과

**학습 환경 - 브런치 8d)**
>**ALS**: D=20, iter=10, num_proc=4, alpha=128.0,reg_u=1.0, reg_i=1.0

>**BPR**: D= 50, iter = 100, num_proc=12, reg_b = 0.01, reg_u = 0.01, reg_i = 0.01, reg_j = 0.01, decay = 0.95 (D가 50이상부터 성능이 비슷하다는 율의 이전 실험결과를 참고하여 D=50으로 설정. 나머지는 모두 기존 세팅 그대로 유지)

>**NCF**: Embedding size = 16, iter = 10, optimizer = Adam, lr = 1e-05 (6번째 에폭부터 1e-06)

**학습 환경 - 카페 1d)**
>**ALS**: D=40, iter=10, num_proc=8, alpha=8.0,reg_u=8.0, reg_i=8.0

>**BPR**: D= 50, iter = 100, num_proc=12, reg_b = 0.01, reg_u = 0.01, reg_i = 0.01, reg_j = 0.01, decay = 0.95 (D가 50이상부터 성능이 비슷하다는 율의 이전 실험결과를 참고하여 D=50으로 설정. 나머지는 모두 기존 세팅 그대로 유지)

>**NCF**: Embedding size = 16, iter = 10, optimizer = Adam, lr = 1e-05 (6번째 에폭부터 1e-06)

**실험 환경 )**
>**실험 데이터**: 브런치 8d(전처리 후 총 유저수 1,940,636명), 카페 1d(전처리 후 총 유저수 809,416명)

>**테스트셋 구성**: 각 유저마다 정답 아이템 1개를 포함한 총 51개의 후보군 아이템 생성 (위와 동일)


**1. 브런치 8d 실험 결과 (HR, NDCG 순)**

![브런치_ncf_hr](https://user-images.githubusercontent.com/36473249/63907710-6d866300-ca57-11e9-87ab-22bc5d34782d.png)

![브런치_ncf_ndcg](https://user-images.githubusercontent.com/36473249/63907711-6d866300-ca57-11e9-9008-9a228f2da9d1.png)

**2. 카페 1d 실험 결과 (HR, NDCG 순)**

![카페_hr](https://user-images.githubusercontent.com/36473249/63907707-6cedcc80-ca57-11e9-84c9-538b2206b9ee.png)

![카페_ndcg](https://user-images.githubusercontent.com/36473249/63907709-6d866300-ca57-11e9-83b2-726cc8871f7e.png)


## 도출된 결론

### **_" MLP(Multi-Layer Perceptron)구조의 뉴럴네트워크를 이용하면 단순한 내적연산보다 더욱 의미 있는 Collaborative Filtering이 가능하다 "_**

#### 근거)
1. 선형 내적과 뉴럴 네트워크가 앙상블로 합쳐진 구조인 Neural Collaborative Filtering 모델은 선형 내적뿐인 ALS보다 더 높은 성능을 보인다.
2. Neural Collaborative Filtering 모델에서 레이어를 추가할수록 성능이 높아지는 경향을 볼 수 있다.


## 향후 과제들

1. NCF와 ALS 간의 격차가 브런치 데이터에서는 컸는데 카페 데이터에서는 작아진 이유를 밝혀내기
2. NCF에 더욱더 많은 레이어 쌓아서 실험해보기
3. K-FOLD를 통해 실험 결과의 신뢰도 더욱 높이기
