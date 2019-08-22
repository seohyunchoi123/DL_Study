from aurochs.misc.serialize import PickleSerializer
from aurochs.buffalo import feature

import tensorflow as tf
import numpy as np
from collections import defaultdict, Counter
import datetime
import random
import pickle


tf.enable_eager_execution()

### 필터링된 데이터 읽어오기

iid = open('/data/private/brunch_8d/iid').readlines()
iid = map(lambda x: x.replace('\n', ''), iid)

uid = open('/data/private/brunch_8d/uid').readlines()
uid = map(lambda x: x.replace('\n', ''), uid)

main =  open('/data/private/brunch_8d/main').readlines()
main = np.array(main[2:])# 
main = map(lambda x: map(int, x.split()), main) 
# main = main[:10000]

print(len(iid))
print(len(uid))
print(len(main))

### trainset, testset 나누기 (Leave-one-out-evaluation)

# main을 dic으로 바꿔주기
dic = defaultdict(list) # dic = {user_id : [item_id1 item_id2, ... ]}
for tmp in main:
    dic[tmp[0]].append(tmp[1])

# main, uid 필터링하기 (한 개만 본 유저들은 다 걸러내기)
uid_idx_filtered = []
main_filtered = []
idx = 1
for k, v in dic.items():
    if len(v) != 1:
        uid_idx_filtered.append(k)
        for v_ in v:
            main_filtered.append([idx, v_, 1])
        idx += 1

uid_filtered = list(np.array(uid)[np.array(uid_idx_filtered)-1])

# main_filtered을 dic_filtered으로 바꿔주기
dic_filtered = defaultdict(list) # dic = {user_id : [item_id1 item_id2, ... ]}
for tmp in main_filtered:
    dic_filtered[tmp[0]].append(tmp[1])

# positive 데이터 준비
train_pos = []
test_pos = []
for k, v in dic_filtered.items():
    for v_ in v[:-1]:
        train_pos.append((k, v_))
    test_pos.append((k, v[-1]))

# negative 데이터 준비 (Negative Sampling)

# tmp = Counter(np.array(main)[:,0])
# max_num_v = max(tmp.values())
max_num_v = 500
mat = np.random.randint(len(iid)+1, size=(len(dic_filtered), max_num_v))

def rm_duplicated(arr, v, ratio):
    unique_arr = set(arr)
    unique_v = set(v)
    diff = len(arr) - len(unique_arr - unique_v) # this should be 0
    if diff:
        return list(unique_arr - unique_v)
    return arr

def negative_sampling(dic, ratio, is_test = False): 
    result = []
    i=0
    for (k, v) in dic.items():
        if len(v) == 1: 
            assert AssertionError
        if is_test:
            row = mat[i][:ratio]
            while len(row) < ratio or 0 in row:
                row = np.random.randint(len(iid)+1, size=ratio)
            row = rm_duplicated(row, v, ratio)
            result.append((k, v[-1], [v[-1]] + list(row)))
        else:
            row = mat[i][:len(v)*ratio]
            while len(row) < ratio or 0 in row:
                row = np.random.randint(len(iid)+1, size= len(v)*ratio)  
            row = rm_duplicated(row, v, ratio)
            result += zip([k]*len(row), list(row))
            
        i += 1
        if i%200000 == 0:
            print 'iter: {}, '.format(i), datetime.datetime.now()
            
    return result

# 합치기 & 라벨구성 
print 'Start', datetime.datetime.now()
train_neg = negative_sampling(dic_filtered, 4)
trainset = np.array(train_pos + train_neg) # [user_id, item_id]
train_label = np.array([1]*len(train_pos) + [0]*len(train_neg))

print 'Start 2', datetime.datetime.now()
testset = negative_sampling(dic_filtered, 50, True)# [user_id, 정답 item_id, 정답포함한 50개 item_id 리스트],  ratio = 100으로하면 메모리 터진다 
print 'Finish', datetime.datetime.now()

del train_pos, train_neg, test_pos

# #####  데이터셋 저장 #####
# pickle.dump(trainset, open('./brunch_8d/trainset.pickle', 'wb'))
# pickle.dump(train_label, open('./brunch_8d/train_label.pickle', 'wb'))
# pickle.dump(testset, open('./brunch_8d/testset.pickle', 'wb'))
# pickle.dump(test_label, open('./brunch_8d/test_label.pickle', 'wb'))
# pickle.dump(test_pos, open('./brunch_8d/test_pos.pickle', 'wb'))

### 모델링

def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, # if절 대신에 이거!!
                            mean_var_with_update, # update 시키는거다
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def one_hot(x, n):
    result = np.zeros(n)
    result[x]=1
    return result

def one_hot_table(data, len_user, len_item):
    result = []
    for i in range(len(data)):
        result.append([one_hot(data[0], len_user), one_hot(data[1], len_item)])
    return np.array(result)

def is_hit(rec_list, true_y): # list, scalar
    if true_y in rec_list:
        return True
    return False


def ndcg(rec_list, true_y):
    dcg = 0
    idcg = 0
    binary_arr = []
    for j in range(len(rec_list)):
        if rec_list[j] == true_y:
            binary_arr.append(1)
        else:
            binary_arr.append(0)
    binary_arr_sorted = sorted(binary_arr, reverse=True)
    for j in range(len(binary_arr)):
        if j == 0:
            dcg += binary_arr[j]
            idcg += binary_arr_sorted[j]
            continue
        dcg += binary_arr[j] / float(np.log2(j+1))
        idcg += binary_arr_sorted[j] / float(np.log2(j+1))
    if dcg == 0:
        return 0
    return dcg/float(idcg)
        

# hyper parameter

embedding_dim_gmf = 16
embedding_dim_mlp = 16
predictive_factors = 256

# Tensor Graph

train_graph = tf.Graph()   
with train_graph.as_default():
    # Placeholder
    uid_x = tf.placeholder(tf.int32, shape=[None], name='uid_x')
    iid_x = tf.placeholder(tf.int32, shape=[None], name='iid_x')
    y = tf.placeholder(tf.float32, shape=[None], name='y')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    is_training = tf.placeholder(tf.bool, name='is_training')

    # GMF layer
        # Embeddig layer
    w_p_gmf = tf.Variable(tf.random_normal(shape=[len(uid_filtered), embedding_dim_gmf], stddev=0.01)) 
    b_p_gmf = tf.Variable(tf.random_normal(shape=[embedding_dim_gmf]))
    p_gmf = tf.add(tf.matmul(tf.one_hot(uid_x, depth=len(uid_filtered)), w_p_gmf), b_p_gmf)
    p_gmf = tf.nn.relu(batch_norm(p_gmf, embedding_dim_gmf, phase_train=is_training))  

    w_q_gmf = tf.Variable(tf.random_normal(shape=[len(iid), embedding_dim_gmf], stddev=0.01)) 
    b_q_gmf = tf.Variable(tf.random_normal(shape=[embedding_dim_gmf]))
    q_gmf = tf.add(tf.matmul(tf.one_hot(iid_x, depth=len(iid)), w_q_gmf), b_q_gmf)
    q_gmf = tf.nn.relu(batch_norm(q_gmf, embedding_dim_gmf, phase_train=is_training))  
        # output layer
    gmf = tf.reduce_sum(tf.multiply(p_gmf, q_gmf), axis=1)
    gmf = tf.reshape(tensor=gmf, shape=[-1,1])

    # MLP layer
        # Embedding layer
    w_p_mlp = tf.Variable(tf.random_normal(shape=[len(uid_filtered), embedding_dim_mlp], stddev=0.01)) 
    b_p_mlp = tf.Variable(tf.random_normal(shape=[embedding_dim_mlp]))
    p_mlp = tf.add(tf.matmul(tf.one_hot(uid_x, depth=len(uid_filtered)), w_p_mlp), b_p_mlp)
    p_mlp = tf.nn.relu(batch_norm(p_mlp, embedding_dim_mlp, phase_train=is_training))  

    w_q_mlp = tf.Variable(tf.random_normal(shape=[len(iid), embedding_dim_mlp], stddev=0.01)) 
    b_q_mlp = tf.Variable(tf.random_normal(shape=[embedding_dim_mlp]))
    q_mlp = tf.add(tf.matmul(tf.one_hot(iid_x, depth=len(iid)), w_q_mlp), b_q_mlp)
    q_mlp = tf.nn.relu(batch_norm(q_mlp, embedding_dim_mlp, phase_train=is_training)) 

        # Z1 layer
    w_1 = tf.Variable(tf.random_normal(shape=[embedding_dim_mlp*2, embedding_dim_mlp*2], stddev=0.01)) 
    b_1 = tf.Variable(tf.random_normal(shape=[embedding_dim_mlp*2], stddev=0.01)) 
    z1 = tf.add(tf.matmul(tf.concat([p_mlp, q_mlp], axis=1), w_1), b_1)
    z1 = tf.nn.relu(batch_norm(z1, embedding_dim_mlp*2, phase_train=is_training))

        # Z2 layer
    w_2 = tf.Variable(tf.random_normal(shape=[embedding_dim_mlp*2, predictive_factors*4], stddev=0.01)) 
    b_2 = tf.Variable(tf.random_normal(shape=[predictive_factors*4], stddev=0.01)) 
    z2 = tf.add(tf.matmul(z1, w_2), b_2)
    z2 = tf.nn.relu(batch_norm(z2, predictive_factors*4, phase_train=is_training))

        # Z3 layer
    w_3 = tf.Variable(tf.random_normal(shape=[predictive_factors*4, predictive_factors*2], stddev=0.01)) 
    b_3 = tf.Variable(tf.random_normal(shape=[predictive_factors*2], stddev=0.01)) 
    z3 = tf.add(tf.matmul(z2, w_3), b_3)
    z3 = tf.nn.relu(batch_norm(z3, predictive_factors*2, phase_train=is_training))

        # Z4 layer
    w_4 = tf.Variable(tf.random_normal(shape=[predictive_factors*2, predictive_factors], stddev=0.01)) 
    b_4 = tf.Variable(tf.random_normal(shape=[predictive_factors], stddev=0.01)) 
    z4 = tf.add(tf.matmul(z3, w_4), b_4)
    z4 = tf.nn.relu(batch_norm(z4, predictive_factors, phase_train=is_training))

        # output layer
    w_5 = tf.Variable(tf.random_normal(shape=[predictive_factors, 1], stddev=0.01)) 
    b_5 = tf.Variable(tf.random_normal(shape=[1], stddev=0.01)) 
    z5 = tf.add(tf.matmul(z4, w_5), b_5)
    mlp = tf.nn.relu(batch_norm(z5, 1, phase_train=is_training))

    # Final Reward
    w_h = tf.Variable(tf.random_normal(shape=[2, 1], stddev=0.01)) 
    reward_hat = tf.nn.sigmoid(tf.matmul(tf.concat([gmf, mlp], axis=1), w_h))
    reward_hat = tf.identity(reward_hat, name='reward_hat')

    # Loss
    loss = -tf.reduce_sum(y*tf.log(reward_hat) + (1-y)*tf.log(1-reward_hat))
    training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


### 학습 & 테스트

batch_size = 512
n_epoch = 30
n_batch = len(trainset)/batch_size + 1
print(n_batch)

with tf.Session(graph = train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    lr = 1e-05
    for epoch in range(n_epoch):
        print('-------------------------------- Epoch: {} --------------------------------'.format(str(epoch)))
        print 'Start Time: ', datetime.datetime.now()
        
        if epoch == 10:
            lr = 1e-06
        
        total_idx = np.arange(len(trainset))
        np.random.shuffle(total_idx)
        batch_idx_start = 0
        for i in range(n_batch): # 테스트할땐 500주고하자
            batch_idx_end = batch_idx_start + batch_size
            batch_idx = total_idx[batch_idx_start:batch_idx_end]
            batch_x = trainset[batch_idx]
            batch_y = train_label[batch_idx]
            sess.run(training, feed_dict={uid_x: batch_x[:,0], 
                                          iid_x: batch_x[:,1], 
                                          y: batch_y,
                                          learning_rate : lr,
                                          is_training: True})
            batch_idx_start = batch_idx_end

            if i % 1000 == 0: 
                print 'Loss: {}, Time: {}'.format(sess.run(loss, feed_dict={uid_x: batch_x[:,0], 
                                                                            iid_x: batch_x[:,1],
                                                                            y: batch_y,
                                                                            is_training: False}),
                                                  datetime.datetime.now())   
        
        # 저장
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.save(sess, save_path= '/data/private/brunch_8d/NCF_Model_256/NCF_Model_{}.ckpt'.format(str(epoch)))
        print('Save Completed')
        
        print('------------------------------ Test Result, Epoch : {} --------------------------------'.format(epoch))
       
        # 실시간 테스트  (1%만 샘플링해서)
        hit = 0
        ndcg_score = 0
        batch_test_idx = np.random.choice(range(len(testset)), int(len(testset)*0.01), replace=False)
        for i in batch_test_idx:
            cand = testset[i][2]
            pred_reward =  sess.run(reward_hat, feed_dict={uid_x: [testset[i][0]]*len(cand),
                                                           iid_x: cand,
                                                           is_training: False})
            rec_list = np.squeeze(np.array(cand)[np.argsort(np.squeeze(pred_reward))[::-1][:10]])
            hit += is_hit(rec_list, testset[i][1]) 
            ndcg_score += ndcg(rec_list, testset[i][1])
        print'HR Score: ', hit/float(len(batch_test_idx))
        print'NDCG Score: ', ndcg_score/float(len(batch_test_idx))
        
    print 'Finish Time: ', datetime.datetime.now()
