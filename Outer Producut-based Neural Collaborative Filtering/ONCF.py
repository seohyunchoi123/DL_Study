from aurochs.misc.serialize import PickleSerializer
from aurochs.buffalo import feature

import tensorflow as tf
import numpy as np
from collections import defaultdict, Counter
import datetime
import random
import pickle


tf.enable_eager_execution()

### 데이터 준비

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

dic_filtered

# positive 데이터 준비
train_pos = []
test_pos = []
for k, v in dic_filtered.items():
    for v_ in v[:-1]:
        train_pos.append((k, v_))
    test_pos.append((k, v[-1]))

#####  trainset, testset 나누기 (Leave-one-out-evaluation) #### 


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


def negative_sampling(dic_filtered, ratio, is_test = False): 
    result = []
    i=0
    for (k, v) in dic_filtered.items():
        if len(v) == 1: 
            raise AssertionError
        if is_test:
            row = mat[i][:ratio]
            while len(row) < ratio or 0 in row:
                row = np.random.randint(len(iid)+1, size=ratio)
            row = rm_duplicated(row, v, ratio)
            result.append((k, [v[-1]] + list(row), 0))
        else:
            row = mat[i][:len(v)*ratio]
            while len(row) < ratio or 0 in row:
                row = np.random.randint(len(iid)+1, size= len(v)*ratio)  
            row = rm_duplicated(row, v, ratio)
            result += zip([k]*len(row), v, list(row))
            
        i += 1
        if i%200000 == 0:
            print 'iter: {}, '.format(i), datetime.datetime.now()
            
    return result

    # 합치기 & 라벨구성 
print 'Start', datetime.datetime.now()
trainset = negative_sampling(dic_filtered, 1)
trainset = np.array(trainset)

print 'Start 2', datetime.datetime.now()
testset = negative_sampling(dic_filtered, 50, True)# [user_id, 정답 item_id, 정답포함한 50개 item_id 리스트],  ratio = 100으로하면 메모리 터진다 
print 'Finish', datetime.datetime.now()

# #####  데이터셋 저장 #####
# pickle.dump(trainset, open('./brunch_8d/trainset.pickle', 'wb'))
# pickle.dump(train_label, open('./brunch_8d/train_label.pickle', 'wb'))
# pickle.dump(testset, open('./brunch_8d/testset.pickle', 'wb'))
# pickle.dump(test_label, open('./brunch_8d/test_label.pickle', 'wb'))
# pickle.dump(test_pos, open('./brunch_8d/test_pos.pickle', 'wb'))

trainset[:20]

### 모델링

def batch_norm(x, n_out, phase_train, scope='bn', is_img = True):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        if is_img:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        else:
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

embedding_dim = 64

### 전체테스트셋으로 성능 측정

# Tensor Graph

train_graph = tf.Graph()   
with train_graph.as_default():
    # Placeholder
    x = tf.placeholder(tf.int32, shape=[None, 3], name='x') # [user_id, i, j]
    is_training = tf.placeholder(tf.bool, name='is_training')
    reg_coef = tf.placeholder(tf.float32, name='reg_coef')
    lr = tf.placeholder(tf.float32, name='lr')
    
    # Variable
        # Embedding 
    w_p = tf.Variable(tf.random_normal(shape=[len(uid_filtered), embedding_dim], stddev=0.01)) 
    b_p = tf.Variable(tf.random_normal(shape=[embedding_dim]))
    w_q = tf.Variable(tf.random_normal(shape=[len(iid), embedding_dim], stddev=0.01)) 
    b_q = tf.Variable(tf.random_normal(shape=[embedding_dim]))
    
        # Convolution
    w1 = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01), name="w1")
    w2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 32], stddev=0.01), name="w2")  
    w3 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 32], stddev=0.01), name="w3") 
    w4 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 32], stddev=0.01), name="w4")
    w5 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 32], stddev=0.01), name="w5") 
    w6 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 32], stddev=0.01), name="w6")
    w_flat = tf.Variable(tf.random_normal(shape=[32, 1], stddev=0.01), name="w_flat")
    
    b1 = tf.Variable(tf.random_normal(shape=[32,32,32], stddev=0.01), name="b1")
    b2 = tf.Variable(tf.random_normal(shape=[16,16,32], stddev=0.01), name="b2")
    b3 = tf.Variable(tf.random_normal(shape=[8,8,32], stddev=0.01), name="b3")
    b4 = tf.Variable(tf.random_normal(shape=[4,4,32], stddev=0.01), name="b4")
    b5 = tf.Variable(tf.random_normal(shape=[2,2,32], stddev=0.01), name="b5")
    b6 = tf.Variable(tf.random_normal(shape=[1,1,32], stddev=0.01), name="b6")
    b_flat = tf.Variable(tf.random_normal(shape=[1], stddev=0.01), name="b_flat")
    
    
    # y_ui
    p_i = tf.add(tf.matmul(tf.one_hot(x[:,0], depth=len(uid_filtered)), w_p), b_p)
    p_i = tf.nn.relu(batch_norm(p_i, embedding_dim, phase_train=is_training, is_img=False))  
    q_i = tf.add(tf.matmul(tf.one_hot(x[:,1], depth=len(iid)), w_q), b_q)
    q_i = tf.nn.relu(batch_norm(q_i, embedding_dim, phase_train=is_training, is_img=False)) 

    e_i = tf.matmul(tf.reshape(p_i, shape=[-1,64,1]), tf.reshape(q_i, shape=[-1,1,64]))
    e_i = tf.reshape(e_i, shape=[-1,64,64,1])

    L1_i = tf.nn.conv2d(input=e_i, filter=w1, strides=[1, 2, 2, 1], padding='SAME') + b1
    L1_i = tf.nn.relu(batch_norm(L1_i, 32, is_training, scope='bn'))
    L2_i = tf.nn.conv2d(input=L1_i, filter=w2, strides=[1, 2, 2, 1], padding='SAME') + b2
    L2_i = tf.nn.relu(batch_norm(L2_i, 32, is_training, scope='bn'))
    L3_i = tf.nn.conv2d(input=L2_i, filter=w3, strides=[1, 2, 2, 1], padding='SAME') + b3
    L3_i = tf.nn.relu(batch_norm(L3_i, 32, is_training, scope='bn'))
    L4_i = tf.nn.conv2d(input=L3_i, filter=w4, strides=[1, 2, 2, 1], padding='SAME') + b4
    L4_i = tf.nn.relu(batch_norm(L4_i, 32, is_training, scope='bn'))
    L5_i = tf.nn.conv2d(input=L4_i, filter=w5, strides=[1, 2, 2, 1], padding='SAME') + b5
    L5_i = tf.nn.relu(batch_norm(L5_i, 32, is_training, scope='bn'))
    L6_i = tf.nn.conv2d(input=L5_i, filter=w6, strides=[1, 2, 2, 1], padding='SAME') + b6
    L6_i = tf.nn.relu(batch_norm(L6_i, 32, is_training, scope='bn'))

    L6_flat_i = tf.reshape(L6_i, shape=[-1, 32])
    y_ui = tf.add(tf.matmul(L6_flat_i, w_flat), b_flat)
    y_ui = tf.identity(y_ui, name='y_ui')
    
    
    # y_uj
    p_j = tf.add(tf.matmul(tf.one_hot(x[:,0], depth=len(uid_filtered)), w_p), b_p)
    p_j = tf.nn.relu(batch_norm(p_j, embedding_dim, phase_train=is_training, is_img=False))  
    q_j = tf.add(tf.matmul(tf.one_hot(x[:,2], depth=len(iid)), w_q), b_q)
    q_j = tf.nn.relu(batch_norm(q_j, embedding_dim, phase_train=is_training, is_img=False)) 

    e_j = tf.matmul(tf.reshape(p_j, shape=[-1,64,1]), tf.reshape(q_j, shape=[-1,1,64]))
    e_j = tf.reshape(e_j, shape=[-1,64,64,1])

    L1_j = tf.nn.conv2d(input=e_j, filter=w1, strides=[1, 2, 2, 1], padding='SAME') + b1
    L1_j = tf.nn.relu(batch_norm(L1_j, 32, is_training, scope='bn'))
    L2_j = tf.nn.conv2d(input=L1_j, filter=w2, strides=[1, 2, 2, 1], padding='SAME') + b2
    L2_j = tf.nn.relu(batch_norm(L2_j, 32, is_training, scope='bn'))
    L3_j = tf.nn.conv2d(input=L2_j, filter=w3, strides=[1, 2, 2, 1], padding='SAME') + b3
    L3_j = tf.nn.relu(batch_norm(L3_j, 32, is_training, scope='bn'))
    L4_j = tf.nn.conv2d(input=L3_j, filter=w4, strides=[1, 2, 2, 1], padding='SAME') + b4
    L4_j = tf.nn.relu(batch_norm(L4_j, 32, is_training, scope='bn'))
    L5_j = tf.nn.conv2d(input=L4_j, filter=w5, strides=[1, 2, 2, 1], padding='SAME') + b5
    L5_j = tf.nn.relu(batch_norm(L5_j, 32, is_training, scope='bn'))
    L6_j = tf.nn.conv2d(input=L5_j, filter=w6, strides=[1, 2, 2, 1], padding='SAME') + b6
    L6_j = tf.nn.relu(batch_norm(L6_j, 32, is_training, scope='bn'))

    L6_flat_j = tf.reshape(L6_j, shape=[-1, 32])
    y_uj = tf.add(tf.matmul(L6_flat_j, w_flat), b_flat)

    # Loss
    param_sum = tf.reduce_sum(map(lambda x : tf.reduce_sum(x**2), [w_p, w_q, w1, w2, w3, w4, w5, w6, w_flat]))
    loss = tf.reduce_sum(-tf.log(tf.nn.sigmoid(y_ui - y_uj))) + reg_coef * tf.reduce_sum(param_sum)
    training = tf.train.AdagradOptimizer(lr).minimize(loss)
    
print('Graph Completed')

### 학습 & 테스트

batch_size = 128
n_epoch = 30
n_batch = len(trainset)/batch_size + 1
print(n_batch)

with tf.Session(graph = train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        if epoch == 0:
            coef = 0
            learning_rate = 0.05
        elif epoch >= 9:
            learning_rate /= 10
        else:
            coef = 1e-06
            
        print('-------------------------------- Epoch: {} --------------------------------'.format(str(epoch)))
        print 'Start Time: ', datetime.datetime.now()
        
        total_idx = np.arange(len(trainset))
        np.random.shuffle(total_idx)
        batch_idx_start = 0
        for i in range(n_batch):
            batch_idx_end = batch_idx_start + batch_size
            batch_idx = total_idx[batch_idx_start:batch_idx_end]
            batch_x = trainset[batch_idx]
            sess.run(training, feed_dict={x: batch_x, 
                                          is_training: True,
                                          reg_coef: coef,
                                          lr: learning_rate})
            batch_idx_start = batch_idx_end

            if i % 1000 == 0: 
                print 'Loss: {}, Time: {}'.format(sess.run(loss, feed_dict={x: batch_x, 
                                                                            is_training: False,
                                                                            reg_coef: coef}),
                                                  datetime.datetime.now())
#                 print 'w1: {}, Time: {}'.format(sess.run(w1, feed_dict={x: batch_x, 
#                                                                         is_training: False}),
#                                                   datetime.datetime.now())
#                 print 'y_uj: {}, Time: {}'.format(sess.run(y_uj, feed_dict={x: batch_x, 
#                                                                             is_training: False,
#                                                                             reg_coef: 0.001}),
#                                                   datetime.datetime.now())
                
        print('------------------------------ Test Result, Epoch : {} --------------------------------'.format(epoch))   
        # 실시간 테스트  
        hit = 0
        ndcg_score = 0
        batch_test_idx = np.random.choice(range(len(testset)), int(len(testset)*0.01), replace=False)
        for i in batch_test_idx:
            cand = testset[i][1]
            pred_reward =  sess.run(y_ui, feed_dict={x: zip([testset[i][0]]*len(cand), cand, [0]*len(cand)), 
                                                     is_training: False})
            rec_list = np.squeeze(np.array(cand)[np.argsort(np.squeeze(pred_reward))[::-1][:10]])
            hit += is_hit(rec_list, cand[0]) 
            ndcg_score += ndcg(rec_list, cand[0])

            
        print'HR Score: ', hit/float(len(batch_test_idx))
        print'NDCG Score: ', ndcg_score/float(len(batch_test_idx))
        
        # 저장
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.save(sess, save_path= '/data/private/brunch_8d/ONCF_Model/Model_{}.ckpt'.format(str(epoch)))
