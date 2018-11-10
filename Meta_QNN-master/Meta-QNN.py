import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from Image_Reading import Image_Reading

height=32
width =32
train_wd = "C:\\Users\\CSH\\Desktop\\image_example\\train\\"
test_wd="C:\\Users\\CSH\\Desktop\\image_example\\test\\"

image, labels, test_image, test_labels, n_class = Image_Reading(height = height,
                                                                width = width,
                                                                train_wd = train_wd,
                                                                test_wd=test_wd)

### CNN Function Moduling
def Conv(n, f, l):
    global C_idx
    global layer
    global recent_n
    global x  # 이걸 붙여줘야 밖에서선언한 전역변수를 여기서도 받아서 쓰겠다는 의미가 된다.
    w = tf.get_variable(dtype=tf.float32, shape=[f, f, recent_n, n], name='w' + str(C_idx))
    if C_idx == 0:  # 첫 컨볼루션일 때
        layer = tf.nn.conv2d(filter=w, input=x, strides=[1, l, l, 1], padding='SAME')
    else:
        layer = tf.nn.conv2d(filter=w, input=layer, strides=[1, l, l, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    C_idx += 1
    recent_n = n


def Pooling(l):
    global P_idx
    global layer
    layer = tf.nn.max_pool(layer, ksize=[1, l, l, 1], strides=[1, l, l, 1], padding='SAME')
    P_idx += 1


def Softmax(n_class):
    global layer
    global P_idx
    if P_idx != 0:
        layer = tf.reshape(layer, shape=[-1, int((height / 2 ** P_idx) * (width / 2 ** P_idx) * recent_n)])  # 사진갯수가 -1
        w = tf.get_variable(dtype=tf.float32,
                            shape=[int((height / 2 ** P_idx) * (width / 2 ** P_idx) * recent_n), n_class],
                            name='softmax_w')
    else:
        layer = tf.reshape(layer, shape=[-1, int(height * width * recent_n)])  # 사진갯수가 -1
        w = tf.get_variable(dtype=tf.float32, shape=[height * width * recent_n, n_class], name='softmax_w')
    b = tf.get_variable(dtype=tf.float32, shape=[n_class], name='softmax_b')
    layer = tf.add(tf.matmul(layer, w), b)


### Building Graph & Training
def Training(structure):
    lr = 0.001
    n_epoch = 40
    batch_size = 200

    train_graph = tf.Graph()
    with train_graph.as_default():
        global x
        x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        y = tf.placeholder(dtype=tf.int32, shape=[None])
        y_onehot = tf.one_hot(y, depth=n_class)

        global C_idx;
        C_idx = 0
        global P_idx;
        P_idx = 0
        global recent_n;
        recent_n = 3

        print('Architecture : ', end="")
        for stage in structure:
            print(stage, end=' ')
            elem, size = stage.split()
            if elem == 'c':
                Conv(int(size), 3, 1)
            elif elem == 'p':
                Pooling(int(size))
            elif elem == 's':
                Softmax(int(size))
        print('\n')

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=layer, labels=y_onehot)
        training = tf.train.AdamOptimizer(lr).minimize(loss)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layer, 1), tf.argmax(y_onehot, 1)),
                                          tf.float32))  # reduce_mean 하기전에  tf.float32로 바꿔줘야 !

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            total_idx = np.arange(len(image));
            np.random.shuffle(total_idx)
            for ite in range(int(len(image) / batch_size)):
                batch_idx = total_idx[ite * batch_size: (ite + 1) * batch_size]
                batch_image = image[batch_idx]
                batch_labels = labels[batch_idx]
                sess.run(training, feed_dict={x: batch_image,
                                              y: batch_labels})
            if epoch % 10 == 0:
                acc = sess.run(accuracy, feed_dict={x: test_image,  # test accuracy
                                                    y: test_labels})
                print('Test Accuracy : ', acc)
        print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
    return (acc)


### 강화학습 적용
conv_list = ['c 64', 'c 128', 'c 256', 'c 512']
pooling_list = ['p 2', 'p 2', 's 4']
# softmax_list = ['s 4']
ch_to_idx = {elem: idx for idx, elem in enumerate(conv_list + pooling_list)}

# state
input_size = 10

# action
action_list = conv_list + pooling_list
output_size = len(action_list)

lr = 0.001
dis = 1

# q
tf.reset_default_graph()
RL_graph = tf.Graph()

with RL_graph.as_default():
    rf_x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
    rf_y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

    w1 = tf.get_variable(dtype=tf.float32, shape=[input_size, 64], name='w1')
    b1 = tf.get_variable(dtype=tf.float32, shape=[64], name='b1')
    w2 = tf.get_variable(dtype=tf.float32, shape=[64, output_size], name='w2')
    b2 = tf.get_variable(dtype=tf.float32, shape=[output_size], name='b2')

    q_layer = tf.add(tf.matmul(rf_x, w1), b1)
    q = tf.add(tf.matmul(q_layer, w2), b2)
    loss = tf.reduce_sum(tf.square(q - rf_y))
    training = tf.train.AdamOptimizer(lr).minimize(loss)

n_episode = 1000

with tf.Session(graph=RL_graph) as sess:
    sess.run(tf.global_variables_initializer())
    replay_memory = []
    acc_list = []
    for episode in range(n_episode):
        done = False
        e = 1 / (2 + episode / 10)
        layer_idx = 0
        state = np.array([0] * input_size).reshape([1, -1])  # 초기 state 설정
        action_record = []
        accuracy = 0
        while not done:
            Qs = sess.run(q, feed_dict={rf_x: state})
            action = action_list[np.argmax(Qs)]
            if np.random.rand(1) < e:
                action = action_list[np.random.choice(len(action_list), 1)[0]]
            if action == 's 4' or state[0][input_size - 2] != 0:  # 소프트맥스를 뽑거나 10개의 레이어를 모두 고르면 done =True를 주자
                done = True
            if len(action_record) == 0 and (action == 'p 2' or action == 's 4'):  # 처음부터 풀링이나 소프트맥스를 고르면 acc=0주고 끝내자
                done = True
                new_state = state.copy()
                new_state[0][layer_idx] = ch_to_idx[action] + 1
                replay_memory.append([state, ch_to_idx[action], new_state, accuracy, done])
                break
            new_state = state.copy()
            new_state[0][layer_idx] = ch_to_idx[action] + 1  # 전체를 +1해서 초기상태를 의미하는 0과 차별화
            # print( "state : ", state, 'new_state : ', new_state)
            action_record.append(action)
            if action_record[-1] == 's 4':
                accuracy = Training(action_record)
            else:
                accuracy = Training(action_record + ['s 4'])
            replay_memory.append(
                [state, ch_to_idx[action], new_state, accuracy, done])  # 이 state일떄 이 action을 하면 accruacy가 이래
            state = new_state.copy()
            layer_idx += 1

        if episode % 10 == 1 and len(replay_memory) > 10:
            for _ in range(50):
                groups = random.sample(replay_memory, 10)
                x_stack = np.empty(0).reshape(0, input_size)
                y_stack = np.empty(0).reshape(0, output_size)
                for group in groups:
                    s, u, n, a, d = group  # state, action, new_state, accuracy, done
                    Q = sess.run(q, feed_dict={rf_x: s})
                    if not d:
                        Q[0, u] = a + dis * np.max(sess.run(q, feed_dict={rf_x: n}))
                    else:
                        Q[0, u] = a
                    x_stack = np.vstack([x_stack, s])
                    y_stack = np.vstack([y_stack, Q])

                sess.run(training, feed_dict={rf_x: x_stack, rf_y: y_stack})
        acc_list.append(accuracy)
        print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ EPISODE RESTART ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
        if episode % 50 == 1 and len(replay_memory) > 10:
            print(sess.run(loss, feed_dict={rf_x: x_stack, rf_y: y_stack}))

plt.bar(range(len(acc_list)), acc_list)
plt.show()