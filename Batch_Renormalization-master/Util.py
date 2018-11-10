import tensorflow as tf


def batch_renorm(x, n_out, phase_train, r_max, d_max, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        # shadow_variable = decay * shadow_variable + (1 - decay) * variable # 주로 decay를 1에 가까이 놓는다

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,  # if절 대신에 이거!!
                            mean_var_with_update,  # update 시키는거다
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        r = tf.stop_gradient(tf.clip_by_value(ema.average(batch_var) / var, 1 / r_max, r_max))
        d = tf.stop_gradient(tf.clip_by_value((ema.average(batch_mean) - mean) / var, -d_max, d_max))

        normed = tf.cond(phase_train,
                         lambda: ((x - mean) / tf.sqrt(var + 1e-3) * r + d) * gamma + beta,
                         lambda: (x - mean) / tf.sqrt(var + 1e-3) * gamma + beta)
        # normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
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