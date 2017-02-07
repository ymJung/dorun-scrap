import tensorflow as tf
import numpy as np
import pymysql

input_vec_size = lstm_size = 7
time_step_size = 60
label_size = 3
evaluate_size = 3
lstm_depth = 4

total_size = 60000
batch_size = 15000
test_size = total_size - batch_size
conn = pymysql.connect(host='192.168.1.210',
                                          user='root',
                                          password='1234',
                                          db='data',
                                          charset='utf8mb4')


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W, B, lstm_size):
    XT = tf.transpose(X, [1, 0, 2])
    XR = tf.reshape(XT, [-1, lstm_size])
    X_split = tf.split(0, time_step_size, XR)
    cell = tf.nn.rnn_cell.GRUCell(lstm_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell = cell, output_keep_prob = 0.5)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * lstm_depth, state_is_tuple = True)
    outputs, _states = tf.nn.rnn(cell, X_split, dtype=tf.float32)
    return tf.matmul(outputs[-1], W) + B, cell.state_size

def read_series_datas(conn, code_dates):
    X = list()
    Y = list()
    for code_date in code_dates:
        cursor = conn.cursor()
        cursor.execute("SELECT open, high, low, close, volume, hold_foreign, st_purchase_inst FROM data.daily_stock WHERE code = %s AND date >= %s ORDER BY date LIMIT %s",
                       (code_date[0], code_date[1], time_step_size + evaluate_size))
        items = cursor.fetchall()
        X.append(np.array(items[:time_step_size]))
        price = items[-(evaluate_size + 1)][3]
        max = items[-evaluate_size][1]
        min = items[-evaluate_size][2]
        for item in items[-evaluate_size + 1:]:
            if max < item[1]:
                max = item[1]
            if item[2] < min:
                min = item[2]
        if (min - price) / price < -0.02:
            Y.append((0., 0., 1.))
        elif (max - price) / price > 0.04:
            Y.append((1., 0., 0.))
        else:
            Y.append((0., 1., 0.))

    arrX = np.array(X)
    norX = (arrX - np.mean(arrX, axis = 0)) / np.std(arrX, axis = 0)
    return norX, np.array(Y)

def read_datas():
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT code FROM data.daily_stock")
    codes = cursor.fetchall()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT date FROM data.daily_stock ORDER BY date")
    dates = cursor.fetchall()[:-(time_step_size + evaluate_size)]
    cnt = total_size
    code_dates = list()
    for date in dates:
        for code in codes:
            code_dates.append((code[0], date[0]))
            if --cnt <= 0:
                break
        if --cnt <= 0:
            break

    np.random.seed()
    np.random.shuffle(code_dates)
    trX = list()
    trY = list()
    trX, trY = read_series_datas(conn, code_dates[:batch_size])
    teX, teY = read_series_datas(conn, code_dates[-test_size:])

    return trX, trY, teX, teY
trX, trY, teX, teY = read_datas()
X = tf.placeholder(tf.float32, [None, time_step_size, input_vec_size])
Y = tf.placeholder(tf.float32, [None, label_size])
W = init_weights([lstm_size, label_size])
B = init_weights([label_size])
py_x, state_size = model(X, W, B, lstm_size)
loss = tf.nn.softmax_cross_entropy_with_logits(py_x, Y)
cost = tf.reduce_mean(loss)
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        test_indices = np.arange(len(teX))
        test_indices = test_indices[0:test_size]
        org = teY[test_indices]
        res = sess.run(predict_op, feed_dict={X: teX[test_indices], Y: teY[test_indices]})
        print(i, np.mean(np.argmax(org, axis=1) == res))