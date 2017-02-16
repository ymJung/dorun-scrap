import tensorflow as tf
import numpy as np
import pymysql
from datetime import date, timedelta

LIMIT_FILTER = 0.70

INPUT_VEC_SIZE = LSTM_SIZE = 7
TIME_STEP_SIZE = 60
LABEL_SIZE = 3
EVALUATE_SIZE = 3
LSTM_DEPTH = 4

BATCH_SIZE = 15000
TRAIN_CNT = 100


class DBManager :
    def __init__(self):
        DB_IP = '192.168.1.210'
        DB_USER = 'root'
        DB_PWD = '1234'
        DB_SCH = 'data'
        DB_ENC = 'utf8mb4'
        self.conn = pymysql.connect(host=DB_IP, user=DB_USER, password=DB_PWD, db=DB_SCH, charset=DB_ENC)
        
    def __del__(self):
        self.conn.close()
        
    def get_codedates(self, code, limit):    
        query = "SELECT date FROM data.daily_stock WHERE code = %s AND date <= %s ORDER BY date ASC"
        cursor = self.conn.cursor()
        cursor.execute(query, (code, limit))
        code_dates = list()        
        dates = cursor.fetchall()
        for date in dates:
            code_dates.append((code, date[0]))
        return code_dates
    def get_items(self, code, date, limit):
        query = "SELECT open, high, low, close, volume, hold_foreign, st_purchase_inst FROM data.daily_stock WHERE code = %s AND date >= %s ORDER BY date ASC LIMIT %s"
        cursor = self.conn.cursor()
        cursor.execute(query, (code, date, limit))
        items = cursor.fetchall()        
        return items
    
    def get_codes(self):
        query = "SELECT DISTINCT code FROM data.daily_stock"
        cursor = self.conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))



    
    
    
def model(code, X, W, B, lstm_size):
    XT = tf.transpose(X, [1, 0, 2]) 
    XR = tf.reshape(XT, [-1, lstm_size])
    X_split = tf.split(0, TIME_STEP_SIZE, XR)
    with tf.variable_scope(code, reuse=False):
        cell = tf.nn.rnn_cell.GRUCell(lstm_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell = cell, output_keep_prob = 0.5)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * LSTM_DEPTH, state_is_tuple = True)

    outputs, _states = tf.nn.rnn(cell, X_split, dtype=tf.float32)

    return tf.matmul(outputs[-1], W) + B, cell.state_size # State size to initialize the stat

def read_series_datas(db, code_dates):
    EXPECT = 6
    X = list()
    Y = list()
    for code_date in code_dates:
        items = db.get_items(code_date[0], code_date[1], TIME_STEP_SIZE + EVALUATE_SIZE)
  
        if len(items) < (EVALUATE_SIZE + TIME_STEP_SIZE):
            break
        X.append(np.array(items[:TIME_STEP_SIZE]))

        st_purchase_inst = items[-(EVALUATE_SIZE + 1)][EXPECT]
        if st_purchase_inst == 0:
            continue
        for i in range(EVALUATE_SIZE, len(items) - EVALUATE_SIZE):
            eval_inst = items[i][EXPECT]
            eval_bef = items[EVALUATE_SIZE-i][EXPECT]
            if eval_bef < eval_inst:
                eval_bef = eval_inst           
        
        if (eval_bef - st_purchase_inst) / st_purchase_inst < -0.02: #percent ? cnt ? 
            Y.append((0., 0., 1.))
        elif (eval_bef - st_purchase_inst) / st_purchase_inst > 0.03:
            Y.append((1., 0., 0.))
        else:
            Y.append((0., 1., 0.))


    arrX = np.array(X)    
    meanX = np.mean(arrX, axis = 0)
    stdX = np.std(arrX, axis = 0)
    norX = (arrX - meanX) / stdX
    norY = np.array(Y)
    return norX, norY
def read_datas(db, code_dates):    
    np.random.seed()
    np.random.shuffle(code_dates)

    trX = list()
    trY = list()
    trX, trY = read_series_datas(db, code_dates)
    teX, teY = read_series_datas(db, code_dates)

    return trX, trY, teX, teY
def analyze(code, limit):      
    db = DBManager()
    code_dates = db.get_codedates(code, limit)
    tf.reset_default_graph()    
    last = code_dates[-1][1]
    trX, trY, teX, teY = read_datas(db, code_dates)
    if (len(trX) == 0):
        return None

    X = tf.placeholder(tf.float32, [None, TIME_STEP_SIZE, INPUT_VEC_SIZE])
    Y = tf.placeholder(tf.float32, [None, LABEL_SIZE])

    W = init_weights([LSTM_SIZE, LABEL_SIZE])
    B = init_weights([LABEL_SIZE])

    py_x, state_size = model(code, X, W, B, LSTM_SIZE)

    loss = tf.nn.softmax_cross_entropy_with_logits(py_x, Y)
    cost = tf.reduce_mean(loss)
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    # Launch the graph in a session
    analyzed = None
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        for loop in range(TRAIN_CNT):
            for start, end in zip(range(0, len(trX), BATCH_SIZE), range(BATCH_SIZE, len(trX)+1, BATCH_SIZE)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            test_indices = np.arange(len(teX))
            org = teY[test_indices]
            res = sess.run(predict_op, feed_dict={X: teX[test_indices], Y: teY[test_indices]})
            
            if loop == TRAIN_CNT-1 :
                result = np.mean(np.argmax(org, axis=1) == res)                
                analyzed = {"stock":code, "per":round(result, 2), "date":limit}
                print(analyzed)
    return analyzed

limit = '2017-02-14'
codes = DBManager().get_codes() #
filtered = list()
for code in codes : 
    analyzed = analyze(code[0], limit)
    if analyzed is not None and analyzed["per"] > LIMIT_FILTER:
        filtered.append(analyzed)
print(filtered)
