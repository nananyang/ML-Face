import tensorflow as tf
import numpy as np
import os
import pickle


class FaceModel:

    def __init__(self, sess, name, classes_num):
        self.sess = sess
        self.name = name
        self.classes_num = classes_num
        self.model()


    def model(self):
        def set_weight(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=1e-2))


        with tf.variable_scope(self.name):

            self.rate = tf.placeholder(tf.float32)
            self.people = tf.placeholder(tf.float32)
            self.X = tf.placeholder(tf.float32, [None,28,28,3])
            #x_img = tf.reshape(self.X, [-1,28,28,3])
            self.Y = tf.placeholder(tf.float32, [None, self.classes_num])

            w1 = set_weight([5, 5, 3, 32])
            w2 = set_weight([3, 3, 32, 64])
            w3 = set_weight([3, 3, 64, 128])
            w4 = set_weight([3, 3, 128, 128])
            wfc1 = set_weight([7 * 7 * 128, 256])
            wfc2 = set_weight([256, self.classes_num])


            b1 = set_weight([32])
            b2 = set_weight([64])
            b3 = set_weight([128])
            b4 = set_weight([128])
            bfc1 = set_weight([256])
            bfc2 = set_weight([self.classes_num])

            h1 = tf.nn.relu(tf.nn.conv2d(self.X, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
            #p1 = tf.nn.max_pool(h1,ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
            d1 = tf.nn.dropout(h1, rate= self.rate)

            h2 = tf.nn.relu(tf.nn.conv2d(d1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
            p2 = tf.nn.max_pool(h2,ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
            d2 = tf.nn.dropout(p2, rate= self.rate)

            h3 = tf.nn.relu(tf.nn.conv2d(d2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
            #p3 = tf.nn.max_pool(h3,ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
            d3 = tf.nn.dropout(h3, rate= self.rate)

            h4 = tf.nn.relu(tf.nn.conv2d(d3, w4, strides=[1, 1, 1, 1], padding='SAME') + b4)
            p4 = tf.nn.max_pool(h4,ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
            d4 = tf.nn.dropout(p4, rate= self.rate)

            d4_conv = tf.reshape(d4, [-1, 7 * 7 * 128])
            hfc1 = tf.nn.relu(tf.matmul(d4_conv, wfc1) + bfc1)
            fd1 = tf.nn.dropout(hfc1, rate= self.rate)

            self.logits = tf.matmul(fd1, wfc2) + bfc2
            self.y_pred = tf.nn.softmax(self.logits)

            self.prediction = tf.argmax(self.y_pred, axis=1)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(self.logits,1),tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def predict(self,x_test, rate = 0):
        return self.sess.run(self.prediction, feed_dict={self.X : x_test, self.rate : rate})

    def get_accuracy(self,x_test,y_test, rate = 0):
        targets_one_hot = np.zeros((y_test.shape[0], self.classes_num))
        targets_one_hot[range(y_test.shape[0]), y_test] = 1
        targets_one_hot = np.array(targets_one_hot)
        return self.sess.run(self.accuracy, feed_dict={self.X : x_test,self.Y : targets_one_hot, self.rate : rate})

    def valid_accuracy(self, x_test, y_test, rate = 0):
        return self.sess.run([self.loss, self.accuracy], feed_dict={self.X: x_test, self.Y: y_test, self.rate: rate})

    def train(self, x_data, y_data, rate = 0.3):
        return self.sess.run([self.loss, self.optimizer], feed_dict={self.X : x_data, self.Y : y_data, self.rate :
            rate})



class Data_Reader():

    def __init__(self, inputs, targets, classes_num ,batch_size = None ):

        shuffled_idx = np.arange(inputs.shape[0])
        np.random.shuffle(shuffled_idx)
        self.inputs = inputs[shuffled_idx]
        self.targets = targets[shuffled_idx]
        self.classes_num = classes_num
        if batch_size is None:
            self.batch_size = 10
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        if len(self.inputs) >= batch_size:
            self.batch_count = self.inputs.shape[0] // self.batch_size
        else:
            self.batch_count = 1

    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()

        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1)
                            * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch +=1

        targets_one_hot = np.zeros((targets_batch.shape[0], self.classes_num))
        #targets_one_hot = tf.one_hot(targets_batch, self.classes_num)
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1
        targets_one_hot = np.array(targets_one_hot)


        return inputs_batch, targets_one_hot

    def __iter__(self):
        return self



class pp:
    def __init__(self,classes_num,train_epoch,batch_size):
        self.classes_num = classes_num
        with open('pp_data.pickle', 'rb') as f:

            temp_data = pickle.load(f)
            a, b, c, d, e, f = temp_data


        a = np.array(a)
        c = np.array(c)
        self.e = np.array(e)
        #print(a.shape[0])
        # b = tf.one_hot(b,5)
        # d = tf.one_hot(d,5)
        # temp_f = np.zeros((f.shape[0],self.classes_num))
        # temp_f[range(f.shape[0]),f] = 1
        # self.f = np.array(temp_f)
        self.f = f


        sess = tf.Session()
        self.fm = FaceModel(sess,'fm',classes_num)
        sess.run(tf.global_variables_initializer())

        self.train_epoch = train_epoch
        self.batch_size = batch_size


        self.train_data = Data_Reader(a,b,classes_num,self.batch_size)
        self.valid_data = Data_Reader(c,d,classes_num,self.batch_size)
        self.test_data = Data_Reader(e,f,classes_num,self.batch_size)
        # print(test_data.inputs)


    def training(self):

        prev_valid_loss = 9999999.
        valid_loss = 0.
        valid_accuray = 0.

        for epochs in range(self.train_epoch):

            curr_epoch_loss = 0.

            for input_batch, target_batch in self.train_data:
                batch_loss, _ = self.fm.train(input_batch, target_batch)
                curr_epoch_loss += batch_loss

            curr_epoch_loss /= self.train_data.batch_count



            for input_batch, target_batch in self.valid_data:
                valid_loss, valid_accuray = self.fm.valid_accuracy(input_batch, target_batch)

            print('epochs' + str(epochs +1)+'\t'+
                  'Training loss: '+'{0:.3f}'.format(curr_epoch_loss)+'\t'+
                  'Validation loss: '+'{0:.3f}'.format(valid_loss)+'\t'+
                  'Validation accuracy:'+'{0:.2f}'.format(valid_accuray*100.)+'%')

            if valid_loss > prev_valid_loss:
                break

            prev_valid_loss = valid_loss

        print('End of training')


        accu = self.fm.get_accuracy(self.e, self.f)
        print(accu)