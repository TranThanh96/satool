import tensorflow as tf
import numpy as np
import time
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cv2
import matplotlib.pyplot as plt
import sys

from common import *
from input import augmentation

class SA:
    def __init__(self, shape, no_classes, classify_model, log_text, alpha=0.3, beta=0.7, learning_rate=0.001, batch_size=3, logs_path='./logs', save_path_models='./models'):
        '''
        when change shape of input, need to change the first layer of Discriminator!!!
        '''
        self.log_text = log_text
        self.classify_model = classify_model
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.h, self.w, self.no_channels = shape
        self.no_classes = no_classes
        
        self.logs_path = logs_path
        self.save_path_models = save_path_models
        self.istraining = tf.placeholder(tf.bool, None, 'istraining')
        self.x1 = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'img_1')
        self.x2 = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'img_2')
        self.x3 = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'img_3')
        self.learning_rate = tf.Variable(learning_rate, name='lr', trainable=False)
        self.y = tf.placeholder(tf.float32, [None, self.no_classes], 'labels')
        self.stop_training = False
    
    def generator(self, x=[], name=None):
        with tf.variable_scope(name+'_concat'):
            input_ = tf.concat(x, axis=3)
        with tf.variable_scope(name):
            net = tf.layers.conv2d(
                input_,
                16,
                [3,3],
                padding='same',
                use_bias = False,
                activation=tf.nn.leaky_relu,
            )
            net = tf.layers.conv2d(
                net,
                16,
                [5,5],
                padding='same',
                use_bias = False,
                activation=tf.nn.leaky_relu,
            )
            net = tf.layers.conv2d(
                net,
                32,
                [7,7],
                padding='same',
                use_bias = False,
                activation=tf.nn.leaky_relu,
            )
            net = tf.layers.conv2d(
                net,
                32,
                [5,5],
                padding='same',
                use_bias = False,
                activation=tf.nn.leaky_relu,
            )
            net = tf.layers.conv2d(
                net,
                1,
                [3,3],
                padding='same',
                use_bias = False,
                activation=tf.nn.leaky_relu,
            )
        return net
    
    def classifier(self, input, istraining=False, reuse=False):
        logits = []
        softmax = []
        with tf.variable_scope('Classifier', reuse=reuse):
            
            exec(self.classify_model)
            # net = block(x, 32, istraining=istraining, name='Block_1')
            # net = block(net, 64, istraining=istraining, name='Block_2')
            # net = block(net, 128, istraining=istraining, name='Block_3')
            # net = block(net, 256, istraining=istraining, num_layers=3 ,name='Block_4')

            # net = tf.layers.flatten(net)

            # with tf.variable_scope('_dense_1'):
            #     net = tf.layers.dense(net, 256)
            #     net = tf.layers.batch_normalization(net, renorm=True, training=istraining)
            #     net = tf.nn.relu(net)
            #     net = tf.layers.dropout(net, training=istraining)

            # with tf.variable_scope('_dense_2'):
            #     net = tf.layers.dense(net, 256)
            #     net = tf.layers.batch_normalization(net, renorm=True, training=istraining)
            #     net = tf.nn.relu(net)
            #     net = tf.layers.dropout(net, training=istraining)

            # logits = tf.layers.dense(net, self.no_classes, activation=None)
            # softmax = tf.nn.softmax(logits)
        return 
    
    def build_model(self):
        generated_imgs = self.generator([self.x1, self.x2], name='Generator')
        
        with tf.variable_scope('concat_samples'):
            concat_imgs = tf.concat([generated_imgs, self.x3], axis=0)

        self.classifier(concat_imgs, istraining=self.istraining)
        logits = tf.get_default_graph().get_tensor_by_name("Classifier/output/BiasAdd:0")

        with tf.variable_scope("loss"):
            loss_G = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.x3, predictions=generated_imgs))
            loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits))
            loss_all = self.alpha * loss_G + self.beta * loss_C

        # self.summ_training_phase = tf.summary.merge_all(key='train')
        # self.generated_imgs = generated_imgs
        self.loss_all = loss_all


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.variable_scope('optim'):
                self.optim = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True).minimize(loss_all)


        self.input_classify = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'input_C')
        self.classifier(self.input_classify, istraining=False, reuse=True)
        logits = tf.get_default_graph().get_tensor_by_name("Classifier_1/output/BiasAdd:0")
        softmax = tf.nn.softmax(logits, name='predict')

        self.loss_c_deploy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits)
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(softmax,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self.no_correct_predict = tf.reduce_sum(tf.cast(correct_prediction, "float")) # so luong mau doan dung
        with tf.variable_scope("compare_train_and_val"):
            self.loss_set = tf.placeholder(tf.float32, name='loss_set_ph')
            tf.summary.scalar('loss_set', self.loss_set, collections=['full_train', 'val'])
            self.accu_set = tf.placeholder(tf.float32, name='accu_val_ph')
            tf.summary.scalar('accu_val', self.accu_set, collections=['full_train', 'val'])            

        self.summ_train_set = tf.summary.merge_all(key='full_train')
        self.summ_val_set = tf.summary.merge_all(key='val')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver_models = tf.train.Saver(max_to_keep=1)
        print('build model successfully...')
        # with tf.Session() as sess:
        #     self.writer_train = tf.summary.FileWriter('./log')
        #     self.writer_train.add_graph(sess.graph)
    
    def export_pb(self):
        if not os.path.isdir('./frozen_model'):
            os.makedirs('./frozen_model')
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            ['input_C','predict'] # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile("./frozen_model/model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
        # print("%d ops in the final graph." % len(output_graph_def.node))
    def random_class(self):
        return np.random.randint(0, self.no_classes, size=1)
    
    def random_batch(self, data):
        batch_size = self.batch_size
        batch_1 = []
        batch_2 = []
        batch_3 = []
        labels = []
        for _ in range(batch_size):
            class_id = np.random.randint(self.no_classes)
            class_size = data[class_id].shape[0]
            samples = np.random.randint(class_size, size=3)
            batch_1.append(data[class_id][samples[0]])
            batch_2.append(data[class_id][samples[1]])
            batch_3.append(data[class_id][samples[2]])
            labels.append(class_id)
        batch_1 = np.array(batch_1)
        batch_2 = np.array(batch_2)
        batch_3 = np.array(batch_3)
        labels =  one_hot(labels, self.no_classes)
        labels =  np.concatenate([labels, labels], axis=0)
        return batch_1, batch_2, batch_3, labels
    
    def summary_train(self, data):
        no_data = 0
        for i in range(self.no_classes):
            no_data += data[i].shape[0]
        if self.batch_size < 8:
            batch_summ = 8
        else:
            batch_summ = self.batch_size
        loss = []
        no_correct_predict = []
        for class_id in range(self.no_classes):
            no_samples = data[class_id].shape[0]
            for i in range(no_samples//batch_summ):
                x_batch = data[class_id][batch_summ*i:batch_summ*(i+1)]
                y_batch = np.ones((batch_summ))*class_id
                y_batch = one_hot(y_batch, self.no_classes)
                loss_batch, no_correct_predict_batch = self.sess.run([self.loss_c_deploy, self.no_correct_predict], feed_dict={self.input_classify: x_batch, self.y: y_batch })
                loss.append(loss_batch)
                no_correct_predict.append(no_correct_predict_batch)

            x_batch = data[class_id][batch_summ*(i+1):]
            y_batch = np.ones((x_batch.shape[0]))*class_id
            y_batch = one_hot(y_batch, self.no_classes)
            if x_batch.size:
                loss_batch, no_correct_predict_batch = self.sess.run([self.loss_c_deploy, self.no_correct_predict], feed_dict={self.input_classify: x_batch, self.y: y_batch })
                loss.append(loss_batch)
                no_correct_predict.append(no_correct_predict_batch)
        
        loss = np.concatenate(loss)
        loss = np.mean(loss)
        accu = np.sum(no_correct_predict)/no_data
        return loss, accu
    
    def summary_val(self, data, labels):
        no_data = data.shape[0]

        if self.batch_size < 8:
            batch_summ = 8
        else:
            batch_summ = self.batch_size
        loss = []
        no_correct_predict = []
        for i in range(no_data//batch_summ):
            x_batch = data[batch_summ*i:batch_summ*(i+1)]
            y_batch = labels[batch_summ*i:batch_summ*(i+1)]
            # y_batch = one_hot(y_batch, self.no_classes)

            loss_batch, no_correct_predict_batch = self.sess.run([self.loss_c_deploy, self.no_correct_predict], feed_dict={self.input_classify: x_batch, self.y: y_batch })
            loss.append(loss_batch)
            no_correct_predict.append(no_correct_predict_batch)
        x_batch = data[batch_summ*(i+1):]
        y_batch = labels[batch_summ*(i+1):]

        if x_batch.size:
            loss_batch, no_correct_predict_batch = self.sess.run([self.loss_c_deploy, self.no_correct_predict], feed_dict={self.input_classify: x_batch, self.y: y_batch })
            loss.append(loss_batch)
            no_correct_predict.append(no_correct_predict_batch)
        loss = np.concatenate(loss)
        loss = np.mean(loss)
        accu = np.sum(no_correct_predict)/no_data

        return loss, accu

    def train(self, data,  data_val, labels_val, epochs, tradition_aug=False, global_summ=0):
        batch_size = self.batch_size
        if batch_size < 8:
            batch_summ = 8
        else:
            batch_summ = batch_size
        no_data = 0
        for i in range(self.no_classes):
            no_data += data[i].shape[0]
        period_summ = no_data//batch_size
        period_reduce_lr = 5*period_summ
        no_iteration=period_summ*epochs

        self.log_text("\nInitiating...")

        self.sess.run(tf.global_variables_initializer())

        if not os.path.isdir(self.logs_path):
            os.makedirs(self.logs_path)
            os.makedirs('{}/train_summ'.format(self.logs_path))
            os.makedirs('{}/val_summ'.format(self.logs_path))

        self.writer_train = tf.summary.FileWriter('{}/training_phase'.format(self.logs_path))
        self.writer_train.add_graph(self.sess.graph)
        self.writer_train_set = tf.summary.FileWriter('{}/train_summ'.format(self.logs_path))
        self.writer_val_set = tf.summary.FileWriter('{}/val_summ'.format(self.logs_path))

        if not os.path.isdir(self.save_path_models):
            os.makedirs(self.save_path_models)

        loss_prev = 10000
        loss_sum_10000_iter_current = []

        # ========== test ==========

        self.log_text("\n Start training...")
        for iter_ in range(no_iteration):
            if self.stop_training:
                self.export_pb()
                self.log_text('\nStop training!')
                return
            x_1_, x_2_, x_3_, y_ = self.random_batch(data)
            if tradition_aug:
                x_1_ = augmentation(x_1_)
                x_2_ = augmentation(x_2_)
                x_3_ = augmentation(x_3_)
            # optim
            _ = self.sess.run(self.optim, feed_dict={self.x1: x_1_, self.x2: x_2_, self.x3: x_3_, self.y: y_, self.istraining: True})
            # sum training phase

            loss_curr = self.sess.run(self.loss_all, feed_dict={self.x1: x_1_, self.x2: x_2_, self.x3: x_3_, self.y: y_, self.istraining: False})

            loss_sum_10000_iter_current.append(loss_curr)

            if (iter_ + 1) % period_reduce_lr == 0:

                loss_mean = np.mean(loss_sum_10000_iter_current)
                if loss_prev <= loss_mean:
                    new_lr = self.sess.run(tf.assign(self.learning_rate, self.learning_rate*0.7))
                    self.log_text('\ndecrease learning rate by 30%: {}'.format(new_lr))
                loss_prev = loss_mean
                loss_sum_10000_iter_current = []
            
            if (iter_+1) % period_summ == 0: # period_summ
                start = time.time()

                loss_train, accu_train = self.summary_train(data)
                loss_val, accu_val = self.summary_val(data_val, labels_val)
                self.log_text("\n")
                self.log_text('\nloss_train: {}, accu_train: {}'.format(loss_train, accu_train))
                self.log_text('\nloss_val: {}, accu_val: {}'.format(loss_val, accu_val))
                summ_val = self.sess.run(self.summ_val_set, feed_dict={self.loss_set: loss_val, self.accu_set: accu_val})
                summ_train = self.sess.run(self.summ_train_set, feed_dict={self.loss_set: loss_train, self.accu_set: accu_train})
                self.writer_val_set.add_summary(summ_val, global_summ)
                self.writer_train_set.add_summary(summ_train, global_summ)

                save_path_models = self.saver_models.save(self.sess, "{}/model_{}/model.ckpt".format(self.save_path_models, global_summ))
                global_summ += 1
                self.log_text('\nsaved model in: {}'.format(save_path_models))
                stop = time.time()
                self.log_text('\ntime for summary: {}'.format(stop-start))
                self.log_text("\n=======================================")

