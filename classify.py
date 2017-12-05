#!/usr/bin/python
# -*- coding: utf-8 -*-

from cnn_model import *
from data.cnews_loader import *
from sklearn import metrics
import sys

import time
from datetime import timedelta


base_dir = 'checkpoint/textcnn/'
test_dir = os.path.join(base_dir, 'test.txt') #测试文件
vocab_dir = os.path.join(base_dir, 'vocab.txt')  #词汇表
save_path = os.path.join(base_dir, 'best_validation')   # 预训练模型保存位置

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':

    print('Configuring CNN model...')
    start_time = time.time()
    config = TCNNConfig()
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    print("Loading test data...")
    x_test = process_file(test_dir, word_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')

    batch_size = 128 #批量数据大小
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
    for i in range(num_batch):   # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    #输出预测结果
    print(y_pred_cls)
    print(len(y_pred_cls))	
    #输出具体中文类别
    #categories = ['体育', '财经', '房产', '家居',
    #    '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    #for i in range(len(y_pred_cls)):
    #	print(categories[y_pred_cls[i]]) 

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
