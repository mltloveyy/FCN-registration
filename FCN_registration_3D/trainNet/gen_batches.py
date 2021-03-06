#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/6 21:06
# @Author  : Hongjian Kang
# @File    : gen_batches.py


import os
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle as pkl

def gen_batches(x_dir: str, y_dir: str, config: dict):
    """
    给定x文件夹和y文件夹，生成batch tensor的函数
    :param x_dir: Moving Image文件夹绝对路径
    :param y_dir: Fixed Image 文件夹绝对路径
    :param config: config["shuffle_batch"]：是否shuffle batch
                    config["batch_size"]：batch大小
                    config["image_size"]：图像的height和width，tuple类型
    :return: Tensor('batch_x', dtype=float32, shape=[batch_size, img_height, img_width, 1])
            Tensor('batch_y', dtype=float32, shape=[batch_size, img_height, img_width, 1])
    """
    # 获得待配准图像绝对路径列表
    x_arr = [os.path.join(x_dir, _) for _ in os.listdir(x_dir)]
    y_arr = [os.path.join(y_dir, _) for _ in os.listdir(y_dir)]
    # 对绝对路径列表进行排序
    x_arr.sort(key=lambda _: int(os.path.split(_)[-1].split(".")[0]))
    y_arr.sort(key=lambda _: int(os.path.split(_)[-1].split(".")[0]))
    # 如果参考图像数量和待配准图像数量不同，那么意味着出错了
    assert len(x_arr) == len(y_arr)

    # 构建输入队列 & batch
    input_queue = tf.train.slice_input_producer([x_arr, y_arr], shuffle=config["shuffle_batch"])
    batch_x, batch_y = tf.train.batch(input_queue, batch_size=config["batch_size"])

    # 定义处理tensor的外部python函数
    # def _f(input_tensor, batch_size: int, img_height: int, img_width: int, img_depth: int, channels: int):
    #     _ = np.stack([normalize(np.array(Image.open(img_name))) for img_name in input_tensor], axis=0)
    #     return _.astype(np.float32).reshape([batch_size, img_height, img_width, img_depth, channels])
    def _f(input_tensor, batch_size: int, img_height: int, img_width: int, img_depth: int, channels: int):
        _ = np.zeros([1, img_height, img_width, img_depth, channels])
        for file_name in input_tensor:
            with open(file_name, 'rb') as f:
                volume_array = pkl.load(f)
            _ = np.vstack((_, volume_array))
        return normalize(_[1:].astype(np.float32).reshape([batch_size, img_height, img_width, img_depth, channels]))


    # 应用外部python函数处理tensor
    batch_x = tf.py_func(_f, [batch_x, config["batch_size"], *config["image_size"], 1], tf.float32)
    batch_y = tf.py_func(_f, [batch_y, config["batch_size"], *config["image_size"], 1], tf.float32)
    print(type(batch_x))

    # 返回batch
    return batch_x, batch_y


def normalize(input_array):
    # norm_array = (input_array - np.mean(input_array)) / np.std(input_array)
    output_array = (input_array - np.min(input_array)) / (np.max(input_array) - np.min(input_array))
    return output_array