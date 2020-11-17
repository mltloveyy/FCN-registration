import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from models.fcn_new import fcnRegressor
from trainNet.config_folder_guard import config_folder_guard
from trainNet.gen_batches import gen_batches
from trainNet.logger import my_logger as logger

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    root_dir = "/home/YY/Documents/FCN_registration_2D/"
    config = config_folder_guard({
        # train_parameters
        'image_size': [320, 320],
        'batch_size': 1,
        'learning_rate': 1e-3,
        'epoch_num': 500,
        'save_interval': 5,
        'shuffle_batch': True,
        # trainNet data folder
        'checkpoint_dir': root_dir + "checkpoints",
        'temp_dir': root_dir + "validate",
        'log_dir': root_dir + "log"
    })

    # 定义验证集和训练集
    train_x_dir = root_dir + "datasets/orient_1103/normolized_train"
    train_y_dir = root_dir + "datasets/orient_1103/resized_train"
    batch_x, batch_y = gen_batches(train_x_dir, train_y_dir, {
        'batch_size': config['batch_size'],
        'image_size': config['image_size'],
        'shuffle_batch': config['shuffle_batch']
    })
    valid_x_dir = root_dir + "datasets/orient_1103/normolized_validate"
    valid_y_dir = root_dir + "datasets/orient_1103/resized_validate"
    valid_x, valid_y = gen_batches(valid_x_dir, valid_y_dir, {
        'batch_size': config['batch_size'],
        'image_size': config['image_size'],
        'shuffle_batch': config['shuffle_batch']
    })
    config['train_iter_num'] = len(os.listdir(train_x_dir)) // config["batch_size"]
    config['valid_iter_num'] = len(os.listdir(valid_x_dir)) // config['batch_size']
    # config['train_iter_num'] = 100
    # config['valid_iter_num'] = 20

    # 定义日志记录器
    train_log = logger(config['log_dir'], 'train.log')
    valid_log = logger(config['log_dir'], 'valid.log')

    # 构建网络
    sess = tf.Session()
    reg = fcnRegressor(sess, True, config)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 开始训练
    print('start training')
    for epoch in range(config['epoch_num']):
        _train_L = []
        _train_L0 = []
        _train_L1 = []
        _train_L2 = []
        _train_L3 = []
        for i in range(config['train_iter_num']):
            _bx, _by = sess.run([batch_x, batch_y])
            _loss_train = reg.fit(_bx, _by)
            _train_L.append(_loss_train[0])
            _train_L0.append(_loss_train[1])
            _train_L1.append(_loss_train[2])
            _train_L2.append(_loss_train[3])
            _train_L3.append(_loss_train[4])
            print('[TRAIN] epoch={:>3d}, iter={:>5d}, loss={:.4f}, loss_0={:.4f}, loss_1={:.4f}, loss_2={:.4f}, loss_3={:.4f}'
                  .format(epoch + 1, i + 1, _loss_train[0], _loss_train[1], _loss_train[2], _loss_train[3], _loss_train[4]))
        train_log.info(
            '[TRAIN] epoch={:>3d}, loss={:.4f}, loss_0={:.4f}, loss_1={:.4f}, loss_2={:.4f}, loss_3={:.4f}'
            .format(epoch + 1, sum(_train_L) / len(_train_L), sum(_train_L0) / len(_train_L0), sum(_train_L1) / len(_train_L1),
                    sum(_train_L2) / len(_train_L2), sum(_train_L3) / len(_train_L3)))

        # 放入验证集进行验证
        _valid_L = []
        _valid_L0 = []
        _valid_L1 = []
        _valid_L2 = []
        _valid_L3 = []
        for j in range(config['valid_iter_num']):
            _valid_x, _valid_y = sess.run([valid_x, valid_y])
            # _loss_valid = reg.deploy(None, epoch, _valid_x, _valid_y)
            _loss_valid = reg.deploy(None, config['batch_size'], j, epoch, _valid_x, _valid_y)
            _valid_L.append(_loss_valid[0])
            _valid_L0.append(_loss_valid[1])
            _valid_L1.append(_loss_valid[2])
            _valid_L2.append(_loss_valid[3])
            _valid_L3.append(_loss_valid[4])
            print('[VALID] epoch={:>3d}, iter={:>5d}, loss={:.4f}, loss_0={:.4f}, loss_1={:.4f}, loss_2={:.4f}, loss_3={:.4f}'
                  .format(epoch + 1, j + 1, _loss_valid[0], _loss_valid[1], _loss_valid[2], _loss_valid[3], _loss_valid[4]))
            if (epoch + 1) % config['save_interval'] == 0:
                _valid_x, _valid_y = sess.run([valid_x, valid_y])
                reg.deploy(config['temp_dir'], config['batch_size'], j, epoch, _valid_x, _valid_y)
                reg.save(sess, epoch, config['checkpoint_dir'])

        valid_log.info(
            '[VALID] epoch={:>3d}, loss={:.4f}, loss_0={:.4f}, loss_1={:.4f}, loss_2={:.4f}, loss_3={:.4f}'
            .format(epoch + 1, sum(_valid_L) / len(_valid_L), sum(_valid_L0) / len(_valid_L0), sum(_valid_L1) / len(_valid_L1),
                    sum(_valid_L2) / len(_valid_L2), sum(_valid_L3) / len(_valid_L3)))

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train()
