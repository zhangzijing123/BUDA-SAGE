

import argparse
import logging
import os
import shutil
import time

import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow.image import ssim
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg_preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Activation, Input, Conv2D, Conv2DTranspose, add, LeakyReLU, Layer, Lambda, \
    BatchNormalization
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
RANDOM_SEED = 28


def create_parser():
    parser = argparse.ArgumentParser(description='Pipeline to run SR training ... ')
    parser.add_argument('--uid', type=str,
                        help="sequence id for the experiment")
    parser.add_argument('--d', type=str, help="#id GPU device")
    parser.add_argument('--verbose', action='store_true', dest='verbose',
                        help='verbose')
    return parser


def residual_block(x, filter_size, act_type, scale_factor, blk_num, init_seed, isBN=False):
    initializer = initializers.glorot_uniform(init_seed)
    tmp = Conv2D(filter_size, kernel_size=(3, 3),
                 kernel_initializer=initializer,
                 padding='same',
                 name=('resblock_%i_conv_1' % blk_num))(x)
    if isBN:
        tmp = BatchNormalization()(tmp)

    if act_type == "LeakyReLU":
        tmp = LeakyReLU(alpha=0.2)(tmp)
    else:
        tmp = Activation(act_type, name=('resblock_%i_act' % blk_num))(tmp)

    tmp = Conv2D(filter_size, kernel_size=(3, 3),
                 kernel_initializer=initializer,
                 padding='same',
                 name=('resblock_%i_conv_2' % blk_num))(tmp)

    if isBN:
        tmp = BatchNormalization()(tmp)

    tmp = Lambda(lambda x: x * scale_factor)(tmp)

    out = add([x, tmp])

    return out


def model(img_row=512, img_col=512, img_channel=2, layers=32,
          features=64, img_out_channel=1,
          act_type='relu', scale_factor=0.1, init_seed=1337):
    initializer = initializers.glorot_uniform(init_seed)
    inputs = Input((img_row, img_col, img_channel))

    input_conv = Conv2DTranspose(filters=4, kernel_size=1,
                                 strides=(1, 1),
                                 kernel_initializer=initializer,
                                 padding='valid')(inputs)

    x = Conv2D(features, kernel_size=(3, 3),
               kernel_initializer=initializer,
               padding='same',
               name=('conv0'))(input_conv)
    conv_1 = x

    for blk_num in range(layers):
        x = residual_block(x, features, act_type, scale_factor, blk_num,
                           init_seed)

    x = Conv2D(features, kernel_size=(3, 3),
               kernel_initializer=initializer,
               padding='same',
               name=('conv_penultimate'))(x)

    x = add([x, conv_1])

    output = Conv2D(img_out_channel, kernel_size=(3, 3),
                    kernel_initializer=initializer,
                    padding='same',
                    name=('conv_final'))(x)

    model = Model(inputs=[inputs], outputs=[output])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    model.compile(loss=mix_loss, optimizer=adam)

    return model


class ConstMultiplierLayer(Layer):
    def __init__(self, val=0.1, **kwargs):
        self.val = val
        super(ConstMultiplierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.k = self.add_weight(
            name='k',
            shape=(),
            initializer=Constant(value=self.val),
            dtype='float32',
            trainable=True,
        )
        super(ConstMultiplierLayer, self).build(input_shape)

    def call(self, x):
        return tensorflow.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape


def mix_loss(y_true, y_pred):
    return 0.5 * ssim_loss(y_true, y_pred) + 0.5 * mean_absolute_error(y_true, y_pred)


def ssim_loss(y_true, y_pred):
    reverse_true = tf.subtract(K.max(K.flatten(y_true)), y_true)
    reverse_pred = tf.subtract(K.max(K.flatten(y_pred)), y_pred)

    return (1 - K.mean(
        ssim(y_true, y_pred, K.max(K.flatten(y_true))))) / 2 + \
           (1 - K.mean(
               ssim(reverse_true, reverse_pred, K.max(K.flatten(reverse_true))))) / 2


def perceptual_loss(y_true, y_pred):
    y_true_3c = K.concatenate([y_true, y_true, y_true])
    y_pred_3c = K.concatenate([y_pred, y_pred, y_pred])

    y_true_3c = vgg_preprocess(y_true_3c)
    y_pred_3c = vgg_preprocess(y_pred_3c)

    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(560, 560, 2))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true_3c) - loss_model(y_pred_3c)))


def exact_loss(sag_input, ax_input):
    _, nrow, ncol, nchan, _ = sag_input.get_shape()

    def loss(y_true, y_pred):
        return tf.math.reduce_mean([mix_loss(ax_input[:, nc, :, :, :], y_pred[:, nc, :, :, :]) for nc in range(nrow)]) + \
               tf.math.reduce_mean([mix_loss(y_true[:, :, nc, :, :], y_pred[:, :, nc, :, :]) for nc in range(ncol)]) + \
               tf.math.reduce_mean([mix_loss(sag_input[:, :, :, nc, :], y_pred[:, :, :, nc, :]) for nc in range(nchan)])
    return loss


def set_up_gpu(gpu_device, verbose):
    if verbose:
        logging.info("Applying GPU: " + gpu_device)

    # setup the GPU
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device


def set_up_random_seed(seed, verbose):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    # the set numpy seed before importing keras
    import numpy as np
    np.random.seed(seed)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.set_random_seed(seed)

    if verbose:
        logging.info("set up the random seed.")


def setup_train(model, uuid, zip_path, filepath):
    model_json = model.to_json()
    model_filename = uuid + "_model.json"
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)
    shutil.move(model_filename, os.path.join(zip_path, model_filename))

    cp_name = os.path.join(zip_path, uuid + "{epoch:02d}.ckpt")
    csv_name = os.path.join(zip_path, uuid + ".csv")

    # set up the metrics list
    csv_cb = CSVLogger(csv_name)

    # set up the checkpoint
    ckpt_cb = ModelCheckpoint(cp_name, monitor='val_loss', save_best_only=False,
                              save_weights_only=False, period=3)

    callbacks = []
    callbacks.append(ckpt_cb)
    callbacks.append(csv_cb)
    fit_helper(model, filepath, callbacks)


def fit_helper(model, filepath, callbacks):
    logging.info("start to train")
    ins_all, outs_all = [], []
    for filename in os.listdir(filepath):
        print('loading -', filename)
        tic = time.time()
        npfile = np.load(os.path.join(filepath, filename))
        ins, outs = npfile['inputs'], npfile['outputs']
        print ('ins-', ins.shape, 'outs-', outs.shape)
        ins_all.append(ins)
        outs_all.append(outs)
        print('takes', time.time() - tic)

    ins_all = np.concatenate(ins_all, axis=0)[..., np.newaxis]
    outs_all = np.concatenate(outs_all, axis=0)[..., np.newaxis]
    model.fit(x=ins_all,
              y=outs_all,
              batch_size=4,  # train_config["batch_size"]
              epochs=30,  # train_config["epochs"]
              verbose=True,
              shuffle=True,
              validation_data=(ins_all[:20], outs_all[:20]),
              callbacks=callbacks)
    return model


def set_up_model():
    tic = time.time()
    m = model(img_row=220, img_col=220, img_channel=9, layers=10,
              features=64)
    print('time=', time.time() - tic)
    return m


def set_up_folder(zip_path, verbose=True):
    if not os.path.exists(zip_path):
        try:
            os.makedirs(zip_path)
            if verbose:
                logging.info(
                    "Creating a new folder to store the yaml and model files" \
                    + zip_path)
            return zip_path

        except OSError as exc:  # Guard against race condition
            raise FileExistsError
    else:
        raise FileExistsError


def run():
    # set args
    logging.info("initialize parser.")
    args = create_parser().parse_args()

    # set up the random seed
    logging.info("set up the random seed and verbose")
    set_up_random_seed(RANDOM_SEED, args.verbose)

    # set up gpu
    logging.info("set up GPU")
    set_up_gpu(args.d, args.verbose)

    # set up the model
    logging.info("set up model")
    model = set_up_model()

    # set up folder
    logging.info("set up folder")
    zip_path = set_up_folder(str(args.uid))

    # run training
    logging.info("set up training")
    setup_train(model, args.uid, zip_path, '/data/long/data/data_spvsd/norm')

    # clear the sessions
    K.clear_session()


if __name__ == "__main__":
    run()

