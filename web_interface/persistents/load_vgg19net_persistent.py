import tensorflow as tf
from lymph.vgg19 import net
from web_interface.test_main import app

loaded_flag = False
global_nets = None
global_mean_pixel = None
global_input_input_images = None


def load_vgg19net():
    # 加载vgg19-f网络
    if loaded_flag is False:
        global global_nets, global_mean_pixel, global_input_input_images, loaded_flag
        global_input_input_images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="input")
        global_nets, global_mean_pixel, _ = net(app.config['VGG_19_NET_FILE_PATH'], global_input_input_images)
        loaded_flag = True

    return global_nets, global_mean_pixel, global_input_input_images


def release_vgg19net():
    global global_nets, global_mean_pixel, global_input_input_images, loaded_flag
    del global_nets, global_mean_pixel, global_input_input_images
    global_nets, global_mean_pixel, global_input_input_images = [None, None, None]
    loaded_flag = False
