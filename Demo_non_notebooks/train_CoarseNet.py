rom __future__ import absolute_import
from __future__ import division

import sys, os
sys.path.append(os.path.realpath('../CoarseNet'))

os.environ['KERAS_BACKEND'] = 'tensorflow'

from datetime import datetime
from MinutiaeNet_utils import *

from keras import backend as K
from keras.optimizers import SGD, Adam

from CoarseNet_utils import *
from CoarseNet_model import *

lr = 0.005

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
sess = K.tf.Session(config=config)
K.set_session(sess)

batch_size = 2
use_multiprocessing = False
input_size = 400

# Can use multiple folders for training
train_set = ['../Dataset/CoarseNet_train/',]
validate_set = ['../path/to/your/data/',]

pretrain_dir = '../Models/CoarseNet.h5'
output_dir = '../output_CoarseNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')
FineNet_dir = '../Models/FineNet.h5'
output_dir = '../output_CoarseNet/trainResults/' + datetime.now().strftime('%Y%m%d-%H%M%S')
logging = init_log(output_dir)
logging.info("Learning rate = %s", lr)
logging.info("Pretrain dir = %s", pretrain_dir)

train(input_shape=(input_size, input_size), train_set=train_set, output_dir=output_dir,
      pretrain_dir=pretrain_dir, batch_size=batch_size, test_set=validate_set,
      learning_config=Adam(lr=float(lr), beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=0.9),
      logging=logging)
