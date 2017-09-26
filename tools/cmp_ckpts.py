#encoding:UTF-8

import numpy as np

path = '/home/zhaoke/justrypython/text-detection-ctpn/checkpoints/'
OLD_CHECKPOINT_FILE = path + "model_final.ckpt"
NEW_CHECKPOINT_FILE = path + "model2_final.ckpt"

import tensorflow as tf
vars_to_rename = {
    "lstm_o/rnn/basic_lstm_cell/weights": "lstm_o/rnn/basic_lstm_cell/kernel",
    "lstm_o/rnn/basic_lstm_cell/biases" : "lstm_o/rnn/basic_lstm_cell/bias",
}
oldreader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
newreader = tf.train.NewCheckpointReader(NEW_CHECKPOINT_FILE)
oldnames = oldreader.get_variable_to_shape_map()
newnames = newreader.get_variable_to_shape_map()
if (len(oldnames) != len(newnames)):
    raise
for i in range(len(newnames)):
    oldname = oldnames.keys()[i]
    oldparas = oldreader.get_tensor(oldname)
    if oldname in newnames:
        newparas = newreader.get_tensor(oldname)
    else:
        newparas = newreader.get_tensor(vars_to_rename[oldname])
    if not np.all(newparas==oldparas):
        print 'False!'
    print np.all(newparas==oldparas)
    
#init = tf.global_variables_initializer()
#saver = tf.train.Saver(new_checkpoint_vars)

#with tf.Session() as sess:
    #sess.run(init)
    #saver.save(sess, NEW_CHECKPOINT_FILE)