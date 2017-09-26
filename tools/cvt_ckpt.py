#encoding:UTF-8

path = '/home/zhaoke/justrypython/text-detection-ctpn/checkpoints/'
OLD_CHECKPOINT_FILE = path + "model_final.ckpt"
NEW_CHECKPOINT_FILE = path + "model2_final.ckpt"

import tensorflow as tf
vars_to_rename = {
    "lstm_o/rnn/basic_lstm_cell/weights": "lstm_o/rnn/basic_lstm_cell/kernel",
    "lstm_o/rnn/basic_lstm_cell/biases" : "lstm_o/rnn/basic_lstm_cell/bias",
}
new_checkpoint_vars = {}
reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
for old_name in reader.get_variable_to_shape_map():
    print old_name
    if old_name in vars_to_rename:
        new_name = vars_to_rename[old_name]
    else:
        new_name = old_name
    new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars)

with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, NEW_CHECKPOINT_FILE)