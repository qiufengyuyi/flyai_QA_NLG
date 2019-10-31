# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from flyai.dataset import Dataset
from model import Model
from path import MODEL_PATH, LOG_PATH
from data_helper import *
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn
from flyai.utils.log_helper import train_log
from flyai.utils import remote_helper
from load_word_vec import load_wordvec
from customDecoder import TopKSampleEmbeddingHelper,TopPSampleEmbeddingHelper
from AdamWParameter import AdamWParameter
from AdamW import AdamOptimizer
'''
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''
# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper
path = remote_helper.get_remote_date('https://www.flyai.com/m/sgns.sogou.word.zip')
remote_helper.get_remote_data("https://www.flyai.com/m/sgns.sogou.word.zip")
word_path = os.path.splitext(path)[0]
# word_path = "sgns.sogou.word"
que_vec,ans_vec = load_wordvec(word_path)
'''
项目中的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
parser.add_argument("-vb", "--VAL_BATCH", default=64, type=int, help="val batch size")
args = parser.parse_args()
#  在本样例中， args.BATCH 和 args.VAL_BATCH 大小需要一致
'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.VAL_BATCH)
model = Model(dataset)

# 超参数
que_dict, ans_dict = load_dict()
encoder_vocab_size = len(que_dict)
decoder_vocab_size = len(ans_dict)
# Batch Size,
batch_size = 64
# RNN Size
rnn_size_list =[256,256,256]
# Number of Layers
num_layers = 3
# Embedding Size
encoding_embedding_size = 300
decoding_embedding_size = 300
# Learning Rate
learning_rate = 0.0008
decayed_learning_rate = learning_rate
dropout_rate = 0.7
attention_size = 256
decay_rate = 0.6
decay_steps = 2000


def gnmt_residual_fn(inputs, outputs):
  """Residual function that handles different inputs and outputs inner dims.
  Args:
    inputs: cell inputs, this is actual inputs concatenated with the attention
      vector.
    outputs: cell outputs
  Returns:
    outputs + actual inputs
  """
  def split_input(inp, out):
    out_dim = out.get_shape().as_list()[-1]
    inp_dim = inp.get_shape().as_list()[-1]
    return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)
  actual_inputs, _ = tf.contrib.framework.nest.map_structure(
      split_input, inputs, outputs)
  def assert_shape_match(inp, out):
    inp.get_shape().assert_is_compatible_with(out.get_shape())
  tf.contrib.framework.nest.assert_same_structure(actual_inputs, outputs)
  tf.contrib.framework.nest.map_structure(
      assert_shape_match, actual_inputs, outputs)
  return tf.contrib.framework.nest.map_structure(
      lambda inp, out: inp + out, actual_inputs, outputs)

class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, attention_cell, cells, use_new_attention=False):
    """Creates a GNMTAttentionMultiCell.
    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
      use_new_attention: Whether to use the attention generated from current
        step bottom layer's output. Default is False.
    """
    cells = [attention_cell] + cells
    self.use_new_attention = use_new_attention
    super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)

  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not tf.contrib.framework.nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    with tf.variable_scope(scope or "multi_rnn_cell"):
      new_states = []

      with tf.variable_scope("cell_0_attention"):
        attention_cell = self._cells[0]
        attention_state = state[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state)
        new_states.append(new_attention_state)

      for i in range(1, len(self._cells)):
        with tf.variable_scope("cell_%d" % i):

          cell = self._cells[i]
          cur_state = state[i]

          if self.use_new_attention:
            cur_inp = tf.concat([cur_inp, new_attention_state.attention], -1)
          else:
            cur_inp = tf.concat([cur_inp, attention_state.attention], -1)

          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)

    return cur_inp, tuple(new_states)


# 输入层
def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    dropout_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    is_train_flag = tf.placeholder(tf.bool,name="is_train")
    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    batch_size_feed = tf.placeholder(shape=(),dtype=tf.int32,name="batch_size")
    sampling_prob = tf.placeholder(shape=(),dtype=tf.float32,name="sampling_prob")
    lr_c = tf.placeholder(tf.float32,name='lr_c')
    wd_c = tf.placeholder(tf.float32,name='wd_c')

    return inputs, targets, learning_rate,dropout_prob,is_train_flag, target_sequence_length, max_target_sequence_length, source_sequence_length,batch_size_feed,sampling_prob,lr_c,wd_c

def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        if keep_prob is not None and is_train is not None:
            out = tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed), lambda: x)
            return out
        return x

# Encoder
"""
在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给RNN进行处理。
在Embedding中，我们使用tf.contrib.layers.embed_sequence，它会对每个batch执行embedding操作。
"""


def get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size,
                      encoding_embedding_size,dropout_keep_prob,is_train):
    """
    构造Encoder层
    参数说明：
    - input_data: 输入tensor
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    - source_sequence_length: 源数据的序列长度
    - source_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    """
    # encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)
    encoder_embed = tf.Variable(que_vec, name="encode_embed", dtype=tf.float32)
    encoder_embed_input = tf.nn.embedding_lookup(encoder_embed, input_data, name="encoder_embed_input")
    encoder_embed_input = dropout(encoder_embed_input,0.5,is_train,name="embed_dropout")

    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=100))
        lstm_cell = rnn.DropoutWrapper(lstm_cell,output_keep_prob=dropout_keep_prob)
        lstm_cell = rnn.ResidualWrapper(lstm_cell,residual_fn=gnmt_residual_fn)
        return lstm_cell

    cell_fw = [get_lstm_cell(rnn_size[i]) for i in range(num_layers)]
    cell_bw = [get_lstm_cell(rnn_size[i]) for i in range(num_layers)]
    # cell_fw = [rnn.DropoutWrapper(cell) for cell in cell_fw]
    # cell_bw = [rnn.DropoutWrapper(cell) for cell in cell_bw]
    encoder_output, encoder_state_fw,encoder_state_bw = \
        stack_bidirectional_dynamic_rnn(cell_fw,cell_bw, encoder_embed_input,
                                    sequence_length=source_sequence_length, dtype=tf.float32)

    # encoder_state_fw = tf.reduce_mean(encoder_state_fw, axis=1)
    # encoder_state_bw = tf.reduce_mean(encoder_state_bw, axis=1)
    W_c = tf.get_variable(
        'encoder_ELMo_Wc',
        shape=(num_layers,),  # [3]
        initializer=tf.zeros_initializer,
        trainable=True
    )
    W_h = tf.get_variable(
        'encoder_ELMo_Wh',
        shape=(num_layers,),  # [3]
        initializer=tf.zeros_initializer,
        trainable=True
    )
    normed_weights_c = tf.split(
        tf.nn.softmax(W_c + 1.0 / num_layers), num_layers
    )
    normed_weights_h = tf.split(
        tf.nn.softmax(W_h + 1.0 / num_layers), num_layers
    )
    state_h_list = []
    state_c_list = []
    for wc,wh, fw,bw in zip(normed_weights_c,normed_weights_h, encoder_state_fw,encoder_state_bw):
        state_c = tf.concat([fw.c,bw.c],1)
        state_h = tf.concat([fw.h,bw.h],1)
        state_c_list.append(wc*state_c)
        state_h_list.append(wh*state_h)
    encoder_state_h = tf.add_n(state_h_list)
    encoder_state_c = tf.add_n(state_c_list)

    # encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
    # encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
    encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

    # encoder_state = tf.concat([encoder_state_fw,encoder_state_bw],axis=-1)
    # encoder_state_pooling = tf.reduce_mean(encoder_state, axis=1)
    encoder_output = tf.concat(encoder_output, 2)
    # encoder_output = tf.layers.dense(encoder_output,rnn_size,activation="relu")
    encoder_output = dropout(encoder_output,dropout_keep_prob,is_train,name="encoder_rnn_output_dropout")
    # encoder_final_state = dropout(encoder_final_state, dropout_keep_prob, is_train, name="encoder_rnn_state_dropout")
    return encoder_output, encoder_state


def process_decoder_input(data, phonem_dict, batch_size):
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], phonem_dict['_sos_']), ending], 1)

    return decoder_input


def decoding_layer(phonem_dict, decoding_embedding_size, num_layers, rnn_size,dropout_keep_prob,is_train,
                   target_sequence_length, max_target_sequence_length, encoder_output,encoder_state, decoder_input,batch_size_feed):
    '''
    构造Decoder层
    参数：
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    '''
    # encoder_output = tf.transpose(encoder_output, [1, 0, 2])
    # 1. Embedding
    target_vocab_size = len(phonem_dict)
    # decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    # decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
    decoder_embeddings = tf.Variable(ans_vec, name="decoder_embed", dtype=tf.float32)
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input, name="decoder_embed_input")
    decoder_embed_input = dropout(decoder_embed_input,0.5,is_train,name="decoder_embed_dropout")
    # 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=100))
        return decoder_cell

    decoder_cell_list = [get_decoder_cell(256*2) for _ in range(3)]
    attention_cell = decoder_cell_list.pop(0)
    # decoder_cell = GNMTAttentionMultiCell(
    #       attention_cell, decoder_cell_list)
    # decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell)
    # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size*2)
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        256, encoder_output, memory_sequence_length=target_sequence_length, normalize=True)

    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(attention_cell, attention_mechanism,
                                                       attention_layer_size=attention_size)

    initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size_feed)
    initial_state = initial_state.clone(cell_state=encoder_state)

    # Output全连接层
    # target_vocab_size定义了输出层的大小
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1))

    # 4. Training decoder
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_embed_input,target_sequence_length,decoder_embeddings,sampling_probability=sampling_prob,time_major=False)
        # training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
        #                                                     sequence_length=target_sequence_length,
        #                                                     time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, training_helper, initial_state, output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length)

    # 5. Predicting decoder
    # 与training共享参数

    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([phonem_dict['_sos_']], dtype=tf.int32),
                               [tf.shape(target_sequence_length)[0]], name='start_token')
        # predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens,
        #                                                              phonem_dict['_eos_'])
        predicting_helper = TopPSampleEmbeddingHelper(decoder_embeddings,start_tokens,phonem_dict['_eos_'],0.9,batch_size=batch_size_feed)

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                             predicting_helper,
                                                             initial_state,
                                                             output_layer)
        predicting_decoder_output, _, _ = \
            tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                              impute_finished=True, maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output


# 上面已经构建完成Encoder和Decoder，下面将这两部分连接起来，构建seq2seq模型
def seq2seq_model(input_data, targets, target_sequence_length, max_target_sequence_length,
                  source_sequence_length, source_vocab_size, rnn_size, num_layers,dropout_keep_prob,is_train,batch_size_feed):
    encoder_output, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoding_embedding_size,dropout_keep_prob,is_train)

    decoder_input = process_decoder_input(targets, ans_dict, batch_size_feed)

    training_decoder_output, predicting_decoder_output = decoding_layer(ans_dict,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        dropout_keep_prob,
                                                                        is_train,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_output,
                                                                        encoder_state,
                                                                        decoder_input,batch_size_feed)

    return training_decoder_output, predicting_decoder_output


# 构造graph
train_graph = tf.Graph()
with train_graph.as_default():
    global_step = tf.Variable(0, name="global_step", trainable=False)
    input_data, targets, lr,dropout_keep_prob,is_train,target_sequence_length, max_target_sequence_length, source_sequence_length,batch_size_feed,sampling_prob,lr_c,wd_c = get_inputs()

    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len(que_dict),
                                                                       rnn_size_list,
                                                                       num_layers,dropout_keep_prob,is_train,batch_size_feed)

    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name="masks")

    # phone_accuracy
    logits_flat = tf.reshape(training_logits, [-1, decoder_vocab_size])
    predict = tf.cast(tf.reshape(tf.argmax(logits_flat, 1), [tf.shape(input_data)[0], -1]),
                      tf.int32, name='predict')
    corr_target_id_cnt = tf.cast(tf.reduce_sum(
        tf.cast(tf.equal(tf.cast(targets, tf.float32), tf.cast(predict, tf.float32)),
                tf.float32) * masks), tf.int32)
    ans_accuracy = corr_target_id_cnt / tf.reduce_sum(target_sequence_length)

    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
        optimizer = AdamOptimizer(learning_rate=lr_c, wdc=wd_c)
        # optimizer = tf.train.AdamOptimizer(lr)

        # 对var_list中的变量计算loss的梯度 该函数为函数minimize()的第一部分，返回一个以元组(gradient, variable)组成的列表
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
        train_op = optimizer.apply_gradients(capped_gradients)
        summary_op = tf.summary.merge([tf.summary.scalar("loss", cost)])


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
    all_steps = dataset.get_step()
    epho_step = int(all_steps/20)
    ap = AdamWParameter(nEpochs=args.EPOCHS,
                        LR=learning_rate,
                        weightDecay=0.005,
                        batchSize=args.BATCH,
                        nBatches=epho_step
                        )
    clr, wdc = ap.getParameter()
    accum_valid_loss = 0.00
    best_valid_loss = 1000.00
    sample_p = 0.0
    for step in range(dataset.get_step()):
        que_train, ans_train = dataset.next_train_batch()
        que_val, ans_val = dataset.next_validation_batch()
        # if step >= epho_step*10 and step % (epho_step*10) == 0:
        #     sample_p = (step/(epho_step*10)-9)/10
        # if step >= epho_step*8 and step % 500 == 0:
        #     decayed_learning_rate = learning_rate * (decay_rate ** int(step / (epho_step*8)))
        que_x, que_length = que_train
        ans_x, ans_lenth = ans_train
        ans_x = process_ans_batch(ans_x, ans_dict, int(sorted(list(ans_lenth), reverse=True)[0]))
        cur_batch_size = len(que_x)
        feed_dict = {input_data: que_x,
                     targets: ans_x,
                     lr: decayed_learning_rate,
                     dropout_keep_prob:dropout_rate,
                     is_train:True,
                     target_sequence_length: ans_lenth,
                     source_sequence_length: que_length,
                     batch_size_feed:cur_batch_size,
                     sampling_prob:sample_p,
                     lr_c:clr,
                     wd_c:wdc
                     }
        fetches = [train_op, cost, training_logits, ans_accuracy]
        _, tra_loss, logits, train_acc = sess.run(fetches, feed_dict=feed_dict)

        val_que_x, val_que_len = que_val
        val_ans_x, val_ans_len = ans_val
        val_ans_x = process_ans_batch(val_ans_x, ans_dict, int(sorted(list(val_ans_len), reverse=True)[0]))
        cur_batch_size = len(val_que_x)
        feed_dict = {input_data: val_que_x,
                     targets: val_ans_x,
                     lr: decayed_learning_rate,
                     dropout_keep_prob: dropout_rate,
                     is_train: False,
                     target_sequence_length: val_ans_len,
                     source_sequence_length: val_que_len,
                     batch_size_feed:cur_batch_size,
                     sampling_prob:sample_p,lr_c:clr,
                     wd_c:wdc}

        val_loss, val_acc = sess.run([cost, ans_accuracy], feed_dict=feed_dict)
        accum_valid_loss += val_loss

        summary = sess.run(summary_op, feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        # 调用系统打印日志函数，这样在线上可看到训练和校验准确率和损失的实时变化曲线
        train_log(train_loss=tra_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        if accum_valid_loss / (step+1) < best_valid_loss and step % 500 == 0:
            model.save_model(sess, MODEL_PATH, overwrite=True)
            best_valid_loss = accum_valid_loss / (step+1)
            # print("-----{}".format(best_valid_loss))
        # # 实现自己的保存模型逻辑
        # if step % epho_step  step % 200 == 0:
        #     model.save_model(sess, MODEL_PATH, overwrite=True)
    if (accum_valid_loss / all_steps) < best_valid_loss:
        model.save_model(sess, MODEL_PATH, overwrite=True)
