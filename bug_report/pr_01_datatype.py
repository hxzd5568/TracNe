# TRACNE located error: https://github.com/apache/tvm/pull/14307

#####################################
###### a data type bug
#####################################

import tensorflow as tf

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf
import os
import tensorflow as tf
import tvm
from tvm import te
from tvm import relay

import numpy as np
import os.path
import tvm.relay.testing.tf as tf_testing

target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from tensorflow.python.framework import graph_util
import tvm.testing


import numpy as np
import pickle
import math
import os

if not os.path.exists("attacksample_id9"):
    os.makedirs("attacksample_id9")
import time


tf.compat.v1.disable_eager_execution()


bs = 100
x = tf.placeholder("float32", [1, 2])  # mnist data image of shape 28*28=784
y = tf.placeholder("float32", [1, 5])
input = pickle.load(open("./study_case_ID_3/input/dict.txt", "rb"))
input32 = pickle.load(open("./study_case_ID_3/input/dict32.txt", "rb"))
input = input32

learning_rate = 0.005
training_epochs = 25
batch_size = 100
display_step = 1


W = tf.Variable(tf.zeros([2, 5]), name="w")
b = tf.Variable(tf.zeros([5]), name="b")

activation = tf.nn.softmax(tf.matmul(x, W) + b, name="softmaxgood")  # Softmax

cost = -tf.reduce_sum(y * tf.log(activation), name="reducegood")  # Cross entropy

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # Gradient Descent
saver = tf.train.Saver()
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, "./pr_01/dmodel.ckpt")
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    saver = tf.train.import_meta_graph(
        "./pr_01/dmodel.ckpt" + ".meta", clear_devices=True
    )
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    saver.restore(sess, "./pr_01/dmodel.ckpt")
    output_node_names = "softmaxgood"
    graph_def = graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        output_node_names.split(","),
    )
mod, params = relay.frontend.from_tensorflow(graph_def)
print(mod["main"])


input_x = np.random.normal(size=(1, 2))
dummy_x = np.int32(input_x * 1e2)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
from tvm.contrib.debugger.debug_executor import GraphModuleDebug

paras = "./pr01"
m = GraphModuleDebug(
    lib["debug_create"]("default", dev),
    [dev],
    lib.graph_json,
    dump_root=paras,
)

m.set_input("Placeholder", tvm.nd.array(dummy_x))
m.set_input(**params)

m.run()

paras = "./pr01"
input_x = np.random.normal(size=(1, 2))
dummy_x = np.int32(input_x)
tvm_out = m.get_output(0, tvm.nd.empty(((1, 5)), "float32")).numpy()
data = relay.load_param_dict(
    bytearray(open(paras + "/_tvmdbg_device_CPU_0/output_tensors.params", "rb").read())
)
for k, v in data.items():
    if "Placeholder" in k:
        tvm_array = v.numpy()
        print(tvm_array, tvm_array.shape)

tvm.testing.assert_allclose(dummy_x, tvm_array, 1e-5)