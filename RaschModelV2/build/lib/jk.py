from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
from Utils.savedata import  create_csv
from Utils.openfile import openfiletest


from edward.models import Bernoulli, Normal

tf.contrib.distributions.kl = tf.contrib.distributions.kl_divergence

def main(_):
    # DATA
    X_data=openfiletest()
    # X_data=np.loadtxt(fname='inputs.txt',dtype=np.int32)
    # X_data = np.loadtext(fname='inputs.txt', dtype=np.float32)
    theta = Normal(loc=0.0, scale=1.0, sample_shape=[5000, 1])
    diff = Normal(loc=0.0, scale=1.0, sample_shape=[1, 21])
    X = Bernoulli(logits=theta - diff)

    mean_theta = Normal(loc=tf.get_variable("mean_theta/loc", [5000, 1]),
                                            scale=tf.nn.softplus(tf.get_variable("mean_theta/scale", [5000, 1])))
    mean_bi = Normal(loc=tf.get_variable("mean_bi/loc", [1, 21]),
                                         scale=tf.nn.softplus(tf.get_variable("mean_bi/scale", [1, 21])))

    inference = ed.KLqp({theta: mean_theta, diff: mean_bi}, data={X: X_data})
    inference.run(n_iter=5000, n_samples=21)
    print(mean_theta.eval())
    print(mean_bi.eval())
    th=mean_theta.eval()
    tb=mean_bi.eval()
    create_csv(th)
    create_csv(tb)





if __name__ == "__main__":
    tf.app.run()
