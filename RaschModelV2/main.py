import edward as ed
import tensorflow as tf
from Utils.savedata import  create_csv
from Utils.openfile import openfiletest


from edward.models import Bernoulli, Normal

tf.contrib.distributions.kl = tf.contrib.distributions.kl_divergence

def main(_):
    # DATA
    X_data=openfiletest()
    theta = Normal(loc=0.0, scale=1.0, sample_shape=[5000, 1])
    diff = Normal(loc=0.0, scale=1.0, sample_shape=[1, 21])
    X = Bernoulli(logits=theta - diff)

    mean_theta = Normal(loc=tf.get_variable("mean_theta/loc", [5000, 1]),
                                            scale=tf.nn.softplus(tf.get_variable("mean_theta/scale", [5000, 1])))
    mean_bi = Normal(loc=tf.get_variable("mean_bi/loc", [1, 21]),
                                         scale=tf.nn.softplus(tf.get_variable("mean_bi/scale", [1, 21])))

    inference = ed.KLqp({theta: mean_theta, diff: mean_bi}, data={X: X_data})
    inference.run(n_iter=5000, n_samples=21)
    print("theta")
    print(mean_theta.eval())
    print("difficulties:")
    print(mean_bi.eval())
    th=mean_theta.eval()
    tb=mean_bi.eval()
    print("create csv file to save theta")
    create_csv(th)
    print("create csv file to save difficulties:")
    create_csv(tb)





if __name__ == "__main__":
    tf.app.run()
