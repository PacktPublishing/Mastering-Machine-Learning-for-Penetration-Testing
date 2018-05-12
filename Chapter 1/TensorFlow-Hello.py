# Display "Hello, world!"
import tensorflow as tf
Message = tf.constant("Hello, world!")
sess = tf.Session()
print(sess.run(Message))
