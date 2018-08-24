# IPython log file


import tensorflow as tf
get_ipython().set_next_input(u'a = tf.VariableScope');get_ipython().magic(u'pinfo tf.VariableScope')
get_ipython().set_next_input(u'a = tf.Variable');get_ipython().magic(u'pinfo tf.Variable')
a = tf.constant([1,1,2])
a
b = tf.constant([2,3,4])
saver = tf.train.Saver()
w = tf.Variable(tf.random_normal([10,10]))
b = tf.Variable(tf.random_normal([10,10]))
z = tf.add(w,b)
saver = tf.train.Saver()
sess = tf.InteractiveSession()
save_path = saver.save(sess, "TF_save_file")
sess.run(tf.initialize_all_variables())
save_path = saver.save(sess, "TF_save_file")
quit()
