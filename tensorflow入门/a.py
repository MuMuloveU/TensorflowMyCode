# -*- coding: utf-8 -*-
#code 3-1 线性回归
# 第一步准备数据
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

#生成模拟数据
train_X = np.linspace(-1, 1, 100)
# y=2x 但是加入了噪声
train_Y = 2 * train_X + np.random.rand(*train_X.shape) * 0.3

#显示模拟数据点
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()


# 第二步 创建模型
#占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
#模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

#前后结构
z = tf.multiply(X, W) + b

# 第二步续 创建反向模型
#反向优化
#cost 为生成值与真实值的平方差
cost = tf.reduce_mean(tf.square(Y - z))
#学习率 调整参数的速度 这个参数一般是小于1的，值越大，代表调整速度越大，但不精确；值越小，精度越高，但是速度更慢。
learning_rate = 0.01
#梯度下降算法（封装好的）learning_rate 为学习率
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# 第三步 迭代训练模型
#初始化所有变量
init =tf.global_variables_initializer()
#定义参数 
training_epochs = 20
display_step = 2

#启动session
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X : x,Y :y})
        #显示训练中的详细信息
        if epoch % display_step == 0 : 
            loss = sess.run(cost,feed_dict={X : train_X,Y : train_Y})
            print ("Epoch:",epoch + 1,"cost=",loss, "w=",sess.run(W),"b=",sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
        
    print (" Finished! ")
    print ("cost=",sess.run(cost,feed_dict={X : train_X,Y:train_Y}),"w=", sess.run(W),"b=",sess.run(b))
    

    


