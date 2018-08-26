import tensorflow as tf
import numpy as np
import os
import random
import PIL.Image as Image
import cv2
import os
import copy
import time
import T3D
import DataGenerator
from settings import *

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--txt',type=str,default='./train.list')

args=parser.parse_args()


def compute_loss(logit,label):
    cross_entropy_mean=tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logit))
    weight_loss=tf.losses.get_regularization_loss()
    total_loss=cross_entropy_mean+weight_loss
    return total_loss

def compute_accuracy(logit,labels):
    correct=tf.equal(tf.argmax(logit,1),labels)
    acc=tf.reduce_mean(tf.cast(correct,tf.float32))
    return acc

def run():
	MODEL_PATH=''
	USE_PRETRAIN=False
	MAX_STEPS=5000
	dataloader=DataGenerator.DataGenerator(filename=args.txt,
                                batch_size=BATCH_SIZE,
                                num_frames_per_clip=NUM_FRAMES_PER_CLIP,
                                shuffle=True,is_da=IS_DA)
	
	
	with tf.Graph().as_default():
		global_step=tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
		input_placeholder=tf.placeholder(tf.float32,shape=[BATCH_SIZE,NUM_FRAMES_PER_CLIP,CROP_SIZE,CROP_SIZE,3])
		label_placeholder=tf.placeholder(tf.int64,shape=[BATCH_SIZE])

		logit=T3D.inference_t3d(input_placeholder)
		
		#define loss:
		loss=compute_loss(logit, label_placeholder)
		tf.summary.scalar('loss',loss)
	#     tf.summary.scalar('loss_total',total_loss)
		acc=compute_accuracy(logit,label_placeholder)
		tf.summary.scalar('accuracy',acc)
		
		#define lr dacay and optimizer:
		learning_rate = tf.train.exponential_decay(0.0002,
		                                           global_step,decay_steps=450,decay_rate=0.3,staircase=True)
		opt_stable=tf.train.AdamOptimizer(learning_rate)
		
		#build dependecy which updating paras before training:
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
		    optim_op=opt_stable.minimize(loss,global_step=global_step,var_list=tf.trainable_variables())
		
		saver=tf.train.Saver(tf.global_variables())
		init=tf.global_variables_initializer()
		config=tf.ConfigProto()
		gpu_options = tf.GPUOptions(allow_growth=True)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
		sess.run(init)
	   
		if USE_PRETRAIN:
		    saver.restore(sess,MODEL_PATH)
		    print('Checkpoint reloaded.')
		else:
		    print('Train from scratch.')
		merged=tf.summary.merge_all()
		#Using tensorboard to trace your training path.
		train_writer=tf.summary.FileWriter('./visual_t3d/train',sess.graph)
	#     test_writer=tf.summary.FileWriter('./visual_logs/test',sess.graph)
		
		duration=0
		print('Start training.')
		for step in xrange(1,MAX_STEPS):
		    
		    start_time=time.time()
		    train_images,train_labels,_=dataloader.next_batch()
		    sess.run(optim_op,feed_dict={
		                    input_placeholder:train_images,
		                    label_placeholder:train_labels})
		    duration+=time.time()-start_time
		    
		    if step!=0 and step % 10==0:
		        curacc,curloss=sess.run([acc,loss],feed_dict={
		                    input_placeholder:train_images,
		                    label_placeholder:train_labels})
		        print('Step %d: %.2f sec -->loss : %.4f =====acc : %.2f' % (step, duration,np.mean(curloss),curacc))
		        duration=0
		        
		    if step!=0 and step % 50==0:
		        mer=sess.run(merged,feed_dict={
		                    input_placeholder:train_images,
		                    label_placeholder:train_labels})
		        train_writer.add_summary(mer, step)
		        
		    if step >1000 and step % 500==0 or (step+1)==MAX_STEPS:
		        saver.save(sess,'./T3DBN_TFCKPT_ITER_{}'.format(step),global_step=step)
		    
		print('done')  

if __name__=='__main__':
	print('Preparing for training,this may take several seconds.')
	run()        
        
    


