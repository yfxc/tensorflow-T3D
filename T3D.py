import tensorflow as tf
import tensorflow.contrib.slim as slim
from settings import *

#Define base convolution layers for 'pseudo-3d convolution'(P3D):
def convS(_X,out_channels,kernel_size=[1,3,3],stride=1,padding='VALID'):
    return slim.conv3d(_X,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                   biases_initializer=None)

def convT(_X,out_channels,kernel_size=[3,1,1],stride=1,padding='VALID'):
    return slim.conv3d(_X,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,
                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       biases_initializer=None)

# def ST_A(_X,out_channels,s_kernel,t_kernel,stride,padding):
#     x=convS(_X,out_channels,s_kernel,stride,padding)
#     x=tf.layers.batch_normalization(x,training=IS_TRAIN)
#     x=tf.nn.relu(x)
#     x=convT(x,out_channels,t_kernel,stride,padding)
#     x=tf.layers.batch_normalization(x,training=IS_TRAIN)
#     x=tf.nn.relu(x)
#     return x
    
def ST_B(_X,out_channels,s_kernel,t_kernel,stride,padding):
    
   
    tmp_x=convS(_X,out_channels,s_kernel,stride,padding)
    tmp_x=tf.layers.batch_normalization(tmp_x,training=IS_TRAIN)
    tmp_x=tf.nn.relu(tmp_x)
    
    x=convT(_X,out_channels,t_kernel,stride,padding)
    x=tf.layers.batch_normalization(x,training=IS_TRAIN)
    x=tf.nn.relu(x)
    
    return x+tmp_x
    
def ST_C(_X,out_channels,s_kernel,t_kernel,stride,padding):
    
    x=convS(_X,out_channels,s_kernel,stride,padding)
    x=tf.layers.batch_normalization(x,training=IS_TRAIN)
    x=tf.nn.relu(x)
    
   
    tmp_x=convT(x,out_channels,t_kernel,stride,padding)
    tmp_x=tf.layers.batch_normalization(tmp_x,training=IS_TRAIN)
    tmp_x=tf.nn.relu(tmp_x)
    
    return x+tmp_x
def p3d(_X,out_channels,kernel_size=[3,3,3],stride=1,padding='SAME',tp='B'):
    s_kernel=kernel_size[1:3]
    s_kernel.insert(0,1)
    t_kernel=[1,1]
    t_kernel.insert(0,kernel_size[0])
    if tp=='B':
        return ST_B(_X,out_channels,s_kernel,t_kernel,stride,padding)
    else:
        return ST_C(_X,out_channels,s_kernel,t_kernel,stride,padding)

def make_block_layer(_X,nums_inchannel,keep_prob=1,tp_idx=0):
    #Caution: Do Not use tf.contrib.layers.batch_norm() !!!
    out=tf.layers.batch_normalization(_X,training=IS_TRAIN)
    out=tf.nn.relu(out)
    out=slim.conv3d(out,BN_SIZE*GROWTH_RATE,[1,1,1],stride=1,biases_initializer=None)
    #Here,we use 'pseudo-3d convolution' instead of traditional 3D convolution(eg. slim.conv3d)
    if tp_idx % 2==0:
        out=p3d(out,GROWTH_RATE,padding='SAME',tp='B')
    else:
        out=p3d(out,GROWTH_RATE,padding='SAME',tp='C')
        #out=slim.conv3d(out,GROWTH_RATE,[3,3,3],stride=1,padding='SAME',biases_initializer=None)
    if(keep_prob!=1):
        out=slim.dropout(out,keep_prob=keep_prob)
    return tf.concat([_X,out],axis=-1)

def build_block(_X,block_num,nums_inchannel):
    tp_idx=0
    for i in range(block_num):
        _X=make_block_layer(_X,nums_inchannel=nums_inchannel,keep_prob=KEEP_PROB,tp_idx=tp_idx)
        tp_idx+=1
    return _X

def TTL(_X,depth=(1,3,4)):
    #As mentioned above,I replace traditional 3D conv by 'pseudo-3d conv' to achive parameters efficiency.
    y1=tf.layers.batch_normalization(_X,training=IS_TRAIN)
    y1=tf.nn.relu(y1)
    y1=slim.conv3d(y1,128,[depth[0],1,1],stride=1,biases_initializer=None) 
    
    
#     y2=tf.layers.batch_normalization(_X,training=IS_TRAIN)
#     y2=tf.nn.relu(y2)
    y2=p3d(_X,128,[depth[1],3,3],stride=1,tp='B')
    #y2=slim.conv3d(y2,128,[depth[1],3,3],stride=1,biases_initializer=None)
    
#     y3=tf.layers.batch_normalization(_X,training=IS_TRAIN)
#     y3=tf.nn.relu(y3)
    y3=p3d(_X,128,[depth[2],3,3],stride=1,tp='C')
    #y3=slim.conv3d(y3,128,[depth[2],3,3],stride=1,biases_initializer=None)
    
#     y4=tf.layers.batch_normalization(_X,training=IS_TRAIN)
#     y4=tf.nn.relu(y4)
    
    return tf.concat([y1,y2,y3],axis=-1)

def Transition(_X,in_channels):
    _X=tf.layers.batch_normalization(_X,training=IS_TRAIN)
    _X=tf.nn.relu(_X)
    _X=slim.conv3d(_X,in_channels,kernel_size=[1,1,1],stride=1,biases_initializer=None)
    _X=slim.avg_pool3d(_X,[2,2,2],stride=2,padding='SAME')
    return _X

def inference_t3d(_X,block_config=(6, 12, 24, 16)):
    
    with slim.arg_scope([slim.conv3d],
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)): 
        out=slim.conv3d(_X,START_CHANNEL,[3,7,7],stride=[1,2,2],padding='SAME',biases_initializer=None)
       
        out=tf.layers.batch_normalization(out,training=IS_TRAIN)
        out=tf.nn.relu(out)
        out=slim.max_pool3d(out,kernel_size=[3,3,3],stride=2,padding='SAME')
        in_channels=START_CHANNEL
        
        for i, num_layers in enumerate(block_config):
            
            out=build_block(out,num_layers,in_channels)
            
            in_channels=in_channels+GROWTH_RATE*num_layers
            if i!=len(block_config)-1:
                if i==0:
                    out=TTL(out,(1,3,6))
                else:
                    out=TTL(out)
                
                in_channels=128*3
                out=Transition(out,in_channels // 2)
                in_channels=in_channels // 2
                
                
        
        out=tf.layers.batch_normalization(out,training=IS_TRAIN)
        out=tf.nn.relu(out)
        #Standard input shape=[BATCH,NUM_CLIP=16,HEIGHT=160,WIDTH=160,RGB=3],makes that kernel_size of AVG_POOL equals '5'
        #If you are about to change size of input,changing the kernel size of 'avg_pool3d' simultaneously.
        out=slim.avg_pool3d(out,kernel_size=[1,5,5])
        out=tf.reshape(out,[out.get_shape().as_list()[0],-1])
        
        out=slim.fully_connected(out,NUM_CLASSES)
        return out

