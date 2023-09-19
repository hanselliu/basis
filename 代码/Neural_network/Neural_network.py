import numpy as np
import math

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    c=np.max(x,axis=0,keepdims=True)  #此时x每一列是一个样本的输出结果，所以求最大要在axis=0
    exp_x=np.exp(x-c)
    sum_exp_x=np.sum(exp_x,axis=0,keepdims=True)
    return exp_x/sum_exp_x
#loss函数-----------------------------------------------------------
def mean_squared_error(y,t):
    return np.sum((y-t)**2)

#t为独热编码
def cross_entropy_error(y,t):
    delta=1e-7  #防止np.log自变量出现接近0的值
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[1]  #行数是节点数10 列数是样本数
    return -np.sum(t*np.log(y+delta))/batch_size  #所有都相加  


class Convolution:
    def __init__(self,w,b,stride=1,pad=0):
        self.w=w
        self.b=b
        self.stride=stride
        self.pad=pad
        
        self.img=None
        self.col=None
        self.col_w=None
        self.imgshape=None
    
    def forward(self,img):
        N, C, H, W = img.shape
        FN, C, FH, FW = self.w.shape
        self.img=img
        self.imgshape=img.shape
        OH=(H+2*self.pad-FH)//self.stride+1
        OW=(W+2*self.pad-FW)//self.stride+1
        img=np.pad(img,[(0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)],'constant')
        
        col=np.zeros((N,C,FH,FW,OH,OW))
        col_w=self.w.reshape(FN,-1).T#变成FN列 C*FH*FW行
        for y in range(FH):
            y_max=y+self.stride*OH
            for x in range(FW):
                x_max=x+self.stride*OW
                col[:,:,y,x,:,:]=img[:,:,y:y_max:self.stride,x:x_max:self.stride]  #每个x y对应共OH*OW*N*C个元素
        col=col.transpose(0,4,5,1,2,3).reshape(N*OH*OW,-1)#C*FH*FW列       
        
        out=np.dot(col,col_w)+self.b#N*OH*OW行 FN列
        out=out.reshape(N,OH,OW,FN).transpose(0,3,1,2)#转为数量 通道 行 列 N FN OH OWd
        
        self.col=col
        self.col_w=col_w
        return out
    
    def backward(self,delta):#delta为N FN OH OW  
        FN, C, FH, FW = self.w.shape
        N,FN,OH,OW=delta.shape
        delta=delta.transpose(0,2,3,1).reshape(-1,FN)#变成N*OH*OW FN
        
        db=np.sum(delta,axis=0)/N  #也就是在列上求sum 输出FN长度的一维数组
        dw=np.dot(self.col.T,delta)/N
        dw=dw.transpose(1,0).reshape(FN, C, FH, FW)
        
        dcol=np.dot(delta,self.col_w.T)  #N*OH*OW C*FH*FW
        #dx = col2im(dcol, self.xshape, self.pool_h, self.pool_w, self.stride, self.pad)
        N, C, H, W = self.imgshape
        #OH=(H+2*self.pad-FH)//self.stride+1
        #OW=(W+2*self.pad-FW)//self.stride+1
        dcol=dcol.reshape(N,OH,OW,C,FH,FW).transpose(0,3,4,5,1,2)
        delta_next=np.zeros((N,C,H+2*self.pad+self.stride-1,W+2*self.pad+self.stride-1))
        
        for y in range(FH):
            y_max=y+self.stride*OH
            for x in range(FW):
                x_max=x+self.stride*OW
                delta_next[:,:,y:y_max:self.stride, x:x_max:self.stride]+=dcol[:,:,y,x,:,:]
        return delta_next[:,:,self.pad:H+self.pad,self.pad:W+self.pad],dw,db


class MaxPooling:
    def __init__(self,size,pad=0):
        self.pool_size = size
        self.stride = size
        self.pad = pad
        self.img = None
        self.imgshape=None
        self.arg_max = None
    
    def forward(self,img):
        N, C, H, W = img.shape
        OH = H//self.pool_size
        OW = W//self.pool_size
        col=np.zeros((N,C,self.pool_size,self.pool_size,OH,OW))
        #img=x
        for y in range(self.pool_size):
            y_max=y+self.stride*OH
            for x in range(self.pool_size):
                x_max=x+self.stride*OW
                col[:,:,y,x,:,:]=img[:,:,y:y_max:self.stride,x:x_max:self.stride]
        col=col.transpose(0,4,5,1,2,3).reshape(N*OH*OW,-1)
        col=col.reshape(-1,self.pool_size*self.pool_size)
        
        out=np.max(col,axis=1)#m每一行的最大 
        arg_max=np.argmax(col,axis=1)#每一行最大的位置  大小为N*OH*OW*C
        out=out.reshape(N,OH,OW,C).transpose(0,3,1,2)#数量 通道 行 列
        #self.img = img
        self.imgshape=img.shape
        self.arg_max = arg_max
        return out
    
    def backward(self,delta):
        N,C,OH,OW=delta.shape
        delta=delta.transpose(0,2,3,1)#变成N,OH,OW,C
        pool_elem=self.pool_size*self.pool_size
        dmax=np.zeros((delta.size,pool_elem))#N*OH*OW*C行 pool_size**2列
        dmax[np.arange(self.arg_max.size),self.arg_max.flatten()]=delta.flatten()
        dmax=dmax.reshape(delta.shape+(pool_elem,))#相当于reshape(delta.shape[0],delta.shape[1],...,pool_elem)
        
        dcol=dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)#变成N*OH*OW行 C*FH*FW列
        
        N, C, H, W = self.imgshape
        #OH = H//self.pool_size
        #OW = W//self.pool_size
        dcol=dcol.reshape(N,OH,OW,C,self.pool_size,self.pool_size).transpose(0,3,4,5,1,2)#与col维度相同
        delta_next=np.zeros((N,C,H+self.stride-1,W+self.stride-1))
        
        for y in range(self.pool_size):
            y_max=y+self.stride*OH
            for x in range(self.pool_size):
                x_max=x+self.stride*OW
                delta_next[:,:,y:y_max:self.stride,x:x_max:self.stride]+=dcol[:,:,y,x,:,:]
        return delta_next[:,:,:H,:W]


class Affine:
    def __init__(self,w,b):
        self.w=w
        self.b=b
        self.x=None
        self.original_x_shape = None
        #self.dw=None
        #self.db=None
    def forward(self,x):  #x行为输入节点数，列为样本数N
        self.original_x_shape = x.shape
        x=x.reshape(x.shape[0],-1)#可能有时候输入的是一维向量 在这里确保为二维 二维输入则没有变换
        self.x=x
        return np.dot(self.w,x)+self.b  #b行数是输出节点数 但有一列。输出维度是输出节点数行 样本数列
    def backward(self,delta):#输入的delta是输出节点数行 样本数列
        dw=np.dot(delta,self.x.T)/self.x.shape[1]
        db=np.sum(delta,axis=1,keepdims=True)/delta.shape[1]
        delta_next=np.dot(self.w.T,delta)
        return delta_next,dw,db
    

class SoftmaxWithCrossEntropy:#通常最后一层
    def __init__(self):
        self.y=None
        self.t=None
        self.loss=None
    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
        return self.y,self.loss
    def backward(self,dout=1):
        batch_size=self.y.shape[1]
        delta=self.y-self.t  #10行N列
        return delta

class Relu:
    def __init__(self):
        self.mask=None
    def forward(self,x):
        self.mask=(x<=0) #保持x形状 满足条件为True,不满足为False
        out=x.copy()    #复制x
        out[self.mask]=0  #坐标为True处为0
        return out
    def backward(self,delta):
        delta[self.mask]=0  
        return delta

class Sigmoid():
    def __init__(self):
        self.out=None
    def forward(self,x):
        
        self.out=1.0/(np.exp(-x)+1.0)
        return self.out
    def backward(self,dout):
        return dout*self.out*(1.0-self.out)
