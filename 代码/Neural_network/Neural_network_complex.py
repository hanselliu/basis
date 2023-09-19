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
    def __init__(self,w,b,stride_w=1,stride_h=1,pad=0,dtype=float):
        self.dtype=dtype
        self.w=w
        self.b=b
        self.stride_w=stride_w
        self.stride_h=stride_h
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
        OH=(H+2*self.pad-FH)//self.stride_h+1
        OW=(W+2*self.pad-FW)//self.stride_w+1
        img=np.pad(img,[(0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)],'constant')
        col=np.zeros((N,C,FH,FW,OH,OW),self.dtype)
        col_w=self.w.reshape(FN,-1).T#变成FN列 C*FH*FW行
        for y in range(FH):
            y_max=y+self.stride_h*OH
            for x in range(FW):
                x_max=x+self.stride_w*OW
                col[:,:,y,x,:,:]=img[:,:,y:y_max:self.stride_h,x:x_max:self.stride_w]  #每个x y对应共OH*OW*N*C个元素
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
        delta_next=np.zeros((N,C,H+2*self.pad+self.stride_h-1,W+2*self.pad+self.stride_w-1),self.dtype)
        
        for y in range(FH):
            y_max=y+self.stride_h*OH
            for x in range(FW):
                x_max=x+self.stride_w*OW
                delta_next[:,:,y:y_max:self.stride_h, x:x_max:self.stride_w]+=dcol[:,:,y,x,:,:]
        return delta_next[:,:,self.pad:H+self.pad,self.pad:W+self.pad],dw,db


class Hermite_Measure:
    def __init__(self,n,n_meas,w1,w2):  # n粒子数  n个粒子每个的测量次数输入一个长度n的一维数组
        self.n=n                       #w1是优化参数一个长度n的列表,w1=[(n_mea_1,4,1,1),(n_mea_2,4,1,1),...(n_mea_n,4,1,1)]
        self.w1=w1
        self.w2=w2                     #w2是最后所有测量量出来再乘的一个系数（优化参数） 要求一维数组
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,0-1j],[0+1j,0]])
        self.Z = np.array([[1,0],[0,-1]])
        self.E = np.array([[1,0],[0,1]])
        self.base=np.zeros((4,2,2),complex)
        self.base[0,:,:]=self.E.T       # 注意一定要转置！！！
        self.base[1,:,:]=self.X.T
        self.base[2,:,:]=self.Y.T
        self.base[3,:,:]=self.Z.T
        
        self.Base=[np.ones([n_mea,4,2,2],complex) for n_mea in n_meas]
        for i in range(self.n):
            #print(self.Base[i].shape)
            self.Base[i]=self.Base[i]*self.base
        #print(self.Base)
    
    def forward(self,sigma):#sigma要求 （N,H,W）
        self.N=sigma.shape[0]
        Base_w=[self.Base[parti]*self.w1[parti] for parti in range(self.n)]#每个元素（n_mea,4,2,2）
        self.ConV=[] #层所在的列表
        self.conv=[sigma] #（N H W）
        self.conv_Out=[]
        self.I_cor=1  #针对全空间单位矩阵的修正，最后结果要减去这个数组的sum
        for particle in range(self.n):
            self.ConV.append(Convolution(Base_w[particle],b=np.array([0]),stride_w=2,stride_h=2,dtype=complex))
            conv_in=np.ones((4,self.conv[-1].shape[0],self.conv[-1].shape[1],self.conv[-1].shape[2]),complex)\
            *self.conv[-1]  #（4 N H W）
            conv_in=conv_in.transpose(1,0,2,3)  #（N 4 H W）
            conv_out=self.ConV[-1].forward(conv_in)  #（N 4 H W）*（n_mea 4 2 2）=(N n_mea H/2 W/2)
            self.conv_Out.append(conv_out)
            self.conv.append(conv_out.reshape(conv_out.shape[0]*conv_out.shape[1],conv_out.shape[2],conv_out.shape[3]))
            # 上一行变为(N n_mea H/2 W/2)===>（N*n_mea,H/2 W/2）
            self.I_cor=np.kron(self.I_cor,self.w1[particle][:,0,:,:].reshape(-1))

        out=self.conv[-1] #最后输出的一定是（N*n_mea_1*n_mea_2*... ,1,1）
        #out=np.dot(self.conv[-1].reshape(-1),self.w2)-np.sum(I_cor)
        return np.real(out)
    
    def backward(self,delta): #delta形状(N*n_mea_1*n_mea_2*... ,1,1) 写作(N*...*n_mea_N ,1,1)
        dW1=[]  #要求元素和dw1=[]的元素相同 
        for particle in range(1,self.n+1):
            #针对全空间单位矩阵的修正
            dw_Icor=1
            for parti in range(1,self.n+1):
                if -parti != -particle:
                    dw_Icor=np.kron(dw_Icor,self.w1[-parti][:,0,:,:].reshape(-1))
            dw_Icor=np.sum(dw_Icor)
            
            #delta的反向传递
            delta=delta.reshape(self.conv_Out[-particle].shape)#变成（N*...*,n_mea,H/2,W/2）
            delta,dBase,db=self.ConV[-particle].backward(delta)#(N*...*,4,H,W)
            
            dBase=dBase*delta.shape[0]  #因为ConV.backward求梯度时已经/N了 但此N非self.N，

            dw1=np.zeros(self.w1[-particle].shape)
            for n_mea in range(dw1.shape[0]):
                dw1[n_mea,0,0,0]=0.5*np.real(dBase[n_mea,0,0,0]+dBase[n_mea,0,1,1])/self.N #-dw_Icor  # dI
                dw1[n_mea,1,0,0]=np.real(dBase[n_mea,0,0,1])/self.N                         # dX
                dw1[n_mea,2,0,0]=np.imag(dBase[n_mea,0,0,1])/self.N                         # dY
                dw1[n_mea,3,0,0]=0.5*np.real(dBase[n_mea,0,0,0]-dBase[n_mea,0,1,1])/self.N  #dZ
            dW1.append(dw1)
            
            delta=np.sum(delta,axis=1,keepdims=True).transpose(1,0,2,3) #(N*...*,4,H,W)=>(N*..*,1,H,W)=>(1,N*..*,H,W)
        
        dW1.reverse()#颠倒顺序
        #sys.exit(0)
        return delta,dW1


class Project_Measure_Product_complex():  #  类型1这个结构必须复参  
    def __init__(self,w,dtype=complex):
        self.dtype=dtype
        self.w=w #代表当前粒子的测量设置(C,FH,FW)
        self.stride_w=w.shape[-1]
        self.stride_h=1
        
    def forward(self,img):
        self.img=img
        self.img_shape=img.shape
        N,C,H,W=img.shape
        C,FH,FW=self.w.shape
        
        OW=W//FW
        self.x=img.transpose(1,0,2,3).reshape(C,N*H*OW,FW)# (N,C,H,W),(C,N,H,W),(C,N*H*OW,FW)
        out=self.x*self.w  #(C,N*H*OW,FW)
        out=np.sum(out,axis=2,keepdims=True)#(C,N*H*OW,1)
        out=out.reshape(C,N,H,OW).transpose(1,0,2,3) #(C,N,H,OW), (N,C,H,OW)
        return out
    
    def backward(self,delta):  #delta为(N,C,H,OW)
        C,FH,FW=self.w.shape
        N,C,H,W=self.img_shape
        OW=W//FW
        
        delta=delta.transpose(1,0,2,3).reshape(C,N*H*OW,1) #(C,N,H,OW) (C,N*H*OW,1)
        dw=np.sum(delta*self.x,axis=1,keepdims=True)/N
        
        delta_out=delta*self.w  #(C,N*H*OW,FW)
        delta_out=delta_out.reshape(C,N,H,W).transpose(1,0,2,3) #(C,N,H,W) (N,C,H,W)
        return delta_out,dw
    
    
class MaxPooling:
    def __init__(self,size,pad=0,dtype=float):
        self.dtype=dtype
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
        
        col=np.zeros((N,C,self.pool_size,self.pool_size,OH,OW),self.dtype)
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
        dmax=np.zeros((delta.size,pool_elem),self.dtype)#N*OH*OW*C行 pool_size**2列
        dmax[np.arange(self.arg_max.size),self.arg_max.flatten()]=delta.flatten()
        dmax=dmax.reshape(delta.shape+(pool_elem,))#相当于reshape(delta.shape[0],delta.shape[1],...,pool_elem)
        
        dcol=dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)#变成N*OH*OW行 C*FH*FW列
        
        N, C, H, W = self.imgshape
        #OH = H//self.pool_size
        #OW = W//self.pool_size
        dcol=dcol.reshape(N,OH,OW,C,self.pool_size,self.pool_size).transpose(0,3,4,5,1,2)#与col维度相同
        delta_next=np.zeros((N,C,H+self.stride-1,W+self.stride-1),self.dtype)
        
        for y in range(self.pool_size):
            y_max=y+self.stride*OH
            for x in range(self.pool_size):
                x_max=x+self.stride*OW
                delta_next[:,:,y:y_max:self.stride,x:x_max:self.stride]+=dcol[:,:,y,x,:,:]
        return delta_next[:,:,:H,:W]


class AveragePooling:
    def __init__(self,size,pad=0,dtype=float):
        self.dtype=dtype
        self.pool_size = size
        self.stride = size
        self.pad = pad
    
    def forward(self,img):
        N, C, H, W = img.shape
        OH = H//self.pool_size
        OW = W//self.pool_size
        col=np.zeros((N,C,self.pool_size,self.pool_size,OH,OW),self.dtype)
        #img=x
        for y in range(self.pool_size):
            y_max=y+self.stride*OH
            for x in range(self.pool_size):
                x_max=x+self.stride*OW
                col[:,:,y,x,:,:]=img[:,:,y:y_max:self.stride,x:x_max:self.stride]
        col=col.transpose(0,4,5,1,2,3).reshape(N*OH*OW,-1)
        col=col.reshape(-1,self.pool_size*self.pool_size)
        
        out=np.sum(col,axis=1)/(self.pool_size**2)  #每一行的平均值
        out=out.reshape(N,OH,OW,C).transpose(0,3,1,2)  #数量 通道 行 列
        self.imgshape=img.shape
        return out
        
    def backward(self,delta):
        N,C,OH,OW=delta.shape
        pool_elem=self.pool_size*self.pool_size
        delta=delta.transpose(0,2,3,1)##变成N,OH,OW,C
        dave=np.ones((delta.size,pool_elem),self.dtype) #N*OH*OW*C行 pool_size**2列
        dave=(dave.T*delta.flatten()).T/pool_elem
        dave=dave.reshape(delta.shape+(pool_elem,))#相当于reshape(delta.shape[0],delta.shape[1],...,pool_elem)
        
        dcol=dave.reshape(dave.shape[0] * dave.shape[1] * dave.shape[2], -1)#变成N*OH*OW行 C*FH*FW列
        
        N, C, H, W = self.imgshape
        dcol=dcol.reshape(N,OH,OW,C,self.pool_size,self.pool_size).transpose(0,3,4,5,1,2)#与col维度相同
        delta_next=np.zeros((N,C,H+self.stride-1,W+self.stride-1),self.dtype)
        
        for y in range(self.pool_size):
            y_max=y+self.stride*OH
            for x in range(self.pool_size):
                x_max=x+self.stride*OW
                delta_next[:,:,y:y_max:self.stride,x:x_max:self.stride]+=dcol[:,:,y,x,:,:]
        return delta_next[:,:,:H,:W]

    
class Sum_of_squares:
    def __init__(self,dtype=float):
        self.dtype=dtype
    
    def forward(self,x):
        self.x=x   # N, C, H, W
        if self.dtype!=complex:
            out=x*x
        else:
            self.x_conj=x.conjugate()
            out=self.x_conj*x
            out=np.real(out)
        return np.sum(out,axis=(2,3),keepdims=True)
    
    def backward(self,delta):
        if self.dtype!=complex:
            delta_next=self.x*delta*2
        else:
            #delta_next=self.x_conj*delta
            delta_next=self.x*delta
        return delta_next


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

class Batch_Norm:  #keras默认参数 momentum=0.99 eps=0.001
    def __init__(self,beta,gamma,momentum,eps,num_features):
        self.eps = eps
        self._momentum = momentum
        #self._beta = np.zeros((num_features,1))
        #self._gamma = np.ones((num_features,1))
        self._beta = beta
        self._gamma = gamma
        
    def forward(self,x,running_mean,running_var,mode):
        self.x=x
        self.N=x.shape[1]
        if mode=='train':
            self.x_mean=x.mean(axis=1).reshape(len(x),1)
            self.x_var =x.var(axis=1).reshape(len(x),1)
            running_mean = (1-self._momentum)*self.x_mean + self._momentum*running_mean
            running_var = (1-self._momentum)*self.x_var + self._momentum*running_var

            self.x_hat = (self.x-self.x_mean)/np.sqrt(self.x_var+self.eps)
            out = self._gamma*self.x_hat + self._beta
        elif mode=='test':
            self.x_hat=(self.x-running_mean)/np.sqrt(self.eps+running_var)
            out=self._gamma*self.x_hat+self._beta
        return out,running_mean,running_var
    
    def backward(self,delta):
        dgamma=np.sum(delta*self.x_hat,axis=1,keepdims=True)
        dbeta=np.sum(delta,axis=1,keepdims=True)
        
        dx_hat=delta*self._gamma
        dx_var=-0.5*np.sum(dx_hat*(self.x-self.x_mean),axis=1,keepdims=True)*np.power(self.x_var+self.eps,-1.5)
        dx_mean=np.sum(-dx_hat/np.sqrt(self.x_var+self.eps),axis=1,keepdims=True)-2*dx_var*np.sum(self.x-self.x_mean,axis=1,keepdims=True)/self.N
        delta=(dx_hat/np.sqrt(self.x_var+self.eps))+(dx_var*2*(self.x-self.x_mean)/self.N)+(dx_mean/self.N)
        return delta,dgamma,dbeta
    

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

class Sigmoid():   #据说这种可以方式sigmoid溢出 虽然不太懂为啥大于小于0分开算就可以避免
    def __init__(self):
        self.out=None
    def forward(self,x):
        mask = (x > 0)
        positive_out = np.zeros_like(x, dtype='float64')
        negative_out = np.zeros_like(x, dtype='float64')
        # 大于0的情况
        positive_out = 1 / (1 + np.exp(-x, positive_out, where=mask))
        positive_out[~mask] = 0 # 清除效于0元素的影响 #~mask 里面的True变False False变True
        
        # 小于等于0的情况
        expx = np.exp(x,negative_out,where=~mask)
        negative_out = expx / (1+expx)
        negative_out[mask] = 0# 清除对大于0元素的影响
        self.out=positive_out + negative_out
        return self.out
    def backward(self,dout):
        return dout*self.out*(1.0-self.out)

class SigmoidWithCrossEntropy:
    def __init__(self):
        self.y=None
        self.t=None
        self.loss=None
    def forward(self,x,t):
        batch_size=x.shape[1]  #就算是二分类 只有一个节点，也是1行 N列  N：样本数
        self.t=t
        
        mask = (x > 0)
        positive_out = np.zeros_like(x, dtype='float64')
        negative_out = np.zeros_like(x, dtype='float64')
        # 大于0的情况
        positive_out = 1 / (1 + np.exp(-x, positive_out, where=mask))
        positive_out[~mask] = 0 # 清除效于0元素的影响 #~mask 里面的True变False False变True
        
        # 小于等于0的情况
        expx = np.exp(x,negative_out,where=~mask)
        negative_out = expx / (1+expx)
        negative_out[mask] = 0# 清除对大于0元素的影响
        self.y=positive_out + negative_out
        
        self.loss=np.sum(-self.t*np.log(self.y+1e-7)-(1-self.t)*np.log(1-self.y+1e-7))/batch_size
        return self.y,self.loss
    def backward(self,delta=1):
        delta=self.y-self.t
        return delta

class Square_Normalize:  #平方归一化层
    def __init__(self):
        self.x=None
    def forward(self,x):
        self.x=x
        self.C=np.sqrt(np.sum(x*x))
        return x/self.C
    def backward(self,delta):
        dout=(1/self.C)-(self.x**2)/(self.C**3)
        delta_out=dout*delta
        return delta_out
