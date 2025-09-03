import torch
import torch.nn.functional as F
from torch.nn import init

def pearson_correlation(x, y, dim=0):
    # Calculate means
    mean_x = torch.mean(x, dim=dim, keepdim=True)
    mean_y = torch.mean(y, dim=dim, keepdim=True)

    # Calculate covariance and variances
    cov_xy = torch.mean((x - mean_x) * (y - mean_y), dim=dim, keepdim=True)
    var_x = torch.mean((x - mean_x)**2, dim=dim, keepdim=True)
    var_y = torch.mean((y - mean_y)**2, dim=dim, keepdim=True)

    # Calculate Pearson correlation coefficient
    corr_coeff = cov_xy / (torch.sqrt(var_x) * torch.sqrt(var_y))

    return corr_coeff


#neural network to compute similarity of two metapopulation nodes
class matchingA(torch.nn.Module):
    def __init__(self, timeDim, strainDim, n, channel= 3, midLayer= 30, device= "cpu"):
        super(matchingA, self).__init__()

        self.channel=  channel
        self.n= n
        self.midLayer= midLayer
        self.strainDim= strainDim

        self.Wu= torch.nn.Linear(timeDim, midLayer*channel, device= device)
        self.Wv= torch.nn.Linear(timeDim, midLayer*channel, device= device)

        self.scalar_a = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.Wnorm= torch.nn.Linear(channel, 1, bias= False, device= device)

        self.AmatBias= torch.rand((n, n), dtype= torch.float32, device=device)*1e-6
        self.AmatBias= torch.nn.Parameter(self.AmatBias)

        self.myEye= torch.eye(n, dtype= torch.float32, device= device)
        self.myMask= torch.ones(n, dtype= torch.float32, device= device)- self.myEye
        self.mySig= torch.nn.Sigmoid()
        self.mySig2= torch.nn.Sigmoid()
        self.init_weight()


    def init_weight(self):
        init.xavier_uniform_(self.Wu.weight)
        init.normal_(self.Wu.bias, mean=0.0, std=1)
        init.xavier_uniform_(self.Wv.weight)
        init.normal_(self.Wv.bias, mean=0.0, std=1)
        init.xavier_uniform_(self.Wnorm.weight)

    def forward(self, x, mode= 0): #[50, 2, timeDim]
        transU= (self.Wu(x)).view(self.n, 1, self.channel*x.shape[1], self.midLayer)#[50, 1, 2*channel, midlayer]

        transV= (self.Wv(x)).view(1, self.n, self.channel*x.shape[1], self.midLayer)#[1, 50, 2*channel, midlayer]

        self.Atemp = F.cosine_similarity(transU, transV, dim=-1)#[50, 50, 2*channel]

        self.Atemp= self.Atemp.view(self.n, self.n, self.channel, self.strainDim)
        self.Atemp= torch.mean(self.Atemp, dim= -1).squeeze()


        Anorm= self.Wnorm(self.Atemp).squeeze()#[50, 50]
        ATemp= Anorm+self.AmatBias
        
        scalar_sig= self.mySig2(self.scalar_a)
        ATemp2= scalar_sig*ATemp+(1-scalar_sig)*ATemp.transpose(0, 1)
        Ainfer= self.mySig(ATemp2)*self.myMask
        return Ainfer

    
    
#neural network to compute the SIR spidemic gradient
class EpisA(torch.nn.Module):
    def __init__(self, input_dim= 20, num_heads= 1, n= 50, device= "cpu"):
        super(EpisA, self).__init__()
        self.device= device
        self.num_heads= num_heads
        self.n= n
        # self.taus= torch.ones((n, num_heads), dtype= torch.float32, device=device)*6
        self.taus= torch.rand((n, num_heads), dtype= torch.float32, device=device)*torch.rand(1).item()*30#*6, torch.rand(1).item()*30
        print(torch.mean(self.taus))
        self.taus= torch.nn.Parameter(self.taus)

        # self.R0dTaus= torch.ones((n, num_heads), dtype= torch.float32, device=device)*1
        self.R0dTaus= torch.rand((n, num_heads), dtype= torch.float32, device=device)
        self.R0dTaus= torch.nn.Parameter(self.R0dTaus)
        self.mat, self.mask= self.create_temporal_mat(input_dim)
        self.myRelu= torch.nn.ReLU()
        self.mySig= torch.nn.Sigmoid()
        self.mySoft= torch.nn.Softmax(dim=2)
        self.myEye= torch.eye(n, dtype= torch.float32, device= device)

    def alpha(self, i, R0, tau):
        return 1-torch.exp(-(R0/tau)*i)
    
    def create_temporal_mat(self, lang):
        mat= torch.zeros((lang, lang), dtype= torch.float32, device= self.device)
        mask= torch.zeros((lang, lang), dtype= torch.float32, device= self.device)
        for i in range(lang):
            for j in range(i+1):
                mat[i, j]= i- j
                mask[i, j]= 1
        return mat[None, None, ...], mask[None, None, ...]

    def forward(self, x, Amat= None): # shape: (1, 2, 20), dim of nodes, dim of heads, dim of signal
        # divide= self.mySoft(self.output)*x[:, :, -1:]#(1, 2, 20), dim of nodes, dim of heads, dim of signal
        # divide= divide.transpose(1, 2)
        tempAmat= Amat.T+ self.myEye
        signal= self.myRelu(x) #\delta S
        Ss= 1- torch.cumsum(signal, dim= -1) #easiy negative
        IsMat= torch.exp(self.mat*torch.log( 1-1/(torch.abs(self.taus[... , None, None])+1.01) ))*self.mask
        Is= torch.matmul(IsMat, signal[..., None]).squeeze(dim=-1)
        alpha= (1-torch.exp(-(1e-3+torch.abs(self.R0dTaus[... , None]))*Is))
        temp= tempAmat[..., None, None]*alpha[:, None, ...]
        # print(alpha.shape)
        Alpha= temp.sum(dim= 0)
        predSignal= Alpha*Ss
        #signalPredict= self.alpha(Is[0: -1], R0, tau)*Ss[0:-1] 
        return predSignal, signal, tempAmat.T

#neural network to compute similarity of two metapopulation nodes
class matchingB(torch.nn.Module):
    def __init__(self, timeDim, strainDim, n, midLayer= 50, device= "cpu"):
        super(matchingB, self).__init__()

        self.Amat= torch.randn((n, n), dtype= torch.float32, device=device)/10-1
        self.Amat= torch.nn.Parameter(self.Amat)
        self.AmatMask= 1- torch.eye(n, dtype= torch.float32, device=device)
        self.mySig= torch.nn.Sigmoid()

    def forward(self, x, mode=0): #[50, 1, 36]
        if mode=="BA"  or mode=="infer2018":
            return self.mySig(self.Amat)*self.AmatMask
        return self.Amat

class ThresholdHook:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, grad):
        return torch.clamp(grad, min=-self.threshold, max=self.threshold)

class EpisB(torch.nn.Module):
    def __init__(self, input_dim= 20, num_heads= 1, n= 50, device= "cpu"):#
        super(EpisB, self).__init__()
        self.device= device
        self.num_heads= num_heads
        self.n= n
        self.taus= torch.ones((n, num_heads), dtype= torch.float32, device=device)*6
        self.taus= torch.nn.Parameter(self.taus)
        self.R0dTaus= torch.ones((n, num_heads), dtype= torch.float32, device=device)*1
        self.R0dTaus= torch.nn.Parameter(self.R0dTaus)
        self.myRelu= torch.nn.ReLU()
        self.mySig= torch.nn.Sigmoid()
        self.mySoft= torch.nn.Softmax(dim=2)
        self.Amateye= torch.eye(n, dtype= torch.float32, device=device)
        self.AmatMask= 1- torch.eye(n, dtype= torch.float32, device=device)

    def alpha(self, i, R0, tau):
        return 1-torch.exp(-(R0/tau)*i)

    def core_function(self, S, I, tempAmat):#x: (50, 2, 1), dim of nodes, dim of heads, (S, I)
        alpha= (1-torch.exp(-torch.abs(self.R0dTaus[... , None])*I))
        temp= tempAmat[..., None, None]*alpha[:, None, ...]
        Alpha= temp.sum(dim= 0)
        Alpha[Alpha>1]= 1
        Alpha[Alpha<0]= 0
        dS= Alpha*S
        S= S- dS
        I= I-I/torch.abs(self.taus[..., None])+dS
        return S, I, dS


    def forward(self, x, Amat): # shape: (50, 2, 20), dim of nodes, dim of heads, dim of signal
        tempAmat= self.AmatMask*self.mySig(Amat.T)/10+ self.Amateye

        signal= self.myRelu(x) #\delta S

        timeHorizonT= signal.shape[2]

        IT= signal[:, :, 0:1]

        ST= torch.ones_like(IT, device= self.device)- IT

        predSignal= []

        for _ in range(timeHorizonT):
            ST, IT, dS= self.core_function(ST, IT, tempAmat)
            predSignal.append(dS.clone())

        predSignal= torch.stack(predSignal, dim= -1).squeeze(dim= -2)

        return predSignal, signal, tempAmat.T

