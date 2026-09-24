import numpy as np
from embedding import encode,decode,embedding,Positional_encoding
import json
print('attention running...........')

np.random.seed(42)

class Attention:

    def __init__(self,d_model=256) -> None:
        self.d_model = d_model

        self.w_q = np.random.randn(d_model,d_model)/np.sqrt(d_model)
        self.w_k = np.random.randn(d_model,d_model)/np.sqrt(d_model)
        self.w_v = np.random.randn(d_model,d_model)/np.sqrt(d_model)

    def softmax(self,x):

        e_x = np.exp(x - np.max(x,axis=1,keepdims=True))
        return e_x/np.sum(e_x,axis=1,keepdims=True)

    def forward(self,X):

        seq_len,d_model=X.shape

        Q = X @ self.w_q
        K = X @ self.w_k
        V = X @ self.w_v
        self.V = V

        raw_score = Q @ K.T
        scaled_score = raw_score / np.sqrt(d_model)

        mask = np.triu(np.ones((seq_len,seq_len)),k=1)
        scaled_score = np.where(mask==1,-np.inf,scaled_score)

        attention_weights = self.softmax(scaled_score)
        self.attention_weights = attention_weights
        
        output = attention_weights @ V

        return output ,attention_weights


class MultiHeadAttention:

    def __init__(self,d_model=256,num_heads=4) -> None:
        assert d_model%num_heads == 0

        self.d_model = d_model
        self.num_heads=num_heads
        self.head_dim = d_model // num_heads

        self.w_q = np.random.randn(d_model,d_model)/np.sqrt(d_model)
        self.w_k = np.random.randn(d_model,d_model)/np.sqrt(d_model)
        self.w_v = np.random.randn(d_model,d_model)/np.sqrt(d_model)
        self.w_o = np.random.randn(d_model,d_model)/np.sqrt(d_model)
        
    def softmax(self,x):
        e_x = np.exp(x - np.max(x,axis=-1,keepdims=True))
        return e_x / np.sum(e_x,axis=-1,keepdims=True)

    def forward(self,X):

        seq_len,_ = X.shape
        self.X = X
        Q = X @ self.w_q
        K = X @ self.w_k
        V = X @ self.w_v

        Q = Q.reshape(seq_len,self.num_heads,self.head_dim)
        K = K.reshape(seq_len,self.num_heads,self.head_dim)
        V = V.reshape(seq_len,self.num_heads,self.head_dim)
       

        Q = Q.transpose(1,0,2)
        K = K.transpose(1,0,2)
        V = V.transpose(1,0,2)
        self.V = V
        self.K = K
        self.Q = Q

        scores = Q @ K.transpose(0,2,1)

        scores = scores / np.sqrt(self.head_dim)

        mask = np.triu(np.ones((seq_len,seq_len)),k=1)

        scores = np.where(mask==1,-np.inf,scores)

        attention_weights = self.softmax(scores)
        self.attention_weights = attention_weights

        head_outputs = attention_weights @ V

        head_outputs = head_outputs.transpose(1,0,2)

        combined = head_outputs.reshape(seq_len,self.d_model)
        self.combined = combined

        output = combined @ self.w_o

        return output,attention_weights

    def backward(self,dvalues):

    
        self.dw_o = self.combined.T @ dvalues
        dcombined = dvalues @ self.w_o.T

        dhead_outputs = dcombined.reshape(dcombined.shape[0],self.num_heads,self.head_dim)

        dhead_outputs = dhead_outputs.transpose(1, 0, 2)

        dattention_weights = dhead_outputs @ self.V.transpose(0, 2, 1)

        dV = self.attention_weights.transpose(0, 2, 1) @ dhead_outputs
        
        dot = np.sum(dattention_weights * self.attention_weights,axis=-1,keepdims=True)
        
        dscores = self.attention_weights * (dattention_weights - dot)
        # scaling backward
        dscores = dscores / np.sqrt(self.head_dim)

        # scores = Q @ K.T
        dQ = dscores @ self.K
        dK = dscores.transpose(0, 2, 1) @ self.Q

        dQ = dQ.transpose(1, 0, 2).reshape(-1, self.d_model)
        dK = dK.transpose(1, 0, 2).reshape(-1, self.d_model)
        dV = dV.transpose(1, 0, 2).reshape(-1, self.d_model)

        self.dw_q =self.X.T @ dQ
        self.dw_k = self.X.T @ dK
        self.dw_v = self.X.T @ dV

        dX_q = dQ @ self.w_q.T
        dX_k = dK @ self.w_k.T
        dX_v = dV @ self.w_v.T

        self.dinputs = dX_q + dX_k + dX_v

# text = input('enter input:')
# with open('merges.json','r') as f:
#     data = json.load(f)

# merges={}
# for pair,new_token in data.items():
#     a,b = map(int,pair.split(','))
#     merges[(a,b)]= new_token

# token_ids = encode(text,merges)
# x = embedding.forward(token_ids)
# print("embedding matrix:\n",x)
# print('shape:',x.shape)

# pe = Positional_encoding(len(token_ids),256)
# pos_encoding = pe.sinusoidal_positional_encoding()
# x = x + pos_encoding

# print("embedding + positional matrix:\n",x)
# print('shape:',x.shape)


# atten = MultiHeadAttention()
# y,weights = atten.forward(x)
# print('muti attention matrix:\n',y)
# print('shape:',y.shape)
# print("multi Attention weights:")
# print(weights)