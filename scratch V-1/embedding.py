import numpy as np
from tokenizer import encode,decode
import json

print('embedding running.........')
class Embedding:
    '''vectors of Higher dimensions'''
    def __init__(self,vocab_size,d_model):
        self.weight = np.random.randn(vocab_size,d_model)*0.02

    def forward(self,token_id):
        return self.weight[token_id]

    def backward(self, dvalues, token_ids):

        self.dweights = np.zeros_like(self.weight)

        for i, token_id in enumerate(token_ids):
            self.dweights[token_id] += dvalues[i]

        self.dinputs = dvalues


embedding = Embedding(vocab_size=756,d_model=256)

class Positional_encoding():
    '''
    positions for embedding
    '''
    def __init__(self,context_length,d_model) -> None:
        self.context_length = context_length
        self.d_model = d_model

    def sinusoidal_positional_encoding(self):
        pe = np.zeros((self.context_length,self.d_model))

        for pos in range(self.context_length):
            for i in range(0,self.d_model,2):
                angle = pos/ (10000**(i/self.d_model))
                pe[pos,i]= np.sin(angle)
                pe[pos,i+1]=np.cos(angle)
        return pe


# text = input()

# with open('merges.json','r') as f:
#     data = json.load(f)

# merges={}
# for pair,new_token in data.items():
#     a,b = map(int,pair.split(','))
#     merges[(a,b)]= new_token

# token_id = encode(text,merges)
# print('token ids:',token_id)
# s = decode(token_id,merges)
# print('docded version:',s)
# x = embedding.forward(token_id)
# print(x)
# print("Embedding output shape:", x.shape)


# pe = Positional_encoding(len(token_id), 256)
# pos_encoding = pe.sinusoidal_positional_encoding()
# x = x + pos_encoding

# print("Final input:")
# print(x)
# print("Final shape:", x.shape)
