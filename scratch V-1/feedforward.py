import numpy as np
from math import erf
from tokenizer import encode,decode
from embedding import embedding,Positional_encoding
from attention import MultiHeadAttention
from normalization import LayerNorm
import json
print('feedforward running..........')
class Add_layer:

    def __init__(self, inputs, neurons):
        self.weights = np.random.randn(inputs, neurons) * np.sqrt(2 / inputs)
        self.bias = np.zeros((1, neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.bias

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_GeLu:

    def forward(self, inputs):
        self.inputs = inputs

        phi_cdf = 0.5 * (1 + np.vectorize(erf)(inputs / np.sqrt(2)))

        self.outputs = inputs * phi_cdf

    def backward(self, dvalues):

        phi_cdf = 0.5 * (1 + np.vectorize(erf)(self.inputs / np.sqrt(2)))

        phi_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-(self.inputs ** 2) / 2)

        gelu_derivative = phi_cdf + self.inputs * phi_pdf

        self.dinputs = dvalues * gelu_derivative


class FeedForward:

    def __init__(self,d_model=256,d_ff=1024) -> None:

        self.layer1 = Add_layer(d_model,d_ff)
        self.gelu = Activation_GeLu()
        self.layer2 = Add_layer(d_ff,d_model)

    def forward(self,inputs):

        self.layer1.forward(inputs)
        self.gelu.forward(self.layer1.outputs)
        self.layer2.forward(self.gelu.outputs)

        return self.layer2.outputs
    
    def backward(self,dvalues):

        self.layer2.backward(dvalues)
        self.gelu.backward(self.layer2.dinputs)
        self.layer1.backward(self.gelu.dinputs)

        self.dinputs = self.layer1.dinputs


# text = input('enter input:')
# with open('merges.json','r') as f:
#     data = json.load(f)

# merges={}
# for pair,new_token in data.items():
#     a,b = map(int,pair.split(','))
#     merges[(a,b)]= new_token

# token_ids = encode(text,merges)
# x = embedding.forward(token_ids)


# pe = Positional_encoding(len(token_ids),256)
# pos_encoding = pe.sinusoidal_positional_encoding()
# x = x + pos_encoding




# atten = MultiHeadAttention()
# attention_output, weights = atten.forward(x)
# print('muti attention matrix:\n',attention_output)
# print('shape:',attention_output.shape)
# print("multi Attention weights:")
# print(weights)


# x = x + attention_output
# normal1 = LayerNorm(256)
# x = normal1.forward(x)
# print("After Attention + Residual + LayerNorm:")
# print(x)
# print("Mean of each token:")
# print(np.mean(x, axis=-1))
# print("\nVariance of each token:")
# print(np.var(x, axis=-1))


# normal = LayerNorm()
# x = normal.forward(x)
# print('normalized\n:',x)
# print("Mean of each token:")
# print(np.mean(x, axis=-1))

# print("\nVariance of each token:")
# print(np.var(x, axis=-1))





# ffn = FeedForward(d_model=256, d_ff=1024)
# ffn_output = ffn.forward(x)
# print("Input shape:", x.shape)
# print("FFN output shape:", ffn_output.shape)



# x = x + ffn_output
# normal2 = LayerNorm(256)
# x = normal2.forward(x)
# print("After FFN + Residual + LayerNorm:")
# print(x)
# print("Mean of each token:")
# print(np.mean(x, axis=-1))
# print("\nVariance of each token:")
# print(np.var(x, axis=-1))
