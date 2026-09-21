from attention import MultiHeadAttention
from normalization import LayerNorm
from feedforward import FeedForward
from embedding import encode,decode,Positional_encoding,embedding
import numpy as np
from math import erf

import json
class TransformerBlock:

    def __init__(self,d_model=256,num_heads=4,d_ff=1024):
        self.attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.normalization1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model,d_ff=d_ff)
        self.normalization2 = LayerNorm(d_model)

    def forward(self,inputs):

        attention_outputs,attention_weights = self.attention.forward(inputs)
        x = inputs + attention_outputs
        x = self.normalization1.forward(x)

        ffn_output = self.ffn.forward(x)
        x = ffn_output + x
        x = self.normalization2.forward(x)

        return x,attention_weights


class Transformer:

    def __init__(self,num_layers=4,d_model=256,num_heads=4,d_ff=1024):
        self.blocks=[]
        for _ in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff
                )
            )

    def forward(self,inputs):
        x = inputs
        attention_weights = []
        for block in self.blocks:

            x,weights = block.forward(x)
            attention_weights.append(weights)

        return x,attention_weights



class Add_layer:

    def __init__(self,inputs,neurons):
        self.weights = np.random.randn(inputs,neurons) * np.sqrt(2/(inputs))
        self.bias = np.zeros((1,neurons))

    def forward(self,inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs,self.weights) + self.bias

    def backward(self,dvalues):

        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbias = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class LanguageModelHead:
    def __init__(self,d_model=256,vocab_size=1256):
        self.output_layer = Add_layer(d_model,vocab_size)

    def forward(self,inputs):
        self.output_layer.forward(inputs)

        return self.output_layer.outputs


class Activation_Softmax:

    def forward(self,inputs):
        exp_value = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_value / np.sum(exp_value,axis=1,keepdims=True)
        self.outputs = probabilities

    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index,(single_output,single_dvalues) in enumerate(zip(self.outputs,dvalues)):

            single_output = single_output.reshape(-1,1)
            jacobian_matrix = (np.diagflat(single_output) - np.dot(single_output,single_output.T))
            self.dinputs[index] = (jacobian_matrix @ single_dvalues)



transformer = Transformer(
    num_layers=4,
    d_model=256,
    num_heads=4,
    d_ff=1024
)

text = input("enter input: ")

with open("merges.json", "r") as f:
    data = json.load(f)

merges = {}

for pair, new_token in data.items():
    a, b = map(int, pair.split(","))
    merges[(a, b)] = new_token


# Tokenize
token_ids = encode(text, merges)

# print("Token IDs:")
# print(token_ids)


# Embedding
x = embedding.forward(token_ids)

# print("Embedding shape:")
# print(x.shape)


# Positional encoding
pe = Positional_encoding(len(token_ids), 256)

pos_encoding = pe.sinusoidal_positional_encoding()

x = x + pos_encoding

# print("After positional encoding:")
# print(x.shape)

output, attention_weights = transformer.forward(x)

print("Transformer output shape:", output.shape)
print("Number of layers:", len(attention_weights))

for i, weights in enumerate(attention_weights):
    print(f"Layer {i+1} attention shape:", weights.shape)




lm_head = LanguageModelHead(d_model=256,vocab_size=756)
logits = lm_head.forward(output)

print("Logits shape:", logits.shape)

softmax = Activation_Softmax()
softmax.forward(logits)
probabilities = softmax.outputs

print("Probabilities shape:", probabilities.shape)
print("\nProbability sums:")

print(np.sum(probabilities, axis=1))
predicted_ids = np.argmax(probabilities, axis=1)

print("\nPredicted token IDs:")
print(predicted_ids)
predicted_text = decode(predicted_ids, merges)

print("\nPredicted text:")
print(predicted_text)
















# # Transformer Block
# block = TransformerBlock(
#     d_model=256,
#     num_heads=4,
#     d_ff=1024
# )

# output, attention_weights = block.forward(x)


# print("\nTransformer block output:")
# print(output)

# print("Output shape:")
# print(output.shape)


# print("\nAttention weights shape:")
# print(attention_weights.shape)