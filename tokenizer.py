
def get_pair_counts(tokens):
    '''
    Counts the occurrences of every consecutive pair of integers.
    '''
    counts={}
    for pair in zip(tokens,tokens[1:]):
        counts[pair] = counts.get(pair,0)+1
    return counts



def merge(tokens,pair,new_token):
    merged=[]
    i=0
    while i < len(tokens):
        if i<len(tokens)-1 and (tokens[i],tokens[i+1])==pair:
            merged.append(new_token)
            i+=2
        else:
            merged.append(tokens[i])
            i+=1
    return merged



def encode(text,merges):
    ids = list(text.encode('utf-8'))
    while len(ids)>=2:
        stats = get_pair_counts(ids)
        pair = min(stats,key=lambda p: merges.get(p,float('inf')))
        if pair not in merges:
            break
        ids = merge(ids,pair,merges[pair])
    return ids



def decode(ids,merges):

    vocab = {i:bytes([i]) for i in range(256)}

    for pair,new_token in sorted(merges.items(),key=lambda item:item[1]):
        vocab[new_token] = vocab[pair[0]]+ vocab[pair[1]]

    byte_chunks = [vocab[idx] for idx in ids if idx in vocab]
    return b"".join(byte_chunks).decode('utf-8',errors='replace')




# import json
# with open('merges.json','r') as f:
#     data = json.load(f)

# merges={}
# for pair,new_token in data.items():
#     a,b = map(int,pair.split(','))
#     merges[(a,b)]= new_token

# vocab_size = 256 + len(merges)
# print('vocab_size:',vocab_size)

# test_text = input()

# encoded = encode(test_text, merges)
# decoded = decode(encoded, merges)

# print("Original:", test_text)
# print("Encoded:", encoded)
# print("Decoded:", decoded)
# print("Match:", test_text == decoded)

# with open('bpe_corpus.txt','r',encoding='utf-8') as f:
#     text = f.read()
#     # text=text[:100_000]
# tokens= list(text.encode('utf-8'))
# merges = {}
# for _ in range(1000): # 10 just for practice
#     pair_counts = get_pair_counts(tokens)
#     if not pair_counts:
#         break
#     best_pair = max(pair_counts, key=pair_counts.get)
#     new_token = 256 + _
#     merges[best_pair] = new_token
#     tokens = merge(tokens,best_pair,new_token)
#     print("Merged:", best_pair, "→", new_token)

# print('\nLearned merges: ')
# print(merges)    

# encoded = encode(input(), merges)

# print("Encoded:", encoded)

# decoded = decode(encoded, merges)

# print("Decoded:", decoded)