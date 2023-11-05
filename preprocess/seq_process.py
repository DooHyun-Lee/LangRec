import gzip 
import json
from tqdm import tqdm 
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

from transformers import GPT2Tokenizer

# modify to your root directory 
SEQ_ROOT = '/home/doolee13/LangRec/preprocess/data'

pretrain_cats = ['Automotive']
pretrain_seq_pathes= [f'{SEQ_ROOT}/{cat}_5.json.gz' for cat in pretrain_cats]

for path in pretrain_seq_pathes:
    assert os.path.exists(path)

# dictionary with key : asin // val : attributes
meta_data = json.load(open('meta_data.json'))

# key : userId, val : interaction log
train_seqs = defaultdict(list)
val_seqs = defaultdict(list)
test_seqs = defaultdict(list)

def meta_to_sentence(asin):
    attr_dict = meta_data[asin]
    title = attr_dict['title']
    brand = attr_dict['brand'] 
    cat = attr_dict['category']
    sentence = f"This product, titled '{title}' and branded as {brand}, falls under the category of {cat}."
    return sentence

sequences = defaultdict(list)
miss_cnt = 0
valid_cnt = 0

'''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
total_token = 0
unrec_token = 0
'''

with gzip.open(pretrain_seq_pathes[0]) as f:
    for line in tqdm(f):
        line = json.loads(line)
        userid = line['reviewerID']
        asin = line['asin']
        time = line['unixReviewTime']
        if asin in meta_data and line.get('summary', None) is not None:
            review = line['summary']
            temp_dict = {}
            temp_dict['attribute'] = meta_to_sentence(asin)
            temp_dict['review'] = review
            temp_dict['asin'] = asin
            sequences[userid].append((time,temp_dict))
            valid_cnt += 1
            '''
            tokenized = tokenizer.encode(temp_dict['attribute'], add_special_tokens=False)
            unrecognized_tokens = [token for token in tokenized if token == tokenizer.unk_token_id]
            total_token += len(tokenized)
            unrec_token += len(unrecognized_tokens)
            '''
        else:
            miss_cnt += 1

#print(f'unrecognized token {unrec_token} among total {total_token} tokens')

length = 0
training_data = []
for user, sequence in tqdm(sequences.items()):
    sequences[user] = [ele[1] for ele in sorted(sequence, key=lambda x: x[0])]
    training_data.append(sequences[user])
    length += len(sequences[user])

print(f'Averaged length : {length/len(sequences)}')

train_data, temp_data = train_test_split(training_data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


# adjustment for validation data for BC evaluation(considering max_len)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
for i, sequences in enumerate(valid_data):
    if len(sequences) > 15:
        valid_data[i] = valid_data[i][-15:]

valid_data_new = []
valid_data_label = []

for sequences in valid_data:
    token_len =0
    for i, seq in enumerate(sequences):
        for k, v in seq.items():
            token_len += len(tokenizer.encode(v, add_special_tokens=False))
    if token_len < 920:
        valid_data_new.append(sequences[:-1])
        valid_data_label.append(sequences[-1])

with open('train_data.json', 'w') as f:
    json.dump(train_data, f)

with open('valid_data.json', 'w') as f:
    json.dump(valid_data_new, f)
    
with open('valid_data_label.json', 'w') as f:
    json.dump(valid_data_label, f)

with open('test_data.json', 'w') as f:
    json.dump(test_data, f) 