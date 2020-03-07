# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:23:42 2020

@author: alecr
"""

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plt
from nltk import sent_tokenize
import more_itertools as mit

NUM_WINES = 20
MIN_ENTRIES = 5

data = pd.read_csv(r"D:\Data\wine-reviews\winemag-data-130k-v2.csv")
#print(data.columns)

data = data[pd.notnull(data['variety'])]
data = data[pd.notnull(data['description'])]

value_counts = data['variety'].value_counts()
top_wines = list(value_counts[value_counts > MIN_ENTRIES].index)

print( "fraction of wines with over", MIN_ENTRIES, "reviews :", value_counts[value_counts > MIN_ENTRIES].shape[0]/float(value_counts.shape[0]))
print( "number of wines with over", MIN_ENTRIES, "reviews :", value_counts[value_counts > MIN_ENTRIES].shape[0])

label_to_idx = {k:v for v, k in enumerate(top_wines)}
idx_to_label = {v: k for k, v in label_to_idx.items()}

data= data[data.variety.isin(top_wines)]
data['label'] = data['variety'].replace(label_to_idx)
labels = data['label'].values

description_list = data["description"].values

#sentences = [' [CLS] '+ phrase + ' [SEP]' for phrase in description_list]
sentences = [' [CLS] ' + ' [SEP] '.join(sent_tokenize(d)) + ' [SEP] 'for d in description_list]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentence:")
print (tokenized_texts[0])


input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

MAX_LEN = 128
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []

def mask_maker(tokenized_text):
    #by description, this should be tokenized_text
    z = np.zeros(MAX_LEN)
    position = 0
    count = 1
    
    for sent in list(mit.split_after(tokenized_text, lambda x: x == '[SEP]')):
        sent_len = len(sent)
        z[position:position+sent_len+1] = count
        position+=sent_len 
        count+=1
    
    return z

for t in tokenized_texts:
	attention_masks.append(mask_maker(t))
    
print("made masks")

## Put in actual network

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_WINES)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

train_inputs = torch.tensor(train_inputs).long()
validation_inputs = torch.tensor(validation_inputs).long()
train_labels = torch.tensor(train_labels).long()
validation_labels = torch.tensor(validation_labels).long()
train_masks = torch.tensor(train_masks).long()
validation_masks = torch.tensor(validation_masks).long()

batch_size = 16

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)



device = torch.device("cuda:0" )#if torch.cuda.is_available() else "cpu")


train_loss_set = []


# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):
  
  
  # Training
  
  # Set our model to training mode (as opposed to evaluation mode)
  model.train()
  
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  # Train the data for one epoch
  for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    # Forward pass
    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    train_loss_set.append(loss.item())    
    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    
    
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1

  print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
    
  # Validation

  # Put model in evaluation mode to evaluate loss on the validation set
  model.eval()

  # Tracking variables 
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  # Evaluate data for one epoch
  for batch in validation_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))