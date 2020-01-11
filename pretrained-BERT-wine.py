#https://mccormickml.com/2019/07/22/BERT-fine-tuning/

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from util import input_maker
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

MAX_LEN = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
data = pd.read_csv(r"D:\Data\wine-reviews\winemag-data-130k-v2.csv")
counter = Counter(data['variety'].tolist())
top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
data = data[data['variety'].map(lambda x: x in top_10_varieties)]
description_list = data['description'].tolist() 
description_list = ["[CLS] " + sentence + " [SEP]" for sentence in description_list]
    
	
#create a list of the index of the wine type
varietal_list = [top_10_varieties[i] for i in data['variety'].tolist()]
varietal_list = np.array(varietal_list)

outputs_binary = np.where(varietal_list>4, 1,0)

outputs_1hot = to_categorical(varietal_list)

outputs_10 = varietal_list

print("created binary and categorical outputs")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_wine_texts = [tokenizer.tokenize(d) for d in description_list]
print("texts loaded and tokenized")

count = 0
for i in range(60): 
    d = tokenized_wine_texts[i]
    if d is None and count<10:
        print(d)
        count+=1
		
sequence_lengths = [len(s) for s in tokenized_wine_texts]
print("the 98th percentile: ", np.percentile(sequence_lengths, 98))
print("max: ", np.max(sequence_lengths))

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_wine_texts]

seqs = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_wine_texts]

none_count = 0
#truncate so the max is 64
inpt_seqs = []
for seq in input_ids:
    new_seq = seq[:(MAX_LEN - 1)]
    new_seq.append(seq[-1])
    if not new_seq:
        #print("seq is none, ", seq)
        none_count +=1
    inpt_seqs.append(new_seq)
	

#len(inpt_seqs)

input_ids = pad_sequences(inpt_seqs, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

#input_ids.shape

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)
  
for i in range(5):
	print(np.sum(attention_masks[i]))
	
print("attention masks created")

# Use train_test_split to split our data into train and validation sets for training

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, outputs_10, 
                                                            random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)
											 
											 
# Convert all of our data into torch tensors, the required datatype for our model

train_inputs = torch.tensor(train_inputs).long()
#print (train_inputs.type())
validation_inputs = torch.tensor(validation_inputs).long()
#print (validation_inputs.type())
train_labels = torch.tensor(train_labels).long()
#print (train_labels.type())
validation_labels = torch.tensor(validation_labels).long()
#print (validation_labels.type())
train_masks = torch.tensor(train_masks).long()
#print (train_masks.type())
validation_masks = torch.tensor(validation_masks).long()
#print (validation_masks.type())

print("train_outputs shape: ",train_inputs.shape)

# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 32 

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# This variable contains all of the hyperparemeter information our training loop needs
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)
					 

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
	
	
train_loss_set = []


#model ready

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
  
  