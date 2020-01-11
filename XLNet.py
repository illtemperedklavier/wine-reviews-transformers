import pandas as pd
from nltk import sent_tokenize, word_tokenize
from itertools import permutations
from pytorch_transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from pytorch_transformers import AdamW
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import io
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

NUM_WINES = 20
data = pd.read_csv(r"C:\Users\alecr\Projects\assignments\Semantic-Health\alec-wine-reviews\data\winemag-data-130k-v2.csv")
data.head()
data.columns
"""
data = data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'country', 'designation',
       'points', 'price', 'province', 'region_1', 'region_2', 'taster_name',
       'taster_twitter_handle', 'title', 'winery'], axis=1)
	   """
data = data[pd.notnull(data['variety'])]
data = data[pd.notnull(data['description'])]

data.shape
value_counts = data['variety'].value_counts()
top_wines = list(value_counts[:NUM_WINES].index)

label_to_idx = {k:v for v, k in enumerate(value_counts.index[:NUM_WINES])}
idx_to_label = {v: k for k, v in label_to_idx.items()}
data= data[data.variety.isin(top_wines)]
data.shape

data['label'] = data['variety'].replace(label_to_idx)
labels = data['label'].values

description_list = data["description"].values
#sentences = tokenize by sentence and rejoin them all together with [SEP] between them and [CLS] at the end

sentences = [' [SEP] '.join(sent_tokenize(phrase)) + ' [SEP] [CLS]' for phrase in description_list]
#sentences = [sentence+ " [SEP] [CLS]" for sentence in description_list]


tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentence:")
print (tokenized_texts[0])

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

MAX_LEN = 128
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []

def mask_maker(description):
    z = np.zeros(MAX_LEN)
    position = 0
    count = 1
    for sent in sent_tokenize(description):
        sent_len = len(tokenizer.tokenize(sent))
        z[position:sent_len+1] = count
        position+=sent_len
        count+=1
    
    return z



for d in description_list:
	attention_masks.append(mask_maker(d))

 
"""
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
"""

print("masks made")


	

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)

dev_inputs, test_inputs, dev_labels, test_labels = train_test_split(validation_inputs, validation_labels, random_state=2018, test_size=0.3)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,random_state=2018, test_size=0.1)

dev_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids,random_state=2018,test_size=0.1)





train_inputs = torch.tensor(train_inputs).long()
validation_inputs = torch.tensor(validation_inputs).long()
train_labels = torch.tensor(train_labels.astype(np.int)).long()
validation_labels = torch.tensor(validation_labels.astype(np.int)).long()
train_masks = torch.tensor(train_masks).long()
validation_masks = torch.tensor(validation_masks).long()

print("train_inputs.shape: ", train_inputs.shape)
print("validation_inputs: ", validation_inputs.shape)
print("train_labels", train_labels.shape)
print("validation_labels", validation_labels.shape)
print("train_masks", train_masks.shape)
print("validation_masks", validation_masks.shape)

batch_size = 16

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0))

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=NUM_WINES)

model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
	

# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 10

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
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
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
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = output[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

model.save_pretrained(r"C:\Users\alecr\Projects\assignments\Semantic-Health\alec-wine-reviews\models\models-XLNet5")
#print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))