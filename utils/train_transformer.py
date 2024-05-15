import torch
import torch.nn as nn
from utils.dataset_transformer import DiscourseDataloader
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification


def train_model(OUTPUTS, MODEL , MODEL_TYPE, TRAIN_LEN, TEST_LEN, CHECKPOINT_DIR, device, DATA_DIR, EPOCHS, STEP_SIZE, BATCH_SIZE, WEIGHT_DECAY, EPS, model_weights, load_model = False):
    

    iteration = 0
    model, criterion, optimizer = initialize_model(OUTPUTS, device, MODEL, STEP_SIZE, WEIGHT_DECAY, EPS, model_weights)
    BEST_ACC = 0
    if load_model == True:
      model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'model_deberta '+ MODEL_TYPE + '.ckpt')))
    for epoch in range(EPOCHS): 
        print (epoch)
        test_predictions = torch.empty(0, device = torch.device('cpu'))
        test_truth = torch.empty(0, device = torch.device('cpu'))
        #Initializing Custom Pytorch Dataset class
        dataset_train = DiscourseDataloader(csv_file = DATA_DIR+'_train.csv', chunk_size = BATCH_SIZE, MODEL = MODEL, len_dataset = TRAIN_LEN, target_type = MODEL_TYPE)
        dataset_test = DiscourseDataloader(csv_file = DATA_DIR+'_test.csv', chunk_size = BATCH_SIZE, MODEL = MODEL, len_dataset = TEST_LEN, target_type = MODEL_TYPE)
        for i in range(len(dataset_train)): 
  
          vectors, targets = dataset_train[i]
          iteration = iteration + 1
          #embeddings shape batch_size * embedding size
          #embedding size defined explicitly
          vectors = vectors.to(device)
          labels = torch.from_numpy(np.array(targets)).type(torch.LongTensor).to(device)
          #Training the model
          optimizer.zero_grad()
          outputs = model(**vectors).logits

          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          
        for i in range(len(dataset_test)): 
          vectors, targets = dataset_test[i]
          vectors = vectors.to(device)
          
          labels = torch.from_numpy(np.array(targets)).type(torch.LongTensor).to(device)
          #Saving test results
          with torch.no_grad():
              outputs = model(**vectors).logits
              test_predictions = torch.cat((test_predictions, outputs.detach().cpu()), 0)
              test_truth = torch.cat((test_truth, labels.cpu()), 0)
            
            
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model_deberta ' + MODEL_TYPE + str(epoch) + '.ckpt'))
        iteration = 0
        test_predictions = test_predictions.argmax(dim = 1)
        test_predictions = test_predictions.numpy()
        test_truth = test_truth.numpy()
        test_acc = (test_truth == test_predictions).sum()/len(test_truth) 
        print (test_acc)
        print(confusion_matrix(test_truth, test_predictions))
            
        test_predictions = torch.empty(0, device = torch.device('cpu'))
        test_truth = torch.empty(0, device = torch.device('cpu'))
            
    return model, BEST_ACC
  
def initialize_model(OUTPUTS, device, MODEL, STEP_SIZE, WEIGHT_DECAY, EPS, model_weights = None):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels = OUTPUTS).to(device)
    if model_weights == None:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        print (model_weights)
        criterion = nn.CrossEntropyLoss(weight = model_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=STEP_SIZE, weight_decay= WEIGHT_DECAY, eps= EPS)
    return model, criterion, optimizer
  
def load_existing_model(MODEL_PATH, MODEL_NAME, OUTPUTS, device):
  
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = OUTPUTS).to(device)
  model.load_state_dict(torch.load(MODEL_PATH, map_location = torch.device(device)))
  return model