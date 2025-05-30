
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, GPT2Model, RobertaModel, RobertaTokenizer, GPT2Tokenizer
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import json
#hey
# Load configurations
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Define the model architectures
# BERT
class BERTClass(torch.nn.Module):
    def __init__(self, config):
        super(BERTClass, self).__init__()
        self.fine_tuning_strategy = config['fine_tuning_strategy']
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(config['dropout_rate'])
        self.l3 = torch.nn.Linear(768, 2)
        
        if self.fine_tuning_strategy == 'full':
            for param in self.l1.parameters():
                param.requires_grad = True
        elif self.fine_tuning_strategy == 'partial':
            # Freeze bottom layers
            for param in self.l1.parameters():
                param.requires_grad = False
            # Unfreeze top layers
            for param in self.l1.encoder.layer[-4:].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        output_1 = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Extract hidden state of the [CLS] token
        cls_token_state = output_1[0][:, 0, :]  # Extracting the hidden state of the [CLS] token
        
        output_2 = self.l2(cls_token_state)  # Apply dropout
        logits = self.l3(output_2)  # Linear layer for classification
        
        if labels is not None:
            # Calculate the CrossEntropyLoss
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        else:
            return logits
        


# BertAttention
class AttentionAggregator(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttentionAggregator, self).__init__()
        self.attention = MultiheadAttention(embed_dim=input_size, num_heads=1)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # BERT output embeddings as input
        query = x.transpose(0, 1) 
        attn_output, _ = self.attention(query, query, query)  
        aggregated = torch.mean(attn_output, dim=0) 
        return self.linear(aggregated)

class BERTWithAttention(nn.Module):
    def __init__(self, config):
        super(BERTWithAttention, self).__init__()
        self.fine_tuning_strategy = config['fine_tuning_strategy']
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.attention_aggregator = AttentionAggregator(input_size=self.bert.config.hidden_size, 
                                                        output_size=self.bert.config.hidden_size)
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        
        if self.fine_tuning_strategy == 'full':
            for param in self.bert.parameters():
                param.requires_grad = True
        elif self.fine_tuning_strategy == 'partial':
            # Freeze bottom layers
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze top layers
            for param in self.bert.encoder.layer[-4:].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        aggregated_output = self.attention_aggregator(outputs.last_hidden_state)
        pooled_output = self.dropout(aggregated_output)
        logits = self.classifier(pooled_output)
        return logits
    
## GPT2    
class GPT2class(nn.Module):
    def __init__(self, config):
        super(GPT2class, self).__init__()
        self.fine_tuning_strategy =  config['fine_tuning_strategy']
        self.gpt2model = GPT2Model.from_pretrained("gpt2")
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.fc1 = nn.Linear(self.gpt2model.config.n_embd, 2)  
        
        
        if self.fine_tuning_strategy == 'full':
            for param in self.gpt2model.parameters():
                param.requires_grad = True
        elif self.fine_tuning_strategy == 'partial':
            # Freeze bottom layers
            for param in self.gpt2model.parameters():
                param.requires_grad = False
            # Unfreeze top layers
            for param in self.gpt2model.h[-4:].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        gpt_out = self.gpt2model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        gpt_out = self.dropout(gpt_out)
        logits = self.fc1(gpt_out[:, 0, :])  # Assuming you want to use only the first token's output for classification

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        else:
            return logits


#Roberta
class RobertaClass(torch.nn.Module):
    def __init__(self, config):
        super(RobertaClass, self).__init__()
        self.fine_tuning_strategy = config['fine_tuning_strategy']
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(config['dropout_rate'])
        self.classifier = torch.nn.Linear(768, 2)
        
        if self.fine_tuning_strategy == 'full':
            for param in self.l1.parameters():
                param.requires_grad = True
        elif self.fine_tuning_strategy == 'partial':
            # Freeze bottom layers
            for param in self.l1.parameters():
                param.requires_grad = False
            # Unfreeze top layers
            for param in self.l1.encoder.layer[-4:].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

#BERTLSTM
class BERTLSTM(nn.Module):
    def __init__(self, config):
        super(BERTLSTM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fine_tuning_strategy =  config['fine_tuning_strategy']
        embedding_size =  self.bert.config.to_dict()['hidden_size']
        self.lstm = nn.LSTM(input_size= 768, 
                            hidden_size=768, 
                            num_layers= config['num_lstm_layers'],
                            batch_first=True,
                            dropout=config['dropout_rate'],
                            bidirectional=True)
        self.dense = nn.Linear(768 * 2, 2)
        self.softmax = nn.Softmax(dim=1)
        
        if self.fine_tuning_strategy == 'full':
            for param in self.bert.parameters():
                param.requires_grad = True
        elif self.fine_tuning_strategy == 'partial':
            # Freeze bottom layers
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze top layers
            for param in self.bert.encoder.layer[-4:].parameters():
                param.requires_grad = True
        

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            embedded = self.bert(input_ids=input_ids.squeeze(1),
                                 attention_mask=attention_mask)
      
        _, (hidden_states, cell_states) = self.lstm(embedded['last_hidden_state'])

        # Concatenate the final hidden states from both directions
        concatenated_hidden = torch.cat((hidden_states[-2,:,:], hidden_states[-1,:,:]), dim=1)

        output = self.dense(concatenated_hidden)
        output = self.softmax(output)
        return output
    

# BERTCNN

class BERTCNN(nn.Module):
    def __init__(self, config):
        super(BERTCNN, self).__init__()
        self.num_labels = 2
        self.fine_tuning_strategy = config['fine_tuning_strategy']
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        
        if self.fine_tuning_strategy == 'full':
            for param in self.bert.parameters():
                param.requires_grad = True
        elif self.fine_tuning_strategy == 'partial':
            # Freeze bottom layers
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze top layers
            for param in self.bert.encoder.layer[-4:].parameters():
                param.requires_grad = True
        
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(in_channels=768*4, out_channels=32, kernel_size=kernel_size)
            for kernel_size in [1, 2, 3, 4, 5]
        ])
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.fc = nn.Linear(32*5, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = torch.cat([outputs.hidden_states[i] for i in range(-4, 0)], dim=-1)
        hidden_states = hidden_states.transpose(-2, -1)
        conv_outs = [F.relu(conv_block(hidden_states)) for conv_block in self.conv_blocks]
        pooled_outs = [F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2) for conv_out in conv_outs]
        concat_out = torch.cat(pooled_outs, dim=1)
        logits = self.fc(concat_out)
        return torch.sigmoid(logits)

class BERTaverage(torch.nn.Module):
    def __init__(self, config):
        super(BERTaverage, self).__init__()
        self.fine_tuning_strategy = config['fine_tuning_strategy']
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(config['dropout_rate'])
        self.l3 = torch.nn.Linear(768, 2)
        
        if self.fine_tuning_strategy == 'full':
            for param in self.l1.parameters():
                param.requires_grad = True
        elif self.fine_tuning_strategy == 'partial':
            # Freeze bottom layers
            for param in self.l1.parameters():
                param.requires_grad = False
            # Unfreeze top layers
            for param in self.l1.encoder.layer[-4:].parameters():
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        output_1 = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Extract hidden states of all tokens
        all_token_states = output_1[0] 
        
        # Perform average pooling over token embeddings
        avg_pooled_output = torch.mean(all_token_states, dim=1)  
        
        output_2 = self.l2(avg_pooled_output)  #  dropout
        logits = self.l3(output_2)  # Linear layer for classification
        
        if labels is not None:
            # Calculate the CrossEntropyLoss
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        else:
            return logits



# Load the tokenizer (use different tokenizers for different models)
tokenizers = {
    "BERTClass": BertTokenizer.from_pretrained('bert_tokenizer'),
    "GPT2class": GPT2Tokenizer.from_pretrained('gpt2_tokenizer'),
    "RobertaClass": RobertaTokenizer.from_pretrained('roberta_tokenizer'),
    "BERTWithAttention": BertTokenizer.from_pretrained('bertattention_tokenizer'),
    "BERTLSTM": BertTokenizer.from_pretrained('bertlstm_tokenizer'),
    "BERTCNN": BertTokenizer.from_pretrained('bertcnn_tokenizer'),
    "BERTaverage": BertTokenizer.from_pretrained('bertaverage_tokenizer')

}

def predict(model_name, text):
    try:
        model = None
        tokenizer = tokenizers[model_name]

        # Load the appropriate model and its config based on the input
        if model_name == "BERTClass":
            config = load_config('bert_config.json')
            model = BERTClass(config)
            model.load_state_dict(torch.load('./model/bert.pt', map_location=torch.device('cpu')))
        elif model_name == "GPT2class":
            config = load_config('gpt2_config.json')
            model = GPT2class(config)
            model.load_state_dict(torch.load('./model/gpt2.pt', map_location=torch.device('cpu')))
        elif model_name == "RobertaClass":
            config = load_config('roberta_config.json')
            model = RobertaClass(config)
            model.load_state_dict(torch.load('./model/roberta.pt', map_location=torch.device('cpu')))
        elif model_name == "BERTWithAttention": 
            config = load_config('bertattention_config.json')
            model = BERTWithAttention(config)
            model.load_state_dict(torch.load('./model/bertattention.pt', map_location=torch.device('cpu')))
        elif model_name == "BERTLSTM":
            config = load_config('bertlstm_config.json')
            model = BERTLSTM(config)
            model.load_state_dict(torch.load('./model/bertlstm.pt', map_location=torch.device('cpu')))
        elif model_name == "BERTCNN":
            config = load_config('bertcnn_config.json')
            model = BERTCNN(config)
            model.load_state_dict(torch.load('./model/bertcnn.pt', map_location=torch.device('cpu')))
        elif model_name == "BERTaverage":
            config = load_config('bertaverage_config.json')
            model = BERTaverage(config)
            model.load_state_dict(torch.load('./model/bertaverage.pt', map_location=torch.device('cpu')))


        model.eval()

        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=200,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Perform inference
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()

        return {
            "prediction": "hateful" if pred_class == 1 else "normal",
            "confidence": confidence
        }
    except Exception as e:
        raise {"error": str(e)}
