
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
#BERT
#BertAttention
#GPT2    
#Roberta
#BERTLSTM
#BERTCNN
#BERTaverage




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
