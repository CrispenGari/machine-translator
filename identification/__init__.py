import json, os
import torch
from torch import nn
from torch.nn import functional as F
from flask import Blueprint, request, make_response, jsonify
from utils.main import Constants
from flask_cors import cross_origin

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
device = Constants.DEVICE
language_identification_blueprint = Blueprint("identification", __name__)
tokenizer = lambda x: x.split(" ")

with open (os.path.join(
    os.getcwd(), 'identification/static/labels_vocab.json'
)) as f:
    LABEL = json.load(f)

with open (os.path.join(
    os.getcwd(), 'identification/static/words_vocab.json'
)) as f:
    TEXT = json.load(f)

class CustomException(Exception):
    pass

class LanguageIndentifierFastText(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 output_dim,
                 pad_index,
                 dropout=.5
                 ):
        super(LanguageIndentifierFastText, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=pad_index
        )
        self.out = nn.Linear(
            embedding_size,
            out_features=output_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)
                              ).squeeze(1)
        return self.out(pooled)
    
# Model instance

INPUT_DIM = len(TEXT)
EMBEDDING_DIM = 100
OUTPUT_DIM =  len(LABEL)
PAD_IDX = TEXT.get(PAD_TOKEN)
labels = {v:k for k,v in LABEL.items()}

language_identifier_model = None

languages = [
 
  {
    "name": "english",
    "id": 0,
    "code": "eng",
  },
  {
    "name": "swedish",
    "id": 1,
    "code": "swe",
  },
  {
    "name": "french",
    "id": 2,
    "code": "fra",
  },
   {
    "name": "germany",
    "id": 3,
    "code": "deu",
  },
  {
    "name": "italian",
    "id": 4,
    "code": "ita",
  },
  {
    "name": "portuguese",
    "id": 5,
    "code": "por",
  },
  {
    "name": "afrikaans",
    "id": 6,
    "code": "afr",
  }
]

def predict_language(model, sent):
  model.eval()
  sent = sent.lower()
  tokenized = tokenizer(sent)
  indexed = [TEXT.get(t) if TEXT.get(t) is not None else TEXT.get(UNK_TOKEN) for t in tokenized]
  tensor = torch.LongTensor(indexed)
  tensor = tensor.unsqueeze(1)
  probabilities = torch.softmax(model(tensor), dim=1)
  prediction = torch.argmax(probabilities, dim=1)
  item = prediction.item()
  probs = probabilities.squeeze(0)
  preds = []
  for i, p in enumerate(probs):
      preds.append({
        "prediction":languages[i],
        "probability": round(p.item(), 2),
    })
  return {
      "label": item,
      "lang": labels[item],
      "prediction":languages[item],
      "probability": round(probs[item].item(), 2),
       "meta":{
        "programmer":"@crispengari",
        "project": "likeme",
        "main": "artificial intelligence (nlp)"
    },
    "predictions":preds
  }

print(" *   LOADING IDENTIFICATION MODEL\n")
language_identifier_model = LanguageIndentifierFastText(
        INPUT_DIM,
        EMBEDDING_DIM,
        OUTPUT_DIM,
        pad_index= PAD_IDX
)
language_identifier_model.load_state_dict(torch.load(os.path.join(
    os.getcwd(), 'identification/model/best-lang-ident-model.pt'
), map_location=device))
print(" *   LOADING IDENTIFICATION MODEL DONE!\n")