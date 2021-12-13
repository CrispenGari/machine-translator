import torch, spacy
import os, json
from utils.main import Constants
from flask import Blueprint, make_response, jsonify, request
from seq2seq.decoder import Decoder
from seq2seq.encoder import Encoder
from seq2seq.attention import Attention
from seq2seq.seq2seq import Seq2Seq
from flask_cors import cross_origin

device = Constants.DEVICE
base_dir = os.path.join(os.getcwd(), "translation/models")
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

class CustomException(Exception):
    pass

# Tokenizers
print(" *   LOADING TOKENIZERS\n")
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')
spacy_fr = spacy.load("fr_core_news_sm")
spacy_it = spacy.load("it_core_news_sm")
spacy_es = spacy.load("es_core_news_sm")
spacy_pt = spacy.load("pt_core_news_sm")

print(" *   LOADING TOKENIZERS DONE!\n")
generalTokenizer = lambda x: str(x).split(" ")


def tokenize_de(sent: str) -> list:
    return [tok.text for tok in spacy_de.tokenizer(sent)]

def tokenize_fr(sent: str) -> list:
    return [tok.text for tok in spacy_fr.tokenizer(sent)]

def tokenize_en(sent: str) -> list:
    return [tok.text for tok in spacy_en.tokenizer(sent)]

def tokenize_it(sent: str) -> list:
    return [tok.text for tok in spacy_it.tokenizer(sent)]

def tokenize_es(sent: str) -> list:
    return [tok.text for tok in spacy_es.tokenizer(sent)]

def tokenize_pt(sent: str) -> list:
    return [tok.text for tok in spacy_pt.tokenizer(sent)]




def createDictMappings(parent_folder: str):
    parent_folder = os.path.join(base_dir, parent_folder, "static")
    src_json_path = os.path.join(parent_folder, "src_vocab.json")
    trg_json_path = os.path.join(parent_folder, "trg_vocab.json")

    with open(src_json_path, 'r') as src, open(trg_json_path, 'r') as trg:
        SRC = json.load(src)
        TRG = json.load(trg)
    return SRC, TRG


ENC_EMB_DIM = DEC_EMB_DIM = 256
ENC_HID_DIM = DEC_HID_DIM = 128
ENC_DROPOUT = DEC_DROPOUT = 0.5


def createModel(
        INPUT_DIM,
        OUTPUT_DIM,
        SRC_PAD_IDX,
        MODEL_NAME
):
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
    return model


# Load all the models here (bi-directional-models)

print(" *   LOADING TRANSLATION MODELS\n")
# Bidirectional Germany - English
DE_DE_DICT, DE_EN_DICT = createDictMappings('eng-deu')

INPUT_DIM = len(DE_DE_DICT)
OUTPUT_DIM = len(DE_EN_DICT)
DE_DE_DICT_REVERSED = {v: k for k, v in DE_DE_DICT.items()}
DE_EN_DICT_REVERSED = {v: k for k, v in DE_EN_DICT.items()}

SRC_PAD_IDX = DE_DE_DICT.get(PAD_TOKEN)
MODEL_NAME = os.path.join(base_dir, "eng-deu", "de-eng.pt")
de_en_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)

SRC_PAD_IDX = DE_DE_DICT.get(PAD_TOKEN)
OUTPUT_DIM = len(DE_DE_DICT)
INPUT_DIM = len(DE_EN_DICT)
MODEL_NAME = os.path.join(base_dir, "eng-deu", "eng-de.pt")
en_de_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)


# Bidirectional French - English
FR_FR_DICT, FR_EN_DICT = createDictMappings('eng-fra')

INPUT_DIM = len(FR_FR_DICT)
OUTPUT_DIM = len(FR_EN_DICT)
FR_FR_DICT_REVERSED = {v: k for k, v in FR_FR_DICT.items()}
FR_EN_DICT_REVERSED = {v: k for k, v in FR_EN_DICT.items()}

SRC_PAD_IDX = FR_FR_DICT.get(PAD_TOKEN)
MODEL_NAME = os.path.join(base_dir, "eng-fra", "fr-eng.pt")
fr_en_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)

SRC_PAD_IDX = FR_FR_DICT.get(PAD_TOKEN)
OUTPUT_DIM = len(FR_FR_DICT)
INPUT_DIM = len(FR_EN_DICT)
MODEL_NAME = os.path.join(base_dir, "eng-fra", "eng-fr.pt")
en_fr_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)




# Bidirectional Afrikaans - English

AF_AF_DICT, AF_EN_DICT = createDictMappings('eng-afr')

INPUT_DIM = len(AF_AF_DICT)
OUTPUT_DIM = len(AF_EN_DICT)
AF_AF_DICT_REVERSED = {v: k for k, v in AF_AF_DICT.items()}
AF_EN_DICT_REVERSED = {v: k for k, v in AF_EN_DICT.items()}

SRC_PAD_IDX = AF_AF_DICT.get(PAD_TOKEN)
MODEL_NAME = os.path.join(base_dir, "eng-afr", "af-eng.pt")
af_en_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)

SRC_PAD_IDX = AF_AF_DICT.get(PAD_TOKEN)
OUTPUT_DIM = len(AF_AF_DICT)
INPUT_DIM = len(AF_EN_DICT)
MODEL_NAME = os.path.join(base_dir, "eng-afr", "eng-af.pt")
en_af_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)




# Bidirectional Italian - English
IT_IT_DICT, IT_EN_DICT = createDictMappings('eng-ita')

INPUT_DIM = len(IT_IT_DICT)
OUTPUT_DIM = len(IT_EN_DICT)
IT_IT_DICT_REVERSED = {v: k for k, v in IT_IT_DICT.items()}
IT_EN_DICT_REVERSED = {v: k for k, v in IT_EN_DICT.items()}

SRC_PAD_IDX = IT_IT_DICT.get(PAD_TOKEN)
MODEL_NAME = os.path.join(base_dir, "eng-ita", "it-eng.pt")
it_en_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)

SRC_PAD_IDX = IT_IT_DICT.get(PAD_TOKEN)
OUTPUT_DIM = len(IT_IT_DICT)
INPUT_DIM = len(IT_EN_DICT)
MODEL_NAME = os.path.join(base_dir, "eng-ita", "eng-it.pt")
en_it_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)


# Bidirectional Swedish - English

SW_SW_DICT, SW_EN_DICT = createDictMappings('eng-swe')

INPUT_DIM = len(SW_SW_DICT)
OUTPUT_DIM = len(SW_EN_DICT)
SW_SW_DICT_REVERSED = {v: k for k, v in SW_SW_DICT.items()}
SW_EN_DICT_REVERSED = {v: k for k, v in SW_EN_DICT.items()}

SRC_PAD_IDX = SW_SW_DICT.get(PAD_TOKEN)
MODEL_NAME = os.path.join(base_dir, "eng-swe", "sw-eng.pt")
sw_en_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)

SRC_PAD_IDX = SW_SW_DICT.get(PAD_TOKEN)
OUTPUT_DIM = len(SW_SW_DICT)
INPUT_DIM = len(SW_EN_DICT)
MODEL_NAME = os.path.join(base_dir, "eng-swe", "eng-sw.pt")
en_sw_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)


# Bidirectional Spanish - English
SP_SP_DICT, SP_EN_DICT = createDictMappings('eng-spa')

INPUT_DIM = len(SP_SP_DICT)
OUTPUT_DIM = len(SP_EN_DICT)
SP_SP_DICT_REVERSED = {v: k for k, v in SP_SP_DICT.items()}
SP_EN_DICT_REVERSED = {v: k for k, v in SP_EN_DICT.items()}

SRC_PAD_IDX = SP_SP_DICT.get(PAD_TOKEN)
MODEL_NAME = os.path.join(base_dir, "eng-spa", "sp-eng.pt")
sp_en_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)

SRC_PAD_IDX = SP_SP_DICT.get(PAD_TOKEN)
OUTPUT_DIM = len(SP_SP_DICT)
INPUT_DIM = len(SP_EN_DICT)
MODEL_NAME = os.path.join(base_dir, "eng-spa", "eng-sp.pt")
en_sp_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)


# Bidirectional Portuguese - English
PO_PO_DICT, PO_EN_DICT = createDictMappings('eng-por')

INPUT_DIM = len(PO_PO_DICT)
OUTPUT_DIM = len(PO_EN_DICT)
PO_PO_DICT_REVERSED = {v: k for k, v in PO_PO_DICT.items()}
PO_EN_DICT_REVERSED = {v: k for k, v in PO_EN_DICT.items()}

SRC_PAD_IDX = PO_PO_DICT.get(PAD_TOKEN)
MODEL_NAME = os.path.join(base_dir, "eng-por", "po-eng.pt")
po_en_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)

SRC_PAD_IDX = PO_PO_DICT.get(PAD_TOKEN)
OUTPUT_DIM = len(PO_PO_DICT)
INPUT_DIM = len(PO_EN_DICT)
MODEL_NAME = os.path.join(base_dir, "eng-por", "eng-po.pt")
en_po_model = createModel(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, MODEL_NAME)

print(" *   LOADING TRANSLATION MODELS DONE!\n")

translation_models_blueprint = Blueprint("translation", __name__)

def translate_sentence(sent, src_field, trg_field, trg_field_reversed, model, device, tokenizer=None, max_len=50):
    model.eval()
    if tokenizer is not None:
        tokens = [token.lower() for token in tokenizer(sent)]
    else:
        tokens = generalTokenizer(sent)
    tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]

    src_indexes = [src_field.get(token) if src_field.get(token) is not None else src_field.get(UNK_TOKEN) for token in
                   tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)
    trg_indexes = [trg_field.get(SOS_TOKEN)]
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        attentions[i] = attention
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.get(EOS_TOKEN):
            break


    trg_tokens = [trg_field_reversed.get(i) for i in trg_indexes]
    return trg_tokens[1:], attentions[:len(trg_tokens) - 1]


def getFunctionParams(from_, to):

    # de-en bidirectional
    if from_ == "de":
        return DE_DE_DICT, DE_EN_DICT, DE_EN_DICT_REVERSED, de_en_model, tokenize_de
    if from_ == "eng" and to == "de":
        return DE_EN_DICT, DE_DE_DICT, DE_DE_DICT_REVERSED, en_de_model, tokenize_en

    # fr-en bidirectional
    if from_ == "fr":
        return FR_FR_DICT, FR_EN_DICT, FR_EN_DICT_REVERSED, fr_en_model, tokenize_fr
    if from_ == "eng" and to == "fr":
        return FR_EN_DICT, FR_FR_DICT, FR_FR_DICT_REVERSED, en_fr_model, tokenize_en
    
    # af-en bidirectional
    if from_ == "af":
        return AF_AF_DICT, AF_EN_DICT, AF_EN_DICT_REVERSED, af_en_model, generalTokenizer
    if from_ == "eng" and to == "af":
        return AF_EN_DICT, AF_AF_DICT, AF_AF_DICT_REVERSED, en_af_model, tokenize_en

    # it-en bidirectional
    if from_ == "it":
        return IT_IT_DICT, IT_EN_DICT, IT_EN_DICT_REVERSED, it_en_model, tokenize_it
    if from_ == "eng" and to == "it":
        return IT_EN_DICT, IT_IT_DICT, IT_IT_DICT_REVERSED, en_it_model, tokenize_en

    # it-en bidirectional
    if from_ == "sw":
        return SW_SW_DICT, SW_EN_DICT, SW_EN_DICT_REVERSED, sw_en_model, generalTokenizer
    if from_ == "eng" and to == "sw":
        return SW_EN_DICT, SW_SW_DICT, SW_SW_DICT_REVERSED, en_sw_model, tokenize_en

    # pt(po)-en bidirectional
    if from_ == "pt":
        return PO_PO_DICT, PO_EN_DICT, PO_EN_DICT_REVERSED, po_en_model, tokenize_pt
    if from_ == "eng" and to == "pt":
        return PO_EN_DICT, PO_PO_DICT, PO_PO_DICT_REVERSED, en_po_model, tokenize_en

    # es(sp) -en bidirectional
    if from_ == "es":
        return SP_SP_DICT, SP_EN_DICT, SP_EN_DICT_REVERSED, sp_en_model, tokenize_es
    if from_ == "eng" and to == "es":
        return SP_EN_DICT, SP_SP_DICT, SP_SP_DICT_REVERSED, en_sp_model, tokenize_en


meta = {
    "name": "ml backend",
    "language": "python",
    "author": "@crispengari",
    "package": "pytorch",
    "description": "language identification and translation graphql api.",
    "project": "noteme"
}