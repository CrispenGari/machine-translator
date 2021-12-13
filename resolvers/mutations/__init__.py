from ariadne import MutationType
from translation import getFunctionParams, translate_sentence, EOS_TOKEN, UNK_TOKEN, device, meta
from identification import predict_language, language_identifier_model
mutation = MutationType()
@mutation.field("translate")
def translate(obj, info, input):
    src_field, trg_field, trg_field_reversed, model, tokenizer = getFunctionParams(input.get("from_"),
                                                                                           input.get("to"))
    
    tokens, _ = translate_sentence(
        input.get("text"),
        src_field,
        trg_field,
        trg_field_reversed,
        model,
        device,
        tokenizer=tokenizer
    )
    return {

        "meta": meta,
        "from_": input.get("from_"),
        "to": input.get("to"),
        "sent": input.get("text"),
        "translation": " ".join(tokens).replace(EOS_TOKEN, ".").replace(UNK_TOKEN, "unknown")
    }

@mutation.field("identify")
def identify(obj, info, input):
    predictions = predict_language(language_identifier_model, input.get("text"))
    
    return predictions

