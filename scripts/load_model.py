import torch
from transformers import AutoTokenizer

from cadet.models.cadet import CADET


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 256

ENCODER_ID = "roberta-base"
DECODER_ID = "facebook/bart-base"

# Choisis ton modèle final
CHECKPOINT_PATH = (
    "results/"
    "model_inference/model.pt" #cadet_finetune_lr_low_results
)

# D'après ton run : zt = 13
N_TARGETS = 13

# Pour le run 182944, meilleur macro F1
THRESHOLD = 0.46

# Si tu utilises le run 160036, mets plutôt :
# CHECKPOINT_PATH = "results/cadet-hatecot-finetune-seed42-20260516_160036/checkpoints/best/model.pt"
# THRESHOLD = 0.46


model = CADET(
    n_targets=N_TARGETS,
    encoder_checkpoint=ENCODER_ID,
    decoder_checkpoint=DECODER_ID,
    style_dim=2,
    conf_dim=256,
    orth_dim=128,
    use_confounder_for_prediction=False,
)

state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID)
decoder_tokenizer = AutoTokenizer.from_pretrained(DECODER_ID)

print("✅ Modèle chargé")
print("Device:", DEVICE)


def predict(text: str):
    enc = encoder_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    dec = decoder_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    dec = {k: v.to(DEVICE) for k, v in dec.items()}

    with torch.no_grad():
        outputs = model(
            enc["input_ids"],
            enc["attention_mask"],
            dec["input_ids"],
            dec["attention_mask"],
        )

    hate_logits = outputs["hate_logits"]

    print("Output keys:", outputs.keys())
    print("hate_logits shape:", hate_logits.shape)

    if hate_logits.shape[-1] == 2:
        probs = torch.softmax(hate_logits, dim=-1)
        hate_prob = probs[0, 1].item()
        non_hate_prob = probs[0, 0].item()
    else:
        hate_prob = torch.sigmoid(hate_logits.view(-1))[0].item()
        non_hate_prob = 1.0 - hate_prob

    pred = int(hate_prob >= THRESHOLD)

    return {
        "text": text,
        "non_hate_prob": non_hate_prob,
        "hate_prob": hate_prob,
        "threshold": THRESHOLD,
        "pred": pred,
        "label": "hate" if pred == 1 else "non-hate",
    }


texts = [
    "My aunt is an immigrant."
]

#The model correctly detects implicit hate when the target group is explicitly introduced in the context. However, when the target is only referred to by an ambiguous pronoun such as “they”, the model tends to classify the sentence as non-hate. This suggests that CADET relies strongly on contextual target cues and struggles with decontextualized anaphoric references.

for text in texts:
    result = predict(text)
    print("\n---")
    print("Texte:", result["text"])
    print("Probabilité non-hate:", result["non_hate_prob"])
    print("Probabilité hate:", result["hate_prob"])
    print("Seuil:", result["threshold"])
    print("Classe prédite:", result["label"])