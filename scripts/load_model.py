import torch
from transformers import AutoTokenizer

from cadet.models.cadet import CADET
from cadet.datasets.dataloader import CADETLoader


DEVICE = "cpu"

DATASET_NAME = "IsHate"
SOURCE_STYLE = "implicit"
TARGET_STYLE = "explicit"
MAX_LENGTH = 256

ENCODER_ID = "roberta-base"
DECODER_ID = "facebook/bart-base"

CHECKPOINT_PATH = (
    "trained_runs/"
    "cadet-IsHate-implicit-seed42-20260411_031450/"
    "checkpoints/best/model.pt"
)


# === 1. Charger le loader pour récupérer n_targets ===
loader = CADETLoader(
    dataset_name=DATASET_NAME,
    source_style=SOURCE_STYLE,
    target_style=TARGET_STYLE,
    target_conf_threshold=0.9,
    encoder_tokenizer_id=ENCODER_ID,
    decoder_tokenizer_id=DECODER_ID,
    max_length=MAX_LENGTH,
    root="Shuwan/cadet-datasets",
    random_seed=42,
)

loader.load_data()
n_targets = loader.n_targets

print(f"Nombre de targets: {n_targets}")


# === 2. Recréer le modèle ===
model = CADET(
    n_targets=n_targets,
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

print("✅ Modèle chargé avec succès")


# === 3. Charger les tokenizers ===
encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID)
decoder_tokenizer = AutoTokenizer.from_pretrained(DECODER_ID)


# === 4. Fonction de prédiction ===
THRESHOLD = 0.05

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

    with torch.no_grad():
        outputs = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
            dec["input_ids"].to(DEVICE),
            dec["attention_mask"].to(DEVICE),
        )

    logits = outputs["hate_logits"]
    probs = torch.softmax(logits, dim=1)

    hate_prob = probs[0, 1].item()
    pred = int(hate_prob >= THRESHOLD)

    return logits, probs, pred

# === 5. Test sur texte ===
text = "Hello how are you?"  # Exemple de texte potentiellement haineux

logits, probs, pred = predict(text)

labels = {
    0: "non-hate",
    1: "hate",
}

print("Texte:", text)
print("Logits:", logits)
print("Probabilité non-hate:", probs[0, 0].item())
print("Probabilité hate:", probs[0, 1].item())
print("Seuil utilisé:", THRESHOLD)
print("Classe prédite:", labels[pred])