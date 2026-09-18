LABELS = ["CLEAN", "OFFENSIVE", "HATE"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

HF_MODEL_IDS = {
    "baseline": "BonTori/phobert-baseline-retrain-hsd",
    "bt": "AnoraLee/vietnamese-hsd-phobert-combined",
    "eda": "BonTori/phobert-eda-retrain-hsd",
    "llm": "BonTori/phobert-llm-segmented-hsd",
    "combined": "AnoraLee/vietnamese-hsd-phobert-bt",
}

EXPERIMENT_ORDER = ["baseline", "bt", "eda", "llm", "combined"]