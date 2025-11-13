import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from perplexity import get_perplexity
from burstiness import burstiness_score
from ensemble import combine_scores
import config

tokenizer = AutoTokenizer.from_pretrained("classifier_model")
model = AutoModelForSequenceClassification.from_pretrained("classifier_model")


def detect_text(text):
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    logits = model(**enc).logits
    prob_ai = torch.softmax(logits, dim=1)[0][1].item()

    ppl = get_perplexity(text)
    burst = burstiness_score(text)

    final_score = combine_scores(prob_ai, ppl, burst)

    if final_score >= config.AI_THRESHOLD:
        label = "AI generated"
    elif final_score >= config.SUSPECT_THRESHOLD:
        label = "Possibly AI generated"
    else:
        label = "Human"

    return {
        "label": label,
        "ai_score": final_score,
        "classifier_prob_ai": prob_ai,
        "perplexity": ppl,
        "burstiness": burst
    }
