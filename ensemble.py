import config


def combine_scores(classifier_prob, perplexity, burstiness):
    perplexity_score = 1.0 / (perplexity + 1)
    burstiness_norm = burstiness

    final = (
        classifier_prob * config.W_CLASSIFIER +
        perplexity_score * config.W_PERPLEXITY +
        burstiness_norm * config.W_BURSTINESS
    )
    return final
