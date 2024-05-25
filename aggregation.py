from copy import deepcopy


def aggregate_models(model_updates, trust_scores):
    aggregated_model = deepcopy(model_updates[0])
    for key in aggregated_model.keys():
        aggregated_model[key] = aggregated_model[key] * trust_scores[0]
        for i in range(1, len(model_updates)):
            aggregated_model[key] += model_updates[i][key] * trust_scores[i]
    return aggregated_model


def equal_aggregate_models(model_updates):
    trust_scores = [1 / len(model_updates) for _ in range(len(model_updates))]
    aggregated_model = deepcopy(model_updates[0])
    for key in aggregated_model.keys():
        aggregated_model[key] = aggregated_model[key] * trust_scores[0]
        for i in range(1, len(model_updates)):
            aggregated_model[key] += model_updates[i][key] * trust_scores[i]
    return aggregated_model
