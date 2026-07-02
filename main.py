import os

import tqdm
import json

from model import *
from data import *


#predicts the next token given the input and model
def predict(inp, prediction_model, sampling_type = "GREEDY", hyperparameter = None):
    sampling_functions = {
        "GREEDY" : greedy_sampling,
        "TOPK" : topk_sampling,
        "NEURON": neuron_sampling,
    }

    #verifies an appropriate sampling type was given and retrieves the corresponding function
    sampling_type = sampling_type.upper()
    assert sampling_type in sampling_functions.keys(), f"Invalid selection function. Options are: {", ".join(sampling_functions.keys())}"
    sampling_func = sampling_functions[sampling_type]

    #ensures top k sampling has a valid k value
    if sampling_type=="TOPK":
        assert hyperparameter is not None, "If using Top K sampling, you must supply a hyperparameter for the value of k to be used"
        assert isinstance(hyperparameter,int) and hyperparameter>0, "Hyperparameter for Top K sampling must be a positive integer"

    #ensures neuron sampling has a valid p value
    if sampling_type=="NEURON":
        assert hyperparameter is not None, "If using Neuron sampling, you must supply a hyperparameter for the value of p to be used"
        assert 1>=hyperparameter>0, "Hyperparameter for Neuron sampling must be a valid probability in the range (0,1]"

    logits = prediction_model.forward(inp.unsqueeze(0))[0][-1]
    res = F.softmax(logits, dim=0)

    ind = sampling_func(res, hyperparameter)
    return ind

#chooses next token index via greedy sampling
def greedy_sampling(probabilities, *args):
    maxind = torch.argmax(probabilities)
    return maxind

#chooses next token index via top K sampling
def topk_sampling(probabilities, k):
    values, indices = torch.topk(probabilities, k=k)

    #chooses randomly based on probabilities
    ind = torch.multinomial(values, num_samples=1)
    return indices[ind.item()]

#chooses next token index via neuron sampling
def neuron_sampling(probabilities, p):
    sorted_values, sorted_indices = torch.sort(probabilities, descending=True)

    #uses mask to get the largest n probabilities that sum to below p
    cum_sums = torch.cumsum(sorted_values, dim=0)
    mask = cum_sums <= p
    top_p_values = sorted_values[mask]
    top_p_indices = sorted_indices[mask]

    #if no sample is smaller than p, just return the most likely one
    if len(top_p_values)==0:
        return sorted_indices[0]

    # chooses randomly based on probabilities
    ind = torch.multinomial(top_p_values, num_samples=1)
    return top_p_indices[ind.item()]


MODEL_NAME = "RobertFripp3"


if __name__ == "__main__":
    print("Loading Data...")
    data = music_snippet_dataset(files, 128)

    print("Loading Model...")
    # loads the model parameters from the saved json and stores them in variables just in case
    with open(os.path.join("models", MODEL_NAME, "model_parameters.json"), 'r') as file:
        model_parameters = json.load(file)
    EMBEDDING_DIM, HEADS, MAX_TOKENS, VOCAB_SIZE, LAYERS = model_parameters.values()

    # model = DecoderOnlyTransformer(
    #     EMBEDDING_DIM=EMBEDDING_DIM,
    #     HEADS=HEADS,
    #     MAX_TOKENS=MAX_TOKENS,
    #     VOCAB_SIZE=VOCAB_SIZE,
    #     LAYERS=LAYERS,
    # )

    # loads model
    model = torch.load(os.path.join("models", MODEL_NAME, MODEL_NAME + ".pt"), weights_only=False)

    # model.load_state_dict(state_dict)
    model.eval()

    sample = data[0][0].clone().detach()

    TOKEN_COUNT = 5000

    with torch.no_grad():
        for i in tqdm.tqdm(range(TOKEN_COUNT - len(sample))):
            part = sample[-128:]
            prediction = predict(
                inp = part,
                prediction_model = model,
                sampling_type = "NEURON",
                hyperparameter = 0.65,
            )
            sample = torch.cat((sample, torch.tensor([prediction])))

    tokens = data.convert_id_to_token(ids=sample.tolist())

    save_stream(tokens_to_stream(tokens), "LarpsTonguesInAspic.midi")



