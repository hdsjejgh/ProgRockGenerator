import music21
import os

import torch
import tqdm
import json

from torch.utils.data import random_split
import torch.optim as optim
from model import *
from data import *


#predicts the next token given the input and model
def predict(inp, prediction_model, sampling_type = "GREEDY", hyperparameter = None):
    sampling_functions = {
        "GREEDY" : greedy_sampling,
        "TOPK" : topK_sampling,
    }

    sampling_type = sampling_type.upper()
    assert sampling_type in sampling_functions.keys(), f"Invalid selection function. Options are: {", ".join(sampling_functions.keys())}"
    sampling_func = sampling_functions[sampling_type]

    if sampling_type=="TOPK":
        assert hyperparameter is not None, "If using Top K sampling, you must supply a hyperparameter for the value of k to be used"

    logits = prediction_model.forward(inp.unsqueeze(0))[0][-1]
    #technically you dont have to softmax it but its done just for debugging in the future
    res = F.softmax(logits, dim=0)

    ind = sampling_func(res)
    return ind

def greedy_sampling(probabilities):
    maxind = torch.argmax(probabilities)
    return maxind

def topK_sampling(probabilities, k):
    values, indices = torch.topk(probabilities, k=k)
    #scales topk probabilities to sum to 1
    prob_sum = values.sum()
    values /= prob_sum

    #chooses randomly based on probabilities
    ind = torch.multinomial(values, num_samples=1)
    return indices[ind.item()]


MODEL_NAME = "RobertFripp3"


if __name__ == "__main__":
    data = music_snippet_dataset(files, 128)

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
    print(sample.size())

    TOKEN_COUNT = 10000

    for i in range(TOKEN_COUNT - len(sample)):
        part = sample[-128:]
        prediction = predict(part, model)
        sample = torch.cat((sample, torch.tensor([prediction])))

    tokens = data.convert_id_to_token(ids=sample.tolist())
    print(tokens)

    save_stream(tokens_to_stream(tokens), "PLEASEWORK2.midi")



