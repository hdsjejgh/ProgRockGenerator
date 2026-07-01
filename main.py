import music21
import os

import torch
import tqdm
import json

#music21.configure.run()

from torch.utils.data import random_split
import torch.optim as optim
from model import *
from data import *


files = [
    "AndYouAndI-Yes","SiberianKhatru-Yes","CloseToTheEdge-Yes", #from Yes' Close To The Edge
    "InTheCourtOfKingCrimson-KingCrimson","Red-KingCrimson", "Discipline-KingCrimson", "Exiles-KingCrimson", "FrameByFrame-KingCrimson", "StarlessAndBibleBlack-KingCrimson"  #from King Crimson's Discography
]
files = list(map(lambda x:os.path.join("midis",x)+".mid",files)) #converts the file names to their actual relative paths



data = music_snippet_dataset(files,128)

#proportion of the data to be split across training, testing, and validation
TRAIN_P = .7
TEST_P = .2
VALID_P = 1-TRAIN_P-TEST_P
assert abs(1-(TRAIN_P+TEST_P+VALID_P)) < 1e-5 and 1>TRAIN_P>0 and 1>TEST_P>0 and 1>VALID_P>0, \
    f"train, test, and validation proportions must be validation proportions which add to 1: {TRAIN_P} + {TEST_P} + {VALID_P} = {(TRAIN_P+TEST_P+VALID_P)}"

train_set, test_set, valid_set = random_split(data, [TRAIN_P, TEST_P, VALID_P])

#creates data loaders for each of the datasets
train_loader = get_loader(
    data=train_set,
    batch_size=32,
    shuffle=True,
)
test_loader = get_loader(
    data=test_set,
    batch_size=32,
    shuffle=False,
)
valid_loader = get_loader(
    data=valid_set,
    batch_size=32,
    shuffle=False,
)

#initializes model parameters between [-.08, .08]
def initialize(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


# trains the model for one epoch
def train(model, loader, optimizer, criterion, clip):
    # puts the model in train mode
    model.train()
    # total loss over all batches
    epoch_loss = 0

    for i, batch in enumerate(loader):
        if i%20==0: print(i)
        src,trg = batch

        # clears gradient in optimizer
        optimizer.zero_grad()

        # calculates predictions based on source

        with torch.amp.autocast(device_type="cuda"):

            output = model.forward(src, trg)

            #changes shape to calculate loss
            output = output.view(-1, VOCAB_SIZE)
            trg = trg.view(-1)

            # calculates loss and gradients
            loss = criterion(output, trg)
            scaler.scale(loss).backward()
            # clips gradient in order to stop exploding gradient
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # updates parameters
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # adds loss this batch to total loss
        epoch_loss += loss.item()
    # returns the average loss per batch
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
    # puts the model in evaluation mode
    model.eval()

    # will sum loss over each batch
    epoch_loss = 0

    # turns off the gradient calculation for speed
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src, trg = batch

            # calculates predictions without teacher forcing
            output = model(src, trg)

            #changes shape to calculate loss
            output = output.view(-1, VOCAB_SIZE)
            trg = trg.view(-1)

            loss = criterion(output, trg)
            # adds loss to the total
            epoch_loss += loss.item()

    # returns the average loss per batch
    return epoch_loss / len(loader)

#predicts the next token given the input and model
def predict(inp, prediction_model):
    logits = prediction_model.forward(inp.unsqueeze(0))[0][-1]
    #technically you dont have to softmax it but its done just for debugging in the future
    res = F.softmax(logits, dim=0)

    maxind = torch.argmax(res)
    return maxind


#this model will lowk replace robert fripp and revive king crimson
MODEL_NAME = "RobertFripp3"
#whether to train a new model (True) or load a model (False)
TRAIN = False

#if training the model
if TRAIN:
    # model hyperparameters
    EMBEDDING_DIM = 256
    HEADS = 4
    MAX_TOKENS = 128
    VOCAB_SIZE = len(data.unique_tokens)
    LAYERS = 2

    model = DecoderOnlyTransformer(
        EMBEDDING_DIM=EMBEDDING_DIM,
        HEADS=HEADS,
        MAX_TOKENS=MAX_TOKENS,
        VOCAB_SIZE=VOCAB_SIZE,
        LAYERS=LAYERS,
    )

    initialize(model)

    # scales gradients in training
    scaler = torch.cuda.amp.GradScaler()
    # uses adamW optimizing algorithm
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), )
    # loss is CCE
    criterion = nn.CrossEntropyLoss(label_smoothing=0.075)
    # total epochs of training
    EPOCHS = 10
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    #makes model directory if it doesnt exist
    os.makedirs(os.path.join("models", MODEL_NAME), exist_ok=True)

    #saves model parameters in json file in the model's directory
    model_parameters = {
        "EMBEDDING_DIM":EMBEDDING_DIM,
        "HEADS":HEADS,
        "MAX_TOKENS":MAX_TOKENS,
        "VOCAB_SIZE":VOCAB_SIZE,
        "LAYERS":LAYERS,
    }
    with open(os.path.join("models", MODEL_NAME, "model_parameters.json"), "w") as f:
        json.dump(model_parameters, f, indent=4)

    best_loss = float('inf')
    for epoch in tqdm.tqdm(range(EPOCHS)):
        # runs through the training set and gets the loss (~30k examples)
        train_loss = train(
            model = model,
            loader = train_loader,
            optimizer = optimizer,
            criterion = criterion,
            clip = 1.0,
        )

        # runs through the validation set and gets the loss (~1k examples)
        valid_loss = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
        )

        # if this epoch got the best validation loss so far, save this version of the model
        if valid_loss < best_loss:
            best_loss = valid_loss

            torch.save(model, os.path.join("models", MODEL_NAME, MODEL_NAME + ".pt"))

        # displays the loss info
        print(f"Train Loss: {train_loss}")
        print(f"Validation Loss: {valid_loss}")

#if not training model (ie, testing or evaluating it)
elif not TRAIN:
    #loads the model parameters from the saved json and stores them in variables just in case
    with open(os.path.join("models", MODEL_NAME, "model_parameters.json"), 'r') as file:
        model_parameters = json.load(file)
    EMBEDDING_DIM,HEADS,MAX_TOKENS,VOCAB_SIZE,LAYERS = model_parameters.values()

    # model = DecoderOnlyTransformer(
    #     EMBEDDING_DIM=EMBEDDING_DIM,
    #     HEADS=HEADS,
    #     MAX_TOKENS=MAX_TOKENS,
    #     VOCAB_SIZE=VOCAB_SIZE,
    #     LAYERS=LAYERS,
    # )

    #loads model
    model = torch.load(os.path.join("models", MODEL_NAME, MODEL_NAME + ".pt"), weights_only=False)

    #model.load_state_dict(state_dict)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    #loss on the testing dataset
    test_loss = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss}")

    sample = data[0][0].clone().detach()
    print(sample.size())

    TOKEN_COUNT = 10000

    for i in range(TOKEN_COUNT-len(sample)):
        part = sample[-128:]
        prediction = predict(part, model)
        sample = torch.cat((sample,torch.tensor([prediction])))

    tokens = data.convert_id_to_token(ids=sample.tolist())
    print(tokens)

    save_stream(tokens_to_stream(tokens),"PLEASEWORK2.midi")
