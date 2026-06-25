import music21
import os

import torch
import tqdm
import json
from music21 import midi
#music21.configure.run()

from torch.utils.data import Dataset, random_split
import torch.optim as optim
from model import *



files = [
    "AndYouAndI-Yes.mid","SiberianKhatru-Yes.mid","CloseToTheEdge-Yes.mid", #from Yes' Close To The Edge (midis seem to be a bit messed up)
    "InTheCourtOfKingCrimson-KingCrimson.mid" #from King Crimson's ITCOTCK
]
files = list(map(lambda x:os.path.join("midis",x),files)) #converts the file names to their actual relative paths


#opens midi into a music21 midi file from the path
def open_midi(file_path):
    mf = midi.MidiFile()
    mf.open(file_path)
    mf.read()
    mf.close()

    return midi.translate.midiFileToStream(mf)


#lists all instruments (i.e., parts) in the given midi object (for debugging purposes)
def list_instruments(midi_obj):
    parts = midi_obj.parts.stream()
    for p in parts:
        print(p.partName)


#converts a given music21 element to a dictionary which contains important info about the note
def make_note(element, pitch=None):
    if pitch is None:
        pitch = element.pitch

    # start time of element measured in quarter-notes since the start of the song
    start = element.offset
    # the duration of the element in notes (ie, quarter-note = 1, eighth-note = .5)
    duration = element.duration.quarterLength

    return {
        "pitch": pitch.midi,
        "start": start,
        "end": start + duration,
        "velocity": getattr(element, "volume", None),
        "instrument": getattr(element, "instrument", 0)
    }


#gets all the note dictionaries from the midi stream and returns them in a list
def get_notes(stream):
    notes = []

    for i in stream.flatten():
        #"if this element is a music note"
        if isinstance(i,music21.note.Note):
            notes.append(make_note(i))

        #"if this element is a chord"
        elif isinstance(i, music21.chord.Chord):
            #creates a note for each individual pitch in the chord
            for pitch in i.pitches:
                notes.append(make_note(i,pitch=pitch))
    return notes


#converts each note to two events specifying the time of it starting and ending
def get_events(notes):
    events = []
    for note in notes:
        #pitch on event
        events.append({
            "type":"ON",
            "pitch":note["pitch"],
            "time": note["start"]
        })

        #pitch off event
        events.append({
            "type": "OFF",
            "pitch": note["pitch"],
            "time": note["end"]
        })
    return events


#test_midi = open_midi(files[3])
#notes = get_notes(test_midi)
#events = get_events(notes)
#print(events)


# maps each wait time from 0.25-8 units to a corresponding wait token
# WAIT_i = wait i 16th notes
WAIT_DURATIONS = {i*.25:f"WAIT_{i}" for i in range(1,33)}

#pitch on and off tokens for each pitch (the midi pitches go from 0-127)
ON_TOKENS = {i:f"NOTE_ON_{i}" for i in range(128)}
OFF_TOKENS = {i:f"NOTE_OFF_{i}" for i in range(128)}


#converts list of on and off events to on and off tokens with wait tokens
def events_to_tokens(events):
    tokens = []

    curr_time = 0
    for e in events:
        note_type = e["type"]
        pitch = e["pitch"]
        time = e["time"]

        if note_type == "OFF":
            curr_token = OFF_TOKENS[pitch]
        elif note_type == "ON":
            curr_token = ON_TOKENS[pitch]

        dtime = time-curr_time

        #breaks down the time between this and the last token into multiple wait tokens until the duration is under .25 (16th note)
        while dtime > .25:
            #fits the largest possible wait
            for i in range(32,0,-1):
                t = i*.25
                #if this duration fits, create the token
                if t<dtime:
                    dtime-=t
                    tokens.append(WAIT_DURATIONS[t])

        #adds the note token after adding all the wait tokens before it
        tokens.append(curr_token)
        curr_time = time

    return tokens


#gets pytorch data loader from the given dataset + other info
def get_loader(data,batch_size,collate=None,shuffle=False):
    return torch.utils.data.DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
        #the preprocessing function
        # collate_fn=collate,
        # pin_memory=True,
        # num_workers=2,
        # persistent_workers=True
    )


#class to represent a dataset of music snippets
#given songs get trasnformed into multiple [(tokens i-j), token j+1] feature and token items
#tokens are converted to unique ids
class music_snippet_dataset(Dataset):
    # noinspection PyTypeChecker
    def __init__(self,song_paths,snippet_length):
        self.features = []
        self.labels = []

        self.unique_tokens = set()
        self.token_to_id = {}
        self.id_to_token = {}

        #iterates over every given song to create samples from
        for path in song_paths:
            midi = open_midi(path)
            notes = get_notes(midi)
            events = get_events(notes)
            tokens = events_to_tokens(events)

            # converts each token to id
            for i in range(len(tokens)):
                #if token is new, create an id for it
                if tokens[i] not in self.unique_tokens:
                    self.token_to_id[tokens[i]] = len(self.unique_tokens)
                    self.id_to_token[len(self.unique_tokens)] = tokens[i]
                    self.unique_tokens.add(tokens[i])
                #converts token to id
                tokens[i] = self.token_to_id[tokens[i]]

            #creates samples using a sliding window of width snippet_length
            for i in range(0,len(tokens)-snippet_length-1):
                self.features.append(tokens[i:i+snippet_length])
                self.labels.append(tokens[i+1:i+snippet_length+1])

        #turns features and labels into tensors
        self.features = torch.tensor(self.features)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.features)

    #gets specific sample from index
    def __getitem__(self, i):
        return self.features[i], self.labels[i]


data = music_snippet_dataset([files[3]],128)

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
MODEL_NAME = "RobertFripp2"
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

    sample = data[0][0]
    print(sample)


    print(predict(sample))
