import music21
import os
from spacy.vocab import Vocab
from music21 import midi
#music21.configure.run()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import *



files = ["AndYouAndI-Yes.mid","SiberianKhatru-Yes.mid","CloseToTheEdge-Yes.mid","InTheCourtOfKingCrimson-KingCrimson.mid"]
files = list(map(lambda x:os.path.join("midis",x),files))

def open_midi(file_path):
    mf = midi.MidiFile()
    mf.open(file_path)
    mf.read()
    mf.close()

    return midi.translate.midiFileToStream(mf)

def list_instruments(midi_obj):
    parts = midi_obj.parts.stream()
    for p in parts:
        print(p.partName)

test_midi = open_midi(files[3])

def make_note(element, pitch=None):
    if pitch is None:
        pitch = element.pitch

    start = element.offset
    duration = element.duration.quarterLength
    return {
        "pitch": pitch.midi,
        "start": start,
        "end": start + duration,
        "velocity": getattr(element, "volume", None),
        "instrument": getattr(element, "instrument", 0)
    }


def get_notes(stream):
    notes = []
    for i in stream.flatten():
        if isinstance(i,music21.note.Note):
            notes.append(make_note(i))

        elif isinstance(i, music21.chord.Chord):
            for pitch in i.pitches:
                notes.append(make_note(i,pitch=pitch))

    return notes

def get_events(notes):
    events = []
    for note in notes:
        events.append({
            "type":"ON",
            "pitch":note["pitch"],
            "time": note["start"]
        })

        events.append({
            "type": "OFF",
            "pitch": note["pitch"],
            "time": note["end"]
        })
    return events

notes = get_notes(test_midi)
events = get_events(notes)
print(events)

WAIT_DURATIONS = {i*.25:f"WAIT_{i}" for i in range(1,33)}
ON_TOKENS = {i:f"NOTE_ON_{i}" for i in range(128)}
OFF_TOKENS = {i:f"NOTE_OFF_{i}" for i in range(128)}


T_V = list(WAIT_DURATIONS.values()) + list(ON_TOKENS.values()) + list(OFF_TOKENS.values())
TOKEN_VOCAB = Vocab()
for t in T_V:
    temp = TOKEN_VOCAB[t]

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

        while dtime > .25:
            for i in range(32,0,-1):
                t = i*.25
                if t<dtime:
                    dtime-=t
                    tokens.append(WAIT_DURATIONS[t])

        tokens.append(curr_token)
        curr_time = time

    return tokens

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

class music_snippet_dataset(Dataset):
    def __init__(self,song_paths,snippet_length):
        self.features = []
        self.labels = []

        self.unique_tokens = set()
        self.token_to_id = {}
        self.id_to_token = {}
        for path in song_paths:
            midi = open_midi(path)
            notes = get_notes(midi)
            events = get_events(notes)
            tokens = events_to_tokens(events)
            for i in range(len(tokens)):
                if tokens[i] not in self.unique_tokens:
                    self.token_to_id[tokens[i]] = len(self.unique_tokens)
                    self.id_to_token[len(self.unique_tokens)] = tokens[i]
                    self.unique_tokens.add(tokens[i])
                tokens[i] = self.token_to_id[tokens[i]]
            for i in range(0,len(tokens)-snippet_length):
                self.features.append(tokens[i:i+snippet_length])
                self.labels.append(tokens[i+snippet_length])
        self.features = torch.tensor(self.features)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


NUM_TOKENS = len(TOKEN_VOCAB)
data = music_snippet_dataset([files[3]],128)



EMBEDDING_DIM = 256
HEADS = 4
MAX_TOKENS= 128
VOCAB_SIZE = len(data.unique_tokens)
LAYERS = 2

model = DecoderOnlyTransformer(
    EMBEDDING_DIM=EMBEDDING_DIM,
    HEADS=HEADS,
    MAX_TOKENS=MAX_TOKENS,
    VOCAB_SIZE=VOCAB_SIZE,
    LAYERS=LAYERS,
)

print(model.forward(torch.unsqueeze(data[0][0],0), torch.unsqueeze(data[0][1],0)))