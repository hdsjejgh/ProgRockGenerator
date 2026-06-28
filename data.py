import music21
from music21 import midi
import torch
from torch.utils.data import Dataset


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

def tokens_to_stream(tokens):
    pass

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