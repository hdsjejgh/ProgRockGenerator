import music21
import os

from music21 import midi
#music21.configure.run()

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


TOKEN_VOCAB = list(WAIT_DURATIONS.values()) + list(ON_TOKENS.values()) + list(OFF_TOKENS.values())

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


print(events_to_tokens(events))