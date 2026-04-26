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
list_instruments(test_midi)