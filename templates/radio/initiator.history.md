## user

simulator\@radio:\~\$ tune 94.7 && scan 94.4..95.0 -v

## assistant

── FREQ: 94.7 MHz SNR: ███████░░ MOD: FM? LOCK: off
STATIONS: MOSS_GARDEN (weak idents), ??? (sporadic jingle)
SIDEBANDS: −75 kHz, +120 kHz HAZARDS: multipath shimmer, ad-spur ghosts

Spectral hints: sideband at +120 kHz shows seasonal PSA harmonics.
simulator\@radio:\~\$

## user

simulator\@radio:\~\$ lock && demod FM && record 5 && qsl "MOSS_GARDEN"

## assistant

LOCK: on
DEMUX: FM stereo pilot detected (19 kHz)
RECORD: 5s captured → note: \[soft pads + field-recording of rain]
QSL: issued card #MG-94.7 with tags \[mythic]\[ecology]\[night]

── FREQ: 94.7 MHz SNR: ████████░ MOD: FM LOCK: on
STATIONS: MOSS_GARDEN (confirmed)
SIDEBANDS: −75 kHz (ad spur), +120 kHz (PSA harmonic) HAZARDS: mild fade

Next: `spectrogram` or `annotate "rain- pads"`
simulator\@radio:\~\$
