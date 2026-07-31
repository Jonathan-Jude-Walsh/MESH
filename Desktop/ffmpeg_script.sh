#!/usr/bin/env bash
# audio source https://github.com/solutionprovider9174/shipsear

IN="/home/jonathan/Documents/Input Dataset/ShipsEar"
OUTDIR="/home/jonathan/Documents/Input Dataset/Standardise"

mkdir -p "$OUTDIR"

# Safe, reliable file walker
find "$IN" -type f -name "*.wav" | while IFS= read -r FILE; do

    # Remove input prefix
    REL="${FILE#$IN/}"

    # Extract category + recording folder
    CATEGORY=$(echo "$REL" | cut -d/ -f1)
    RECORDING=$(echo "$REL" | cut -d/ -f2)
    BASENAME=$(basename "$FILE" .wav)

    # Mirror directory structure
    WORKDIR="$OUTDIR/$CATEGORY/$RECORDING"
    mkdir -p "$WORKDIR"

    echo "Processing: $CATEGORY / $RECORDING / $BASENAME.wav"

    # -----------------------------
    # 1. Standardise
    # -----------------------------
    STD="$WORKDIR/${BASENAME}_standardised.wav"

    ffmpeg -y -i "$FILE" \
        -ac 1 \
        -ar 96000 \
        -c:a pcm_s24le \
        "$STD"

    # -----------------------------
    # 2. Segment (5 s windows)
    # -----------------------------
    SEGDIR="$WORKDIR/segments"
    mkdir -p "$SEGDIR"

    ffmpeg -y -i "$STD" \
        -f segment \
        -segment_time 5 \
        -c copy \
        "$SEGDIR/seg_%04d.wav"

    # -----------------------------
    # 3. Pre-processing
    # -----------------------------
    PROCDIR="$WORKDIR/processed"
    mkdir -p "$PROCDIR"

    find "$SEGDIR" -type f -name "*.wav" | while IFS= read -r S; do
        SEGBASE=$(basename "$S")
        OUT="$PROCDIR/$SEGBASE"

        ffmpeg -y -i "$S" \
            -af "highpass=f=5,lowpass=f=1600,loudnorm=I=-16:LRA=11:TP=-1.5" \
            -c:a pcm_s24le \
            "$OUT"
    done

    echo "Finished: $CATEGORY / $RECORDING"

done

echo "All processing complete."
