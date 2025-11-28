import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf


# ---------- helpers ----------

def save_spec(y, sr, out_png, n_fft=2048, hop=512, title=""):
    """Save a spectrogram image for a time-domain signal y."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    SdB = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(8, 3))
    librosa.display.specshow(SdB, sr=sr, hop_length=hop,
                             x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def hpss_split(y, n_fft=2048, hop=512, kf=17, kt=7):
    """
    Harmonic–percussive split using librosa's HPSS.

    Returns:
        harm: harmonic component (piano+trumpet)
        perc: percussive component (drums)
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = librosa.magphase(S)

    H, P = librosa.decompose.hpss(mag, kernel_size=(kf, kt))

    eps = 1e-10
    Mh = H / (mag + eps)
    Mp = P / (mag + eps)

    Sh = Mh * S
    Sp = Mp * S

    harm = librosa.istft(Sh, hop_length=hop)
    perc = librosa.istft(Sp, hop_length=hop)
    return harm, perc


def trumpet_mask(harm, sr, n_fft=2048, hop=512,
                 harmonics=6, bandwidth_semitones=0.5):
    """
    Very simple trumpet isolator:
      - run pitch tracking on the harmonic signal
      - build a mask around multiples of f0
      - return trumpet and piano signals
    """
    S = librosa.stft(harm, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)

    # f0 tracking with PYIN
    f0, voiced, _ = librosa.pyin(
        harm,
        fmin=librosa.note_to_hz("C3"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=n_fft,
        hop_length=hop,
    )

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_freqs, n_frames = mag.shape

    # convert bandwidth from semitones to multiplicative factor
    semitone_ratio = 2 ** (1 / 12)
    bw_factor = semitone_ratio ** bandwidth_semitones

    M = np.zeros_like(mag)

    for m in range(n_frames):
        if np.isnan(f0[m]) or f0[m] <= 0:
            continue

        base = f0[m]
        for k in range(1, harmonics + 1):
            center = k * base
            low = center / bw_factor
            high = center * bw_factor
            band = (freqs >= low) & (freqs <= high)
            M[band, m] = 1.0

    # simple normalization to [0,1]
    if np.max(M) > 0:
        M = M / np.max(M)

    # apply mask
    St = M * S
    trumpet = librosa.istft(St, hop_length=hop)

    Sp = (1.0 - M) * S
    piano = librosa.istft(Sp, hop_length=hop)

    return trumpet, piano, f0


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True,
                        help="input wav file (e.g. data/demo/demo.wav)")
    parser.add_argument("--out", dest="out", default="results",
                        help="output folder (default: results)")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, "stems"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)

    # --- load audio ---
    # mono=True -> average stereo to mono
    y, sr = librosa.load(args.inp, sr=None, mono=True)

    print(f"Loaded {args.inp}, length {len(y)/sr:.2f} s, sr = {sr}")

    # save a copy of the mixture
    mix_path = os.path.join(args.out, "stems", "mixture_copy.wav")
    sf.write(mix_path, y, sr)
    save_spec(y, sr,
              os.path.join(args.out, "figs", "mix.png"),
              n_fft=args.n_fft, hop=args.hop,
              title="Mixture")

    # --- HPSS: drums vs harmonic bed ---
    harm, perc = hpss_split(
        y,
        n_fft=args.n_fft,
        hop=args.hop,
        kf=17,
        kt=7,
    )

    sf.write(os.path.join(args.out, "stems", "drums.wav"), perc, sr)
    sf.write(os.path.join(args.out, "stems", "harmonic.wav"), harm, sr)

    save_spec(perc, sr,
              os.path.join(args.out, "figs", "drums.png"),
              n_fft=args.n_fft, hop=args.hop,
              title="Drums (percussive)")
    save_spec(harm, sr,
              os.path.join(args.out, "figs", "harmonic.png"),
              n_fft=args.n_fft, hop=args.hop,
              title="Harmonic (piano+trumpet)")

    print("HPSS done: drums.wav and harmonic.wav written.")

    # --- trumpet / piano split from harmonic bed ---
    trumpet, piano, f0 = trumpet_mask(
        harm,
        sr,
        n_fft=args.n_fft,
        hop=args.hop,
        harmonics=6,
        bandwidth_semitones=0.5,
    )

    sf.write(os.path.join(args.out, "stems", "trumpet.wav"), trumpet, sr)
    sf.write(os.path.join(args.out, "stems", "piano.wav"), piano, sr)

    save_spec(trumpet, sr,
              os.path.join(args.out, "figs", "trumpet.png"),
              n_fft=args.n_fft, hop=args.hop,
              title="Trumpet (approx)")
    save_spec(piano, sr,
              os.path.join(args.out, "figs", "piano.png"),
              n_fft=args.n_fft, hop=args.hop,
              title="Piano (approx)")

    print("Trumpet/piano split done: trumpet.wav and piano.wav written.")
    print("All outputs in:", args.out)


if __name__ == "__main__":
    main()
