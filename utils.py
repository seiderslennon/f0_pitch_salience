import numpy as np
import math
import librosa
import matplotlib.pyplot as plt
import torch

def f0_to_salience_time_major(
    f0_times,
    f0_hz,
    freqs_hz,
    frame_times,
    sigma_cents=25.0,
):
    """
    Returns salience with shape (T, F):
        T = number of frames
        F = number of frequency bins
    """

    F = len(freqs_hz)
    T = len(frame_times)
    salience_FT = np.zeros((F, T), dtype=np.float32)

    log2_freqs = np.log2(freqs_hz)

    # interpolate F0 onto the HCQT/CQT frame time grid
    f0_interp = np.interp(frame_times, f0_times, f0_hz, left=0.0, right=0.0)

    for t_idx, f0 in enumerate(f0_interp):
        if f0 <= 0:
            # unvoiced → leave as zeros
            continue

        log2_f0 = math.log2(f0)
        diff_cents = 1200.0 * (log2_freqs - log2_f0)

        sal_t = np.exp(-0.5 * (diff_cents / sigma_cents) ** 2)
        m = sal_t.max()
        if m > 0:
            sal_t /= m

        salience_FT[:, t_idx] = sal_t

    # transpose to (T, F) for your generator
    salience_TF = salience_FT.T
    return salience_TF

def compute_hcqt(y, sr, harmonics=[0.5,1,2,3,4],
                 bins_per_octave=60, n_octaves=6,
                 hop_length=512, fmin=32.7):

    n_bins = bins_per_octave * n_octaves
    fmin = librosa.note_to_hz("C2")

    freqs = librosa.cqt_frequencies(
        n_bins=n_bins,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
    )

    hcqt_list = []

    for h in harmonics:
        # correct pitch shift call
        y_h = librosa.effects.pitch_shift(
            y, sr=sr, n_steps=12*np.log2(h)
        )

        C = librosa.cqt(
            y_h, sr=sr, hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            n_bins=n_bins,
        )
        Cmag = np.abs(C)  # (F, T)
        Cmag = Cmag.T     # (T, F)

        hcqt_list.append(Cmag)

    # final shape (H, T, F)
    hcqt = np.stack(hcqt_list, axis=0)
    return hcqt, freqs

def _to_numpy_if_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def plot_sample_and_label(sample, label, sr=None, hop_length=None, output_path="sample_and_label.png"):
    """
    sample: (C, T, F)
    label:  (T, F, 1)
    output_path: file to write the composite PNG visualization
    """

    sample = _to_numpy_if_tensor(sample)
    label = _to_numpy_if_tensor(label)

    if sample.ndim != 3:
        raise ValueError(f"Expected sample with 3 dims (C, T, F), got {sample.shape}")

    if label.ndim == 3 and label.shape[-1] == 1:
        label = label[..., 0]
    elif label.ndim != 2:
        raise ValueError(f"Expected label with shape (T, F) or (T, F, 1), got {label.shape}")

    C, T, F = sample.shape

    # Pick harmonic 0 (fundamental layer)
    hcqt_slice = sample[0].T        # → (F, T)
    salience_slice = label.T        # → (F, T)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 1) Plot HCQT magnitude
    im1 = ax[0].imshow(
        hcqt_slice,
        aspect='auto',
        origin='lower',
        interpolation='nearest'
    )
    ax[0].set_title("HCQT")
    ax[0].set_ylabel("Frequency bin")
    fig.colorbar(im1, ax=ax[0], fraction=0.015)

    # 2) Plot target salience
    im2 = ax[1].imshow(
        salience_slice,
        aspect='auto',
        origin='lower',
        interpolation='nearest'
    )
    ax[1].set_title("Target Salience")
    ax[1].set_xlabel("Time frames")
    ax[1].set_ylabel("Frequency bin")
    fig.colorbar(im2, ax=ax[1], fraction=0.015)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def iter_samples_for_track(hcqt, target_salience, n_time_frames):
    """
    hcqt: (C, T, F)
    target_salience: (T, F)
    yields (sample, label, t_idx)
      sample: (C, n_time_frames, F)
      label:  (n_time_frames, F, 1)
    """
    
    #GETTING slices for samples 
    if target_salience.ndim == 3 and target_salience.shape[-1] == 1:
        target_salience_base = target_salience[..., 0]
    elif target_salience.ndim == 2:
        target_salience_base = target_salience
    else:
        raise ValueError(
            "target_salience must have shape (T, F) or (T, F, 1); "
            f"got {target_salience.shape}"
        )

    T = hcqt.shape[1]
    for t_idx in range(0, T - n_time_frames + 1):
        sample = hcqt[:, t_idx : t_idx + n_time_frames, :]
        label_slice = target_salience_base[t_idx : t_idx + n_time_frames, :]
        if isinstance(label_slice, torch.Tensor):
            label = label_slice.unsqueeze(-1)
        else:
            label = label_slice[:, :, np.newaxis]
        yield sample, label, t_idx

def visualize(model, hcqt, salience):
    """
    Visualize model input, target salience, and prediction.
    hcqt:    torch.Tensor (B, C, T, F)
    salience torch.Tensor (B, T, F, 1)
    """
    model.eval()
    with torch.no_grad():
        predicted_salience = model(hcqt)

    fig = plt.figure(figsize=(12, 12))
    n_examples = min(3, hcqt.shape[0])  # show up to 3 examples

    for i in range(n_examples):

        # -------------------------
        # 1. Plot the HCQT (harmonic 1 → channel index 1)
        # -------------------------
        plt.subplot(n_examples, 3, 1 + 3*i)
        plt.imshow(hcqt[i, 1].cpu().T, origin="lower", cmap="magma", aspect='auto')
        plt.colorbar()
        plt.title("HCQT (h=1)")
        plt.axis("tight")

        # -------------------------
        # 2. Plot the TARGET salience
        # -------------------------
        plt.subplot(n_examples, 3, 2 + 3*i)
        plt.imshow(salience[i, :, :, 0].cpu().T, origin="lower", cmap="magma", aspect='auto')
        plt.colorbar()
        plt.title("Target Salience")
        plt.axis("tight")

        # -------------------------
        # 3. Plot the MODEL prediction
        # -------------------------
        plt.subplot(n_examples, 3, 3 + 3*i)
        plt.imshow(torch.sigmoid(predicted_salience[i, :, :, 0]).cpu().T,
                   origin="lower", cmap="magma", aspect='auto')
        plt.colorbar()
        plt.title("Predicted Salience")
        plt.axis("tight")

    plt.tight_layout()
    fig.savefig("model_salience.png", dpi=150, bbox_inches="tight")
    plt.close(fig)