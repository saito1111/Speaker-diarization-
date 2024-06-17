# audio_analysis.py

import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torchaudio.transforms import Spectrogram, AmplitudeToDB
from IPython.display import Audio, display

def load_audio(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    waveform, sample_rate = torchaudio.load(file_path)
    print(f"Loaded {file_path} with sample rate {sample_rate} and waveform shape {waveform.shape}")
    return waveform, sample_rate

def compute_af(theta, frequencies, distance_between_mics, c):
    delta_m = distance_between_mics
    v_m_theta = (2 * np.pi * frequencies * delta_m * np.cos(theta) / c).unsqueeze(1)
    print(f"Computed AF for θ={theta} with shape {v_m_theta.shape}")
    return v_m_theta

def compute_metrics(waveforms, sample_rate):
    distance_between_mics = 0.1  # en mètres (à ajuster selon la configuration)
    c = 343  # vitesse du son en m/s    
    print("Starting computation of metrics...")

    print("Calculating spectrograms...")
    transform = Spectrogram(n_fft=2048, hop_length=1024, power=None)
    specs = [transform(waveform) for waveform in waveforms]
    for idx, spec in enumerate(specs):
        print(f"Spectrogram shape for microphone {idx + 1}: {spec.shape}")

    print("Calculating LPS...")
    amplitude_to_db = AmplitudeToDB()
    lps = [amplitude_to_db(spec.abs()) for spec in specs]
    for idx, lp in enumerate(lps):
        print(f"LPS shape for microphone {idx + 1}: {lp.shape}")

    print("Calculating phases...")
    phases = [torch.angle(spec) for spec in specs]
    for idx, phase in enumerate(phases):
        print(f"Phase shape for microphone {idx + 1}: {phase.shape}")

    # Calculate IPD for all pairs of microphones
    num_mics = len(specs)
    ipd_pairs = {}
    print("Calculating IPD for all pairs of microphones...")
    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            print(f"Calculating IPD for Microphones {i + 1} and {j + 1}")
            ipd_pairs[(i, j)] = torch.angle(specs[i]) - torch.angle(specs[j])
            print(f"IPD shape for microphones {i + 1} and {j + 1}: {ipd_pairs[(i, j)].shape}")

    # Calculate AF for all pairs of microphones
    theta_values = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi, 1.25 * np.pi, 1.5 * np.pi, 1.75 * np.pi]
    frequencies = torch.linspace(0, sample_rate // 2, specs[0].size(-2))
    af_dict = {}
    print("Calculating AF for all pairs of microphones...")
    for theta in theta_values:
        af_sum = torch.zeros((frequencies.size(0), specs[0].size(-1))).clone()
        for (i, j), ipd in ipd_pairs.items():
            print(f"Calculating AF contribution for Microphones {i + 1} and {j + 1} with θ={theta} radians")
            v_m_theta = compute_af(theta, frequencies, distance_between_mics, c).reshape(-1, 1)
            af_sum += torch.cos(v_m_theta - ipd.squeeze(0))
        af_dict[theta] = af_sum
        print(f"AF shape with θ={theta} radians: {af_dict[theta].shape}")

    print("Finished computation of metrics.")
    return specs, lps, phases, ipd_pairs, af_dict

def plot_spectrograms(lps, sample_rate, duration=10):
    subsample_length = duration * sample_rate // 1024  # Assuming hop_length is 1024
    times = np.linspace(0, duration, subsample_length)
    frequencies = torch.linspace(0, sample_rate // 2, lps[0].size(-2))

    plt.figure(figsize=(12, 12))
    for i, lp in enumerate(lps):
        plt.subplot(len(lps), 1, i + 1)
        plt.pcolormesh(times, frequencies.numpy(), lp[0][:, :subsample_length].numpy(), shading='gouraud')
        plt.title(f'Spectrogramme d\'Amplitude - Microphone {i + 1}')
        plt.ylabel('Fréquence [Hz]')
        plt.colorbar(label='Amplitude [dB]')
    plt.tight_layout()
    plt.show()

def plot_phases(phases, sample_rate, duration=10):
    subsample_length = duration * sample_rate // 1024  # Assuming hop_length is 1024
    times = np.linspace(0, duration, subsample_length)
    frequencies = torch.linspace(0, sample_rate // 2, phases[0].size(-2))

    plt.figure(figsize=(12, 12))
    for idx, phase in enumerate(phases):
        plt.subplot(len(phases), 1, idx + 1)
        plt.pcolormesh(times, frequencies.numpy(), phase[0, :, :subsample_length].numpy(), shading='gouraud')
        plt.title(f'Phase - Microphone {idx + 1}')
        plt.ylabel('Fréquence [Hz]')
        plt.colorbar(label='Phase [radians]')
    plt.tight_layout()
    plt.show()

def plot_ipd_pairs(ipd_pairs, sample_rate, duration=10):
    subsample_length = duration * sample_rate // 1024  # Assuming hop_length is 1024
    times = np.linspace(0, duration, subsample_length)
    frequencies = torch.linspace(0, sample_rate // 2, list(ipd_pairs.values())[0].size(-2))

    # Identify the first and last keys in the dictionary
    first_key = list(ipd_pairs.keys())[0]
    last_key = list(ipd_pairs.keys())[-1]

    plt.figure(figsize=(12, 6))
    for idx, (i, j) in enumerate([first_key, last_key]):
        ipd = ipd_pairs[(i, j)]
        plt.subplot(2, 1, idx + 1)
        plt.pcolormesh(times, frequencies.numpy(), ipd[0][:, :subsample_length].numpy(), shading='gouraud')
        plt.title(f'IPD - Microphones {i + 1} et {j + 1}')
        plt.ylabel('Fréquence [Hz]')
        plt.colorbar(label='Phase [radians]')
    plt.tight_layout()
    plt.show()    

def plot_af_pairs(af_dict, sample_rate, duration=10):
    subsample_length = duration * sample_rate // 1024  # Assuming hop_length is 1024
    times = np.linspace(0, duration, subsample_length)
    frequencies = torch.linspace(0, sample_rate // 2, list(af_dict.values())[0].size(0))
    theta_values = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi, 1.25 * np.pi, 1.5 * np.pi, 1.75 * np.pi]

    plt.figure(figsize=(12, 24))  # Adjust the figure size for more subplots
    for idx, theta in enumerate(theta_values):
        af = af_dict[theta]
        plt.subplot(len(theta_values), 1, idx + 1)
        plt.pcolormesh(times, frequencies.numpy(), af[:, :subsample_length].numpy(), shading='gouraud', vmin=0, vmax=25)
        plt.title(f'AFθ - θ={theta} rad')
        plt.ylabel('Fréquence [Hz]')
        plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()



