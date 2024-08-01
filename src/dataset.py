import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import log_loss
import numpy as np
from src.metrics import *
from src.gest_segm import *




def extract_features(waveform_segment, sample_rate=16000, n_fft=512, hop_length=256, theta_values=[0, np.pi/2, np.pi, 3*np.pi/2], mic_pairs=[(0, 4), (1, 5), (2, 6), (3, 7)]):
    """
    Extract Log Power Spectrum (LPS), Inter-channel Phase Difference (IPD), and Angle Feature (AF) from the audio signal.
    
    Parameters:
    waveforms (list of np.array): List of audio signals from multiple microphones.
    sample_rate (int): Sample rate of the audio signals.
    n_fft (int): FFT size.
    hop_length (int): Hop length for STFT.
    theta_values (list of float): List of angles in radians for computing AF.
    mic_pairs (list of tuples): Pairs of microphones to compute IPD.
    
    Returns:
    torch.Tensor: Concatenated feature array (LPS, IPD, AF).
    """
    specs, lps, phases, ipd_pairs, af_dict = compute_metrics(waveform_segment, sample_rate, n_fft, hop_length, theta_values, mic_pairs)

    print("Processing IPD features...")
    ipd_array_list = [torch.tensor(ipd_pairs[pair]) for pair in mic_pairs]
    ipd_array = torch.cat(ipd_array_list, dim=0)  # Concatenate along the channel axis
    print(f"IPD array shape: {ipd_array.shape}")

    print("Processing AF features...")
    af_array_list = [torch.tensor(af_dict[theta]).unsqueeze(0) for theta in theta_values]  # Add an extra dimension
    af_array = torch.cat(af_array_list, dim=0)  # Concatenate along the channel axis
    print(f"AF array shape: {af_array.shape}")

    print("Processing LPS features...")
    lps_array_list = [torch.tensor(lp) for lp in lps]
    lps_array = torch.cat(lps_array_list, dim=0)  # Concatenate along the channel axis
    print(f"LPS array shape: {lps_array.shape}")

    # Ensure all arrays have the same number of frames (time steps)
    max_time_steps = max(lps_array.shape[2], ipd_array.shape[2], af_array.shape[2])
    if lps_array.shape[2] < max_time_steps:
        padding = (0, 0, 0, max_time_steps - lps_array.shape[2])
        lps_array = torch.nn.functional.pad(lps_array, padding, mode='constant', value=0)  # Pad along time dimension
    if ipd_array.shape[2] < max_time_steps:
        padding = (0, 0, 0, max_time_steps - ipd_array.shape[2])
        ipd_array = torch.nn.functional.pad(ipd_array, padding, mode='constant', value=0)  # Pad along time dimension
    if af_array.shape[2] < max_time_steps:
        padding = (0, 0, 0, max_time_steps - af_array.shape[2])
        af_array = torch.nn.functional.pad(af_array, padding, mode='constant', value=0)  # Pad along time dimension

    print("Concatenating all features into a single feature array...")
    features = torch.cat((lps_array, ipd_array, af_array), dim=0)
    print(f"Final feature array shape: {features.shape}")

    return features




class DiarizationDataset(Dataset):
    def __init__(self, features, labels_vad, labels_osd):
       
        
        self.features = features
        self.labels_vad = labels_vad
        self.labels_osd = labels_osd
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        vad_label = self.labels_vad[:, idx]
        osd_label = self.labels_osd[idx]
        return feature, vad_label, osd_label
    
def build_dataset(file_directory, rttm_directory, segment_duration=4, sample_rate=16000, n_fft=512, hop_length=256, theta_values=[0, np.pi/2, np.pi, 3*np.pi/2], mic_pairs=[(0, 4), (1, 5), (2, 6), (3, 7)], num_speakers=4):
    all_segments = get_segments_from_directory(rttm_directory)
    decoupled_segments = decoupe_segments(all_segments, segment_duration)
    
    num_frames = int(segment_duration * sample_rate / hop_length)
    frame_size = hop_length / sample_rate

    waveform_segments, sample_rate = get_waveform(file_directory, segment_duration)
    features = []
    labels_vad = []
    labels_osd = []

    for i in range(0, len(waveform_segments)):
        segment_batch = decoupled_segments[i*num_speakers:(i+1)*num_speakers]
        if len(segment_batch) < num_speakers:
            break

        waveform = np.array(waveform_segments[i])  # Convert list to numpy array
        feature = extract_features(waveform, sample_rate, n_fft, hop_length, theta_values, mic_pairs)
        vad_label = segments_to_vad_labels(segment_batch, num_speakers, num_frames, frame_size)
        osd_label = segments_to_osd_labels(segment_batch, num_frames, frame_size)

        features.append(feature)
        labels_vad.append(vad_label)
        labels_osd.append(osd_label)
    
    features = torch.stack(features)
    labels_vad = torch.stack(labels_vad)
    labels_osd = torch.stack(labels_osd)

    dataset = DiarizationDataset(features, labels_vad, labels_osd)
    return dataset
#/////////////////////////TEST//////////////
def generate_vad_labels1(decoupled_segments, segment_duration, frame_size, num_speakers):
    num_frames_per_segment = int(segment_duration / frame_size)
    batch_size = len(decoupled_segments)
    seq_len_2 = num_frames_per_segment  # Assuming seq_len*2 is the same as the number of frames per segment

    vad_labels = torch.zeros((batch_size, seq_len_2, num_speakers))

    # Création d'un dictionnaire pour mapper les identifiants de locuteurs à des indices numériques
    speaker_to_idx = {}
    current_idx = 0

    for batch_idx, interval in enumerate(decoupled_segments):
        for file_id, speaker_id, start, duration in interval:
            start_frame = int((start % segment_duration) / frame_size)
            end_frame = int(((start + duration) % segment_duration) / frame_size)
            
            if speaker_id not in speaker_to_idx:
                speaker_to_idx[speaker_id] = current_idx
                current_idx += 1

            speaker_idx = speaker_to_idx[speaker_id]
            if speaker_idx >= num_speakers:
                raise ValueError(f"Number of unique speakers exceeds the specified num_speakers ({num_speakers})")
                
            vad_labels[batch_idx, start_frame:end_frame, speaker_idx] = 1

    return vad_labels

def generate_vad_labels(decoupled_segments, segment_duration, frame_size, num_speakers):
    num_frames_per_segment = int(segment_duration / frame_size)
    batch_size = len(decoupled_segments)
    seq_len_2 = num_frames_per_segment  # Assuming seq_len*2 is the same as the number of frames per segment

    vad_labels = torch.zeros((batch_size, seq_len_2, num_speakers))

    # Création d'un dictionnaire pour mapper les identifiants de locuteurs à des indices numériques
    speaker_to_idx = {}
    current_idx = 0

    for batch_idx, interval in enumerate(decoupled_segments):
        print(f"Processing batch {batch_idx+1}/{batch_size}")
        for file_id, speaker_id, start, duration in interval:
            current_start = start
            current_end = start + duration
            print(f"  Processing segment: speaker_id={speaker_id}, start={start}, duration={duration}")
            
            while current_start < current_end:
                start_frame = int((current_start % segment_duration) / frame_size)
                end_frame = min(num_frames_per_segment, int(((current_start + frame_size) % segment_duration) / frame_size))
                
                # Ajouter une condition pour éviter la boucle infinie
                if start_frame >= end_frame:
                    end_frame = start_frame + 1
                
                print(f"    Current start: {current_start}, end: {current_end}, start_frame: {start_frame}, end_frame: {end_frame}")

                if speaker_id not in speaker_to_idx:
                    speaker_to_idx[speaker_id] = current_idx
                    current_idx += 1

                speaker_idx = speaker_to_idx[speaker_id]
                if speaker_idx >= num_speakers:
                    raise ValueError(f"Number of unique speakers exceeds the specified num_speakers ({num_speakers})")

                vad_labels[batch_idx, start_frame:end_frame, speaker_idx] = 1

                # Update current_start to the end of the current segment part
                previous_start = current_start
                current_start += (end_frame - start_frame) * frame_size
                print(f"    Updated current_start: {current_start}")

                # Break condition to avoid infinite loop
                if current_start == previous_start:
                    print("    Breaking loop to avoid infinite loop.")
                    break

                if end_frame == num_frames_per_segment and current_start < current_end:
                    batch_idx += 1
                    print(f"    Moving to next batch: {batch_idx}")
                    if batch_idx >= batch_size:
                        raise ValueError("Segments extend beyond the provided batch size.")
                    current_start = batch_idx * segment_duration
                    print(f"    Updated current_start for new batch: {current_start}")

    return vad_labels






def generate_osd_labels(decoupled_segments, segment_duration, frame_size):
    num_frames_per_segment = int(segment_duration / frame_size)
    batch_size = len(decoupled_segments)
    seq_len_2 = num_frames_per_segment  # Assuming seq_len*2 is the same as the number of frames per segment

    osd_labels = torch.zeros((batch_size, seq_len_2))

    for batch_idx, interval in enumerate(decoupled_segments):
        frame_activity = torch.zeros(seq_len_2)

        for file_id, speaker_id, start, duration in interval:
            start_frame = int((start % segment_duration) / frame_size)
            end_frame = int(((start + duration) % segment_duration) / frame_size)
            frame_activity[start_frame:end_frame] += 1

        osd_labels[batch_idx] = (frame_activity > 1).float()

    return osd_labels


def plot_generated_labels(vad_labels, segment_duration, frame_size, max_time,tick):
    """
    Trace les labels VAD générés.

    Parameters:
    vad_labels (torch.Tensor): Les labels VAD de forme [batch_size, seq_len*2, num_speakers].
    segment_duration (int): La durée de chaque segment en secondes.
    frame_size (float): La durée de chaque frame en secondes.
    max_time (int): Le temps maximum de l'enregistrement en secondes.
    max_batches (int): Nombre maximum de batches à tracer pour une meilleure lisibilité.
    """
    num_frames_per_segment = int(segment_duration / frame_size)
    num_speakers = vad_labels.shape[2]
    batch_size = vad_labels.shape[0]
    seq_len_2 = vad_labels.shape[1]


    fig, axes = plt.subplots(num_speakers, 1, figsize=(12, 12), sharex=True, sharey=True)

    for speaker_idx in range(num_speakers):
        ax = axes[speaker_idx]
        for batch_idx in range(batch_size):
            for frame_idx in range(seq_len_2):
                if vad_labels[batch_idx, frame_idx, speaker_idx] == 1:
                    start_time = frame_idx * frame_size
                    end_time = start_time + frame_size
                    ax.plot([start_time, end_time], [batch_idx, batch_idx], color='blue')
                    ax.set_yticks(range(batch_size))
                    ax.set_yticklabels([f'Batch {i+1}' for i in range(batch_size)])
                    ax.set_ylabel(f'Speaker {speaker_idx+1}')
                    ax.grid(True)    
           

    max_time = segment_duration * (seq_len_2 // num_frames_per_segment)
    plt.xticks(np.arange(0, max_time + tick, tick))
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()