import os
import numpy as np
import matplotlib.pyplot as plt

def parse_rttm_file(file_path):
    segments = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == 'SPEAKER':
                file_id = parts[1]
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                segments.append((file_id, speaker_id, start_time, duration))
    return segments

def get_segments_from_directory(directory):
    all_segments = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.rttm'):
            file_path = os.path.join(directory, file_name)
            segments = parse_rttm_file(file_path)
            all_segments.extend(segments)
    return all_segments



def decoupe_segments(segments, segment_duration, max_time):
    """
    Divise les segments en intervalles de temps fixes et place chaque segment dans le bon intervalle.

    Parameters:
    segments (list of tuples): La liste des segments, chaque segment étant (file_id, speaker_id, start_time, duration).
    segment_duration (int): La durée de chaque intervalle de temps en secondes.
    max_time (int): Le temps maximum de l'enregistrement en secondes.

    Returns:
    list of lists: Une liste où chaque élément est une liste de segments présents dans cet intervalle de temps.
    """
    # Calculer le nombre d'intervalles de temps nécessaires
    num_intervals = (max_time + segment_duration - 1) // segment_duration
    
    # Initialiser une liste de listes pour stocker les segments par intervalle de temps
    decoupled_segments = [[] for _ in range(num_intervals)]

    for segment in segments:
        file_id, speaker_id, start_time, duration = segment
        end_time = start_time + duration

        current_start = start_time
        while current_start < end_time:
            current_end = min(current_start + segment_duration, end_time)
            interval_index = int(current_start // segment_duration)
            new_duration = current_end - current_start
            decoupled_segments[interval_index].append((file_id, speaker_id, current_start, new_duration))
            current_start = current_end

    return decoupled_segments









def plot_segments(segments, ax,tick, title="Speaker Segments Over Time"):
    speakers = sorted(set([seg[1] for seg in segments]))
    speaker_to_y = {speaker: i for i, speaker in enumerate(speakers)}

    for rec_id, speaker, start, duration in segments:
        start = float(start)
        duration = float(duration)
        y = speaker_to_y[speaker]
        color = 'blue'
        ax.plot([start, start + duration], [y, y], color=color, label=speaker)

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speakers")
    ax.set_title(title)
    ax.grid(True)

    # Configurer les ticks de l'axe x pour qu'ils soient affichés toutes les 25 secondes
    max_time = max([start + duration for _, _, start, duration in segments])
    ax.set_xticks(np.arange(0, max_time + tick, tick))

    # Eviter les labels dupliqués dans la légende
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    
    
def filter_segments_before_n_seconds(segments, n):
    filtered_segments = []
    for seg in segments:
        start_time = seg[2]
        duration = seg[3]
        end_time = start_time + duration

        if start_time < n:
            if end_time > n:
                duration = n - start_time
            filtered_segments.append((seg[0], seg[1], start_time, duration))
    return filtered_segments


def plot_segments_dec(decoupled_segments, segment_duration, ax,tick, title="Speaker Segments Over Time"):
    # Obtenir tous les speakers à partir des segments découpés
    all_segments = [seg for interval in decoupled_segments for seg in interval]
    speakers = sorted(set([seg[1] for seg in all_segments]))
    speaker_to_y = {speaker: i for i, speaker in enumerate(speakers)}

    for interval_idx, interval in enumerate(decoupled_segments):
        for rec_id, speaker, start, duration in interval:
            start = float(start)
            duration = float(duration)
            y = speaker_to_y[speaker]
            color = 'blue'
            ax.plot([start, start + duration], [y, y], color=color, label=speaker)

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speakers")
    ax.set_title(title)
    ax.grid(True)

    # Configurer les ticks de l'axe x pour qu'ils soient affichés toutes les 25 secondes
    max_time = segment_duration * len(decoupled_segments)
    ax.set_xticks(np.arange(0, max_time + tick, tick))

    # Eviter les labels dupliqués dans la légende
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())