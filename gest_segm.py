import os
import matplotlib.pyplot as plt

def parse_rttm_file(file_path):
    segments = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == 'SPEAKER':
                rec_id = parts[1]
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                segments.append((rec_id, speaker_id, start_time, duration))
    return segments

def get_segments_from_directory(directory):
    all_segments = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.rttm'):
            file_path = os.path.join(directory, file_name)
            segments = parse_rttm_file(file_path)
            all_segments.extend(segments)
    return all_segments

def plot_segments(segments, title="Speaker Segments Over Time"):
    speakers = sorted(set([seg[1] for seg in segments]))
    speaker_to_y = {speaker: i for i, speaker in enumerate(speakers)}

    plt.figure(figsize=(12, 8))
    for rec_id, speaker, start, duration in segments:
        y = speaker_to_y[speaker]
        plt.plot([start, start + duration], [y, y], label=speaker)
    
    plt.yticks(range(len(speakers)), speakers)
    plt.xlabel("Time (s)")
    plt.ylabel("Speakers")
    plt.title(title)
    plt.grid(True)
    plt.show()

def filter_segments_before_n_seconds(segments, n):
    filtered_segments = [seg for seg in segments if seg[2] < n]
    return filtered_segments