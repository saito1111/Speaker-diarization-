import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torchaudio.transforms import Spectrogram, AmplitudeToDB
from IPython.display import Audio, display
import logging
import torch
import numpy as np
from torchaudio.transforms import Spectrogram, AmplitudeToDB



def load_audio(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    waveform, sample_rate = torchaudio.load(file_path)
    print(f"Loaded {file_path} with sample rate {sample_rate} and waveform shape {waveform.shape}")
    return waveform, sample_rate

def get_waveform(file_directory, segment_duration=4):
    """
    Récupère les waveforms des différents microphones et les divise en segments de 4 secondes.
    
    Parameters:
    file_directory (str): Le répertoire contenant les fichiers audio.
    segment_duration (int): La durée de chaque segment en secondes.
    
    Returns:
    list of list of np.array: Liste des segments de waveforms pour chaque segment de temps.
    int: Le taux d'échantillonnage des fichiers audio.
    """
    waveforms = []
    sample_rate = None

    for mic_num in range(1, 9):
        sample_wav = os.path.join(file_directory, f"ES2002c.Array1-0{mic_num}.wav")
        print(f"Processing {sample_wav}")

        # Charger le fichier audio
        waveform, sample_rate = load_audio(sample_wav)
        waveforms.append(waveform)

    # Diviser les waveforms en segments de 4 secondes
    segment_length = segment_duration * sample_rate
    num_segments = len(waveforms[0][0]) // segment_length

    # Initialiser une liste de listes pour stocker les segments pour chaque temps et microphone
    all_segments = [[None for _ in range(8)] for _ in range(num_segments)]

    for mic_idx, waveform in enumerate(waveforms):
        for seg_idx in range(num_segments):
            segment = waveform[:, seg_idx*segment_length:(seg_idx+1)*segment_length]
            all_segments[seg_idx][mic_idx] = segment

    # Vérifier et ignorer les segments partiels
    if len(waveforms[0][0]) % segment_length != 0:
        all_segments.pop()

    return all_segments, sample_rate

def calculate_distance(mic_position1, mic_position2):
    """
    Calcule la distance entre deux microphones.

    :param mic_position1: tuple (x1, y1) représentant la position du premier microphone
    :param mic_position2: tuple (x2, y2) représentant la position du deuxième microphone
    :return: distance entre les deux microphones
    """
    x1, y1 = mic_position1
    x2, y2 = mic_position2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_relative_angle(look_direction, mic_position1, mic_position2):
    """
    Calcule l'angle relatif entre la direction de regard et la paire de microphones.

    :param look_direction: direction de regard θ en radians
    :param mic_position1: tuple (x1, y1) représentant la position du premier microphone
    :param mic_position2: tuple (x2, y2) représentant la position du deuxième microphone
    :return: angle relatif entre la direction de regard et la paire de microphones
    """
    x1, y1 = mic_position1
    x2, y2 = mic_position2
    mid_point_x = (x1 + x2) / 2
    mid_point_y = (y1 + y2) / 2
    angle_mic_pair = np.arctan2(mid_point_y, mid_point_x)
    relative_angle = look_direction - angle_mic_pair
    return relative_angle

def compute_af(theta, frequencies, delta_m, theta_m, c):
    """
    Calcule le vecteur de phase v_m_theta pour une paire de microphones donnée.

    :param theta: direction de regard θ en radians
    :param frequencies: les fréquences des spectrogrammes
    :param delta_m: la distance entre la paire de microphones
    :param theta_m: l'angle relatif entre la direction de regard et la paire de microphones
    :param c: la vitesse du son en m/s
    :return: vecteur de phase v_m_theta
    """
    v_m_theta = (2 * np.pi * frequencies * delta_m * np.cos(theta_m - theta) / c).unsqueeze(1)
    print(f"Computed AF for θ={theta} with shape {v_m_theta.shape}")
    return v_m_theta




def compute_af(theta, frequencies, delta_m, theta_m, c):
    """
    Calculate angle feature (AF) for given parameters.
    """
    v_m_theta = 2 * np.pi * frequencies * delta_m * np.cos(theta_m - theta) / c
    return v_m_theta

def compute_metrics(waveforms, sample_rate,n_fft=2048,hop_length=1024,theta_values =  [0,  np.pi/2, np.pi,  3*np.pi/2], mic_pairs=[(0, 4), (1, 5), (2, 6), (3, 7)]): 
    c = 343  # Speed of sound in m/s    
    print("Starting computation of metrics...")

    print("Calculating spectrograms...")
    transform = Spectrogram(n_fft, hop_length, power=None)
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

    # Calculate IPD for provided pairs of microphones
    ipd_pairs = {}
    print("Calculating IPD for provided pairs of microphones...")
    for (i, j) in mic_pairs:
        print(f"Calculating IPD for Microphones {i + 1} and {j + 1}")
        ipd_pairs[(i, j)] = torch.angle(specs[i]) - torch.angle(specs[j])
        print(f"IPD shape for microphones {i + 1} and {j + 1}: {ipd_pairs[(i, j)].shape}")

    # Define microphone positions (assuming a circular array)
    mic_positions = [(0.1 * np.cos(i * np.pi / 4), 0.1 * np.sin(i * np.pi / 4)) for i in range(8)]
    
    # Calculate relative distances and angles for provided pairs of microphones
    mic_pairs_info = {}
    for (i, j) in mic_pairs:
        delta_m = np.sqrt((mic_positions[i][0] - mic_positions[j][0])**2 + (mic_positions[i][1] - mic_positions[j][1])**2)
        theta_m = np.arctan2(mic_positions[j][1] - mic_positions[i][1], mic_positions[j][0] - mic_positions[i][0])
        mic_pairs_info[(i, j)] = (delta_m, theta_m)
    
    frequencies = torch.linspace(0, sample_rate // 2, specs[0].size(-2))
    af_dict = {}
    theta_values = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi]  # Example theta values
    
    print("Calculating AF for provided pairs of microphones...")
    for theta in theta_values:
        af_sum = torch.zeros((frequencies.size(0), specs[0].size(-1))).clone()
        for (i, j), ipd in ipd_pairs.items():
            print(f"Calculating AF contribution for Microphones {i + 1} and {j + 1} with θ={theta} radians")
            delta_m, theta_m = mic_pairs_info[(i, j)]
            v_m_theta = compute_af(theta, frequencies, delta_m, theta_m, c).reshape(-1, 1)
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

def plot_af_pairs(af_dict, sample_rate,theta_values ,duration=10):
    subsample_length = duration * sample_rate // 1024  # Assuming hop_length is 1024
    times = np.linspace(0, duration, subsample_length)
    frequencies = torch.linspace(0, sample_rate // 2, list(af_dict.values())[0].size(0))
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



def plot_energy_distribution(af_dict, n_seconds, ax):
    total_duration = 2424  # Durée totale du spectrogramme en secondes
    logging.info(f'Total duration: {total_duration} seconds')

    # Initialisation d'un dictionnaire pour stocker les énergies
    energy_dict = {}

    # Calculer l'énergie pour chaque direction θ en sommant les normes au carré des valeurs d'AF sur toutes les fréquences
    for theta, af in af_dict.items():
        energy = (af ** 2).sum(axis=0)  # Somme de la norme au carré sur les fréquences
        energy_dict[theta] = energy
        logging.debug(f'Energy calculated for theta={theta}: {energy}')
        logging.debug(f'Energy shape for theta={theta}: {energy.shape}')

    # Normalisation des énergies pour obtenir une distribution entre 0 et 1
    total_energy = sum(energy_dict.values())
    logging.info(f'Total energy: {total_energy}')
    
    normalized_energy_dict = {theta: energy / total_energy for theta, energy in energy_dict.items()}

    # Déterminer le nombre total d'échantillons dans `af`
    af_length = af_dict[list(af_dict.keys())[0]].shape[1]
    logging.info(f'AF length (number of samples): {af_length}')
    
    # Calculer le nombre d'échantillons correspondant à n_seconds
    n_samples = int((n_seconds / total_duration) * af_length)
    logging.info(f'Number of samples for {n_seconds} seconds: {n_samples}')

    # Tracer la distribution d'énergie
    for theta, energy in normalized_energy_dict.items():
        times = np.linspace(0, total_duration, len(energy))
        logging.debug(f'Times for theta={theta}: {times}')
        
        # Filtrer les énergies pour ne prendre que celles avant n_seconds
        energy_filtered = energy[:n_samples]
        times_filtered = times[:n_samples]
        logging.debug(f'Filtered energy for theta={theta}: {energy_filtered}')
        
        ax.plot(times_filtered, energy_filtered, label=f'θ={theta} rad')

    ax.set_title('Distribution d\'énergie normalisée en fonction du temps pour différentes directions θ')
    ax.set_xlabel('Temps (secondes)')
    ax.set_ylabel('Énergie normalisée')
    ax.legend()
    ax.grid(True)
