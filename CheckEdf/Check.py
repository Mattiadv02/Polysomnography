###################################################
# CSV COMBINATO DI EDF E EVENTI, SPALMATI IN 8 ORE
import csv
import re
import pyedflib
import os
from datetime import datetime, timedelta
import numpy as np

def is_valid_edf(file_path):
    """Check if the EDF file is valid and can be opened."""
    try:
        with pyedflib.EdfReader(file_path) as f:
            return True
    except Exception as e:
        print(f"Errore durante la validazione del file EDF: {file_path}\n{e}")
        return False

def resample_signal(data, original_frequency, target_frequency):
    if original_frequency == target_frequency:
        return data
    elif original_frequency > target_frequency:
        factor = int(round(original_frequency / target_frequency))
        return data[::factor]
    else:
        factor = int(round(target_frequency / original_frequency))
        return np.repeat(data, factor)

def parse_edf_file(file_path, target_frequency):
    try:
        f = pyedflib.EdfReader(file_path)
    except OSError as e:
        print(f"Errore durante l'apertura del file EDF: {file_path}\n{e}")
        return None, None, None

    n_signals = f.signals_in_file
    edf_data = {}
    for i in range(n_signals):
        signal_label = f.signal_label(i).strip()
        sample_frequency = f.getSampleFrequency(i)
        signal_data = f.readSignal(i)

        resampled_data = resample_signal(signal_data, sample_frequency, target_frequency)
        edf_data[signal_label] = {
            'sample_frequency': target_frequency,
            'signal_data': resampled_data,
            'data_length': len(resampled_data)
        }

    f.close()
    n_row = max(len(data['signal_data']) for data in edf_data.values())
    return edf_data, n_row, target_frequency

def parse_event_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    ora_inizio = None
    blocchi = []
    blocco_corrente = []

    for i, line in enumerate(lines):
        if line.strip() == "[Ora di inizio]":
            ora_inizio = lines[i + 1].strip()

        if line.startswith("[Canale_"):
            if blocco_corrente:
                blocchi.append("".join(blocco_corrente))
                blocco_corrente = []
        blocco_corrente.append(line)

    if blocco_corrente:
        blocchi.append("".join(blocco_corrente))

    blocchi.pop(0)

    dati_eventi = {}

    for idx, blocco in enumerate(blocchi):
        eventi = []
        righe_blocco = blocco.splitlines()
        for i, riga in enumerate(righe_blocco):
            if "[Eventi Canale_" in riga:
                nome_blocco = riga.split("]")[0].replace("[Eventi ", "").strip()
                eventi_righe = righe_blocco[i + 2:]
                for evento_riga in eventi_righe:
                    match = re.search(r'Inizio:\s*(.*?);\s*Durata \[ms\]:\s*(\d+);\s*Evento:\s*(.*?);', evento_riga)
                    if match:
                        evento_info = {
                            "Inizio": match.group(1),
                            "Durata [ms]": int(match.group(2)),
                            "Evento": match.group(3)
                        }
                        eventi.append(evento_info)
                break
        if nome_blocco:
            dati_eventi[nome_blocco] = eventi
        else:
            dati_eventi[f"Blocco_{idx + 1}"] = eventi

    return dati_eventi, ora_inizio

def convert_event_in_df(eventi_dict, n_row, ora_inizio, frequenza_campionamento):
    ora_inizio_dt = datetime.strptime(ora_inizio, '%H:%M:%S')
    vettori_eventi = {}
    for nome_blocco, eventi in eventi_dict.items():
        vettore = [None] * n_row

        for evento in eventi:
            evento_inizio_dt = datetime.strptime(evento["Inizio"], '%H:%M:%S')
            if evento_inizio_dt < ora_inizio_dt:
                evento_inizio_dt += timedelta(days=1)

            offset_secondi = int((evento_inizio_dt - ora_inizio_dt).total_seconds())
            offset_campionamento = offset_secondi * frequenza_campionamento
            durata_campionamento = (evento["Durata [ms]"] * frequenza_campionamento) // 1000

            for i in range(offset_campionamento, offset_campionamento + durata_campionamento):
                if i < len(vettore):
                    vettore[i] = evento["Evento"]

        vettori_eventi[nome_blocco] = vettore

    return vettori_eventi

def create_combined_csv(edf_data, events, corrected_events, output_csv_path):
    max_duration = max(len(events.get(channel, [])) for channel in events) if events else 0
    if corrected_events:
        max_duration = max(max_duration, max(len(corrected_events.get(channel, [])) for channel in corrected_events))

    channel_buffers = {channel: [] for channel in events}
    corrected_buffers = {channel: [] for channel in corrected_events}

    for channel, data in events.items():
        buffer = data + ["NULL"] * (max_duration - len(data))
        channel_buffers[channel] = buffer

    for channel, data in corrected_events.items():
        buffer = data + ["NULL"] * (max_duration - len(data))
        corrected_buffers[channel] = buffer

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = list(edf_data.keys()) + list(events.keys()) + [f"{col}_corretto" for col in corrected_events.keys()]
        writer.writerow(header)

        for i in range(max_duration):
            row = []
            for signal in edf_data.keys():
                row.append(edf_data[signal]['signal_data'][i] if i < len(edf_data[signal]['signal_data']) else "NULL")
            for channel in channel_buffers.keys():
                row.append(channel_buffers[channel][i])
            for channel in corrected_buffers.keys():
                row.append(corrected_buffers[channel][i])
            writer.writerow(row)

# Directory processing logic
base_dir = "C:/Users/Utente/Desktop/Polysomnography/CheckEdf/Polisonnografie Anonime"
output_dir = "C:/Users/Utente/Desktop/Polysomnography/Eventi_Spalmati"
target_frequency = 10

for file_name in os.listdir(base_dir):
    if file_name.endswith("_edf.edf"):
        base_name = file_name.split("_edf.edf")[0]
        edf_path = os.path.join(base_dir, file_name)
        events_path = os.path.join(base_dir, f"{base_name}_events.txt")
        corrected_events_path = os.path.join(base_dir, f"{base_name}_event_corretto.txt")
        output_csv_path = os.path.join(output_dir, f"{base_name}_combined.csv")

        if not is_valid_edf(edf_path):
            with open(os.path.join(output_dir, "error_log.txt"), 'a') as log_file:
                log_file.write(f"File EDF non valido: {edf_path}\n")
            continue

        edf_data, n_row, frequenza_campionamento = parse_edf_file(edf_path, target_frequency)
        if edf_data is None:
            continue

        events = {}
        corrected_events = {}
        if os.path.exists(events_path):
            eventi_dict, ora_inizio = parse_event_file(events_path)
            events = convert_event_in_df(eventi_dict, n_row, ora_inizio, target_frequency)

        if os.path.exists(corrected_events_path):
            corrected_eventi_dict, corrected_ora_inizio = parse_event_file(corrected_events_path)
            corrected_events = convert_event_in_df(corrected_eventi_dict, n_row, corrected_ora_inizio, target_frequency)

        create_combined_csv(edf_data, events, corrected_events, output_csv_path)


################################################################
# #SPETTROGRAMMI in immagini

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def genera_spettrogrammi_da_csv(file_path, fs=1.0, durata=28800):
    """
    Genera e salva uno spettrogramma per ogni colonna numerica del file CSV e
    stampa il valore minimo, massimo e medio del segnale per ciascuna colonna.

    Parametri:
    - file_path: percorso del file CSV di input.
    - fs: frequenza di campionamento (Hz). Impostare in base alla frequenza di acquisizione dei dati.
    - durata: durata del segnale in secondi (default: 28800 per 8 ore).
    """
    # Carica il file CSV
    df = pd.read_csv(file_path)

    # Itera sulle colonne del DataFrame
    for colonna in df.columns:
        try:
            # Converte la colonna in numerico, ignorando valori non numerici
            segnale = pd.to_numeric(df[colonna], errors='coerce').dropna().values  # Rimuovi NaN

            # Salta le colonne non numeriche o con pochi dati validi
            if len(segnale) == 0:
                print(f"Colonna {colonna} non contiene dati numerici validi. Skipping.")
                continue

            # Calcola i valori min, max e media
            min_val = np.min(segnale)
            max_val = np.max(segnale)
            mean_val = np.mean(segnale)

            print(f"Colonna {colonna} - Min: {min_val}, Max: {max_val}, Media: {mean_val}")

            # Calcola lo spettrogramma con una durata complessiva di 28800 secondi
            f, t, Sxx = spectrogram(segnale, fs)

            # Normalizza i valori di tempo t per riflettere correttamente i secondi
            t = t * (durata / t[-1])  # Adatta il tempo per coprire 28800 secondi

            # Gestisce il logaritmo dei valori zero aggiungendo un valore piccolo
            Sxx_log = 10 * np.log10(Sxx + 1e-10)

            # Trova i 10 valori più alti nel canale specificato (se è "b'Resp nasal'")
            if colonna == "b'Resp nasal'":
                # Appiattisce l'array Sxx_log per trovare i picchi indipendentemente da t e f
                flat_indices = np.argpartition(Sxx_log.flatten(), -10)[-10:]
                peak_values = Sxx_log.flatten()[flat_indices]

                # Ottiene le coordinate dei picchi (frequenza, tempo)
                peak_coords = np.unravel_index(flat_indices, Sxx_log.shape)
                peak_times = t[peak_coords[1]]
                peak_freqs = f[peak_coords[0]]

            # Crea il plot dello spettrogramma
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
            plt.colorbar(label='Intensità (dB)')
            plt.ylabel('Frequenza [Hz]')
            plt.xlabel('Tempo [sec]')
            plt.title(f'Spettrogramma per la colonna: {colonna}')


            # Se è il canale "b'Resp nasal'", aggiunge i picchi al grafico
            if colonna == "b'Resp nasal'":
                # Usa scatter per i picchi con un marker grande, blu e con bordo bianco per massima visibilità
                plt.scatter(peak_times, peak_freqs, color='cyan', edgecolor='black', s=500, zorder=5, label='Top 10 picchi', marker='o')

                # Aggiungi etichette numeriche accanto a ciascun picco
                for i, (pt, pf) in enumerate(zip(peak_times, peak_freqs)):
                    plt.text(pt, pf, f'{i + 1}', color='black', fontsize=12, ha='center', va='center', fontweight='bold')

                plt.legend()

            # Salva il grafico come immagine
            plt.savefig(f'spettrogramma_{colonna}.png', dpi=300)
            plt.close()  # Chiudi la figura per evitare sovrapposizioni

        except Exception as e:
            print(f"Errore nella generazione dello spettrogramma per la colonna {colonna}: {e}")

    print("Spettrogrammi generati e valori statistici calcolati.")


# Esempio di utilizzo
genera_spettrogrammi_da_csv("C:/Users/Utente/Desktop/Polysomnography/Eventi_Spalmati/output_combined1.csv", fs=100)

################################################################
#PLOT_SPETTRO COMBO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def plotta_spettrogramma_con_segnale(file_path, nome_canale, fs=1.0):
    """
    Carica un file CSV e genera uno spettrogramma del canale specificato (colonna),
    sovrapponendo il plot del segnale con i 10 valori massimi evidenziati.

    Parametri:
    - file_path: Percorso del file CSV di input.
    - nome_canale: Nome della colonna da plottare.
    - fs: Frequenza di campionamento (Hz) del segnale. Default è 1 Hz.
    """
    # Carica il file CSV
    df = pd.read_csv(file_path)

    # Controlla se il canale esiste nel file CSV
    if nome_canale not in df.columns:
        print(f"Colonna '{nome_canale}' non trovata nel file.")
        return

    # Estrai i dati del canale specificato
    segnale = pd.to_numeric(df[nome_canale], errors='coerce').dropna().values  # Rimuovi NaN

    # Crea un asse temporale (in secondi) basato sulla frequenza di campionamento
    tempo = np.arange(len(segnale)) / fs

    # Calcola lo spettrogramma
    f, t, Sxx = spectrogram(segnale, fs)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)  # Scala in dB con gestione valori zero

    # Adatta i valori di tempo `t` per coprire la durata totale del segnale
    t = t * (tempo[-1] / t[-1])

    # Inverti l'asse delle frequenze per il corretto allineamento con il plot
    f = f[::-1]
    Sxx_log = Sxx_log[::-1, :]

    # Trova i 10 valori massimi nel segnale e i loro indici
    max_indices = np.argpartition(segnale, -10)[-10:]  # Indici dei 10 valori massimi
    max_indices = max_indices[np.argsort(segnale[max_indices])]  # Ordina gli indici per valore crescente

    # Crea il plot con spettrogramma e segnale sovrapposti
    plt.figure(figsize=(12, 6))

    # Plot dello spettrogramma con trasparenza
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis', alpha=0.5)
    plt.colorbar(label='Intensità (dB)')
    plt.ylabel('Frequenza [Hz]')
    plt.xlabel('Tempo [sec]')
    plt.title(f"Spettrogramma e segnale del Canale: {nome_canale}")

    # Plot del segnale sovrapposto
    plt.plot(tempo, segnale, label=nome_canale, color='red', alpha=0.4)  # Rendiamo il segnale trasparente

    # Evidenzia i 10 valori massimi con i pallini rossi anche nello spettrogramma
    for idx in max_indices:
        peak_time = tempo[idx]
        peak_value = segnale[idx]

        # Individua il tempo del picco nel segnale e visualizzalo sullo spettrogramma
        plt.plot(peak_time, f[np.argmax(Sxx[:, np.argmin(np.abs(t - peak_time))])],
                 'ro')  # Pallino rosso sullo spettrogramma
        plt.text(peak_time, f[np.argmax(Sxx[:, np.argmin(np.abs(t - peak_time))])], f'{peak_value:.2f}', color='yellow',
                 fontsize=10,
                 ha='center', va='bottom')  # Etichetta con valore sullo spettrogramma

    plt.legend()
    plt.grid()
    plt.show()


# Esempio di utilizzo
plotta_spettrogramma_con_segnale(
    "C:/Users/Utente/Desktop/Polysomnography/Eventi_Spalmati/output_combined1.csv",
    "b'Resp nasal      '",
    fs=100
)

################################################################
import csv
from datetime import datetime, timedelta
import statistics


def analizza_eventi(file_path, file_csv_path):
    conteggi_eventi = {
        "Flusso": {},
        "SpO2": {}
    }
    canale_attivo = None

    statistiche = {
        "Flusso": {
            "tempo_totale_apnea_ipopnea_ms": 0,
            "apnea_max_durata": 0,
            "ipopnea_max_durata": 0,
            "apnea_max_inizio": "",
            "ipopnea_max_inizio": "",
            "conteggio_apnee": 0,
            "conteggio_ipopnee": 0,
        },
        "SpO2": {
            "tempo_totale_desaturazione_ms": 0,
            "desaturazione_max_durata": 0,
            "desaturazione_max_inizio": "",
            "durata_media_desaturazione_sec": 0,
            "conteggio_desaturazioni": 0
        }
    }

    orario_inizio_sonno = None
    orario_fine_sonno = None

    # Analisi del file di eventi
    with open(file_path, 'r') as file:
        for linea in file:
            linea = linea.strip()

            if linea == "[Eventi Canale_00 Flusso]":
                canale_attivo = "Flusso"
                continue
            elif linea == "[Eventi Canale_06 SpO2]":
                canale_attivo = "SpO2"
                continue
            elif canale_attivo and (linea.startswith("[") and linea.endswith("]")):
                canale_attivo = None
                continue

            if canale_attivo and "Evento:" in linea:
                parts = linea.split(";")
                orario_inizio = parts[0].split("Inizio:")[1].strip()

                # Safe conversion for duration
                try:
                    durata_ms = int(parts[1].split("Durata [ms]:")[1].strip())
                except (ValueError, IndexError):
                    print(f"Errore nella conversione della durata. Riga: {linea}")
                    durata_ms = 0  # Assign a default value if conversion fails

                evento_nome = parts[2].split("Evento:")[1].strip()

                if orario_inizio_sonno is None:
                    orario_inizio_sonno = orario_inizio
                orario_fine_sonno = orario_inizio

                if evento_nome not in conteggi_eventi[canale_attivo]:
                    conteggi_eventi[canale_attivo][evento_nome] = 1
                else:
                    conteggi_eventi[canale_attivo][evento_nome] += 1

                if canale_attivo == "Flusso" and ("Apnea" in evento_nome or "Ipopnea" in evento_nome):
                    statistiche["Flusso"]["tempo_totale_apnea_ipopnea_ms"] += durata_ms
                    if "Apnea" in evento_nome:
                        statistiche["Flusso"]["conteggio_apnee"] += 1
                        if durata_ms > statistiche["Flusso"]["apnea_max_durata"]:
                            statistiche["Flusso"]["apnea_max_durata"] = durata_ms
                            statistiche["Flusso"]["apnea_max_inizio"] = orario_inizio
                    elif "Ipopnea" in evento_nome:
                        statistiche["Flusso"]["conteggio_ipopnee"] += 1
                        if durata_ms > statistiche["Flusso"]["ipopnea_max_durata"]:
                            statistiche["Flusso"]["ipopnea_max_durata"] = durata_ms
                            statistiche["Flusso"]["ipopnea_max_inizio"] = orario_inizio

                elif canale_attivo == "SpO2" and "Desaturazione" in evento_nome:
                    statistiche["SpO2"]["tempo_totale_desaturazione_ms"] += durata_ms
                    statistiche["SpO2"]["conteggio_desaturazioni"] += 1
                    if durata_ms > statistiche["SpO2"]["desaturazione_max_durata"]:
                        statistiche["SpO2"]["desaturazione_max_durata"] = durata_ms
                        statistiche["SpO2"]["desaturazione_max_inizio"] = orario_inizio

    def converti_ms_a_hms(millis):
        """Converte millisecondi in una stringa formato HH:MM:SS"""
        return str(timedelta(milliseconds=millis))

    statistiche["Flusso"]["tempo_totale_apnea_ipopnea_hms"] = converti_ms_a_hms(
        statistiche["Flusso"]["tempo_totale_apnea_ipopnea_ms"])
    statistiche["SpO2"]["tempo_totale_desaturazione_hms"] = converti_ms_a_hms(
        statistiche["SpO2"]["tempo_totale_desaturazione_ms"])

    if statistiche["SpO2"]["conteggio_desaturazioni"] > 0:
        statistiche["SpO2"]["durata_media_desaturazione_sec"] = (
                statistiche["SpO2"]["tempo_totale_desaturazione_ms"] / statistiche["SpO2"][
            "conteggio_desaturazioni"] / 1000
        )

    # Analisi del file CSV per statistiche aggiuntive
    valori_csv = []
    conteggi_eventi_csv = {}

    with open(file_csv_path, 'r') as file_csv:
        reader = csv.reader(file_csv)
        errori_conversione = []  # Per raccogliere gli errori di conversione

        for riga_num, linea in enumerate(reader, start=1):
            if not linea or linea[0].startswith('#'):
                continue  # Salta le righe vuote o commentate

            evento_nome = linea[2]  # Supponiamo che la terza colonna indichi il nome dell'evento

            # Safe conversion for value
            try:
                valore = float(linea[1])  # Supponiamo che la seconda colonna indichi il valore
                valori_csv.append(valore)
            except (ValueError, IndexError):
                errori_conversione.append(f"Errore nella conversione del valore dalla riga {riga_num}: {linea}")

            # Contare le occorrenze degli eventi
            if evento_nome not in conteggi_eventi_csv:
                conteggi_eventi_csv[evento_nome] = 1
            else:
                conteggi_eventi_csv[evento_nome] += 1

    # Calcolo delle statistiche
    statistiche_csv = {
        "max_valore": max(valori_csv) if valori_csv else 0,
        "min_valore": min(valori_csv) if valori_csv else 0,
        "media_valore": statistics.mean(valori_csv) if valori_csv else 0,
        "deviazione_standard": statistics.stdev(valori_csv) if len(valori_csv) > 1 else 0,
    }

    # Stampa dei risultati
    with open("risultati_eventi_statistiche.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Canale", "Tipo", "Descrizione", "Valore"])

        for canale, eventi in conteggi_eventi.items():
            for evento, conteggio in eventi.items():
                writer.writerow([canale, "Conteggio Evento", evento, conteggio])

        for canale, stats in statistiche.items():
            for chiave, valore in stats.items():
                writer.writerow([canale, "Statistica", chiave, valore])

        # Scrittura delle statistiche aggiuntive
        for chiave, valore in statistiche_csv.items():
            writer.writerow(["CSV", "Statistica", chiave, valore])

        # Stampa degli errori di conversione
        if errori_conversione:
            writer.writerow(["Errore", "Descrizione"])
            for errore in errori_conversione:
                writer.writerow([errore])

    return conteggi_eventi, statistiche, statistiche_csv


# Esempio di utilizzo
file_path = 'C:/Users/Utente/Desktop/Polysomnography/CheckEdf/Polisonnografie Anonime/1_events.txt'
file_csv_path = "C:/Users/Utente/Desktop/Polysomnography/Eventi_Spalmati/output_combined1.csv"

conteggi, statistiche, statistiche_csv = analizza_eventi(file_path, file_csv_path)

# Stampa dei risultati
for canale in ["Flusso", "SpO2"]:
    print(f"\nConteggi degli eventi nel canale '[Eventi Canale_{0 if canale == 'Flusso' else 6} {canale}]':")
    for evento, conteggio in conteggi[canale].items():
        print(f"{evento}: {conteggio} occorrenze")

    print(f"\nStatistiche per il canale '{canale}':")
    for key, value in statistiche[canale].items():
        print(f"{key}: {value}")

print("\nStatistiche dal file CSV:")
for key, value in statistiche_csv.items():
    print(f"{key}: {value}")

#############################################################################
#interruzioni ogni 10 secondi
import pyedflib
import pandas as pd
import numpy as np

# Carica il file EDF
file_path = 'C:/Users/Utente/Desktop/Polysomnography/CheckEdf/Polisonnografie Anonime/1_edf.edf'  # Specifica qui il percorso del file EDF
edf = pyedflib.EdfReader(file_path)

# Trova il canale "Resp nasal"
channel_name = "Resp nasal"
channel_index = edf.getSignalLabels().index(channel_name)
colonna = edf.readSignal(channel_index)

# Chiude il file EDF dopo aver letto i dati
edf.close()

# Frequenza di campionamento (10 campioni al secondo)
sampling_rate = 10

# Numero di campioni da scartare (5 minuti)
start_offset = 5 * 60 * sampling_rate  # 5 minuti = 300 secondi

# Scarta i primi 5 minuti
colonna_troncata = colonna[start_offset:]

# Trova e stampa il valore massimo e minimo della colonna troncata
valore_massimo = colonna_troncata.max()
valore_minimo = colonna_troncata.min()
print("Valore massimo (troncato):", valore_massimo)
print("Valore minimo (troncato):", valore_minimo)

# Normalizza i valori della colonna troncata
colonna_normalizzata = (colonna_troncata - valore_minimo) / (valore_massimo - valore_minimo)

# Calcola la media dei valori normalizzati
media_normalizzata = colonna_normalizzata.mean()
print("Media dei valori normalizzati:", media_normalizzata)

# Crea un DataFrame per salvare i risultati e aggiungere le colonne "APNEA" e "IPOPNEA"
df_result = pd.DataFrame({"Resp nasal normalizzato": colonna_normalizzata})
df_result["APNEA"] = False
df_result["IPOPNEA"] = False

# Parametri per l'analisi
window_size = 1 * sampling_rate  # Finestra di 10 secondi (100 campioni)
soglia_apnea = 0.1  # Soglia per apnea
soglia_ipopnea = 0.3  # Soglia per ipopnea

# Scansione dei dati troncati
for start in range(0, len(colonna_normalizzata) - window_size + 1, sampling_rate):
    # Calcola la media della finestra corrente
    end = start + window_size
    finestra = colonna_normalizzata[start:end]
    media_temp = finestra.mean()

    # Condizione per apnea: media_temp fuori dalla soglia +- soglia_apnea
    if not (media_normalizzata - soglia_apnea <= media_temp <= media_normalizzata + soglia_apnea):
        df_result.loc[start:end - 1, "APNEA"] = True

    # Condizione per ipopnea: media_temp fuori dalla soglia +- soglia_ipopnea
    if not (media_normalizzata - soglia_ipopnea <= media_temp <= media_normalizzata + soglia_ipopnea):
        df_result.loc[start:end - 1, "IPOPNEA"] = True

# Salva il DataFrame aggiornato con le colonne APNEA e IPOPNEA in un nuovo CSV
output_path = 'C:/Users/Utente/Desktop/Polysomnography/Eventi_Spalmati/valori_normalizzati_con_eventi.csv'
df_result.to_csv(output_path, index=False)

print(f"Analisi completata. I risultati sono stati salvati in {output_path}")


#########################################################################
# PLOT RESP_NASAL DEVIAZIONE STANDARD A CAMPANA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Caricamento del file CSV
file_path = 'C:/Users/Utente/Desktop/Polysomnography/Eventi_Spalmati/1_combined.csv'  # Modifica con il percorso del tuo file

data = pd.read_csv(file_path)

# Verifica che la colonna sia presente
target_column = "b'Resp nasal'"
if target_column not in data.columns:
    raise ValueError(f"La colonna '{target_column}' non è presente nel file CSV.")

# Estrazione della colonna e calcolo della deviazione standard
resp_nasal_data = data[target_column][3000:].dropna()  # Rimuove valori NaN
std_dev = np.std(resp_nasal_data)

# Calcolo dei valori per la curva a campana
mean = np.mean(resp_nasal_data)
x = np.linspace(min(resp_nasal_data), max(resp_nasal_data), 1000)
p = norm.pdf(x, mean, std_dev)

# Calcolo range
soglia = 0.1
soglia_ipop = 0.03
soglia_app = 0.005
range_max = mean * (1 + soglia)
range_max_ipo = mean * (1 + soglia_ipop)
range_max_app = mean * (1 + soglia_app)
range_min = mean * (1 - soglia)
range_min_app = mean * (1 - soglia_app)
range_min_ipo = mean * (1 - soglia_ipop)

# Creazione del grafico
plt.figure(figsize=(10, 6))

# Istogramma
# count, bins, ignored = plt.hist(resp_nasal_data, bins=30, density=True, alpha=0.6, color='blue', label='Istogramma')

# Sovrapposizione della curva a campana
plt.plot(x, p, 'r-', linewidth=2, label='Curva a campana')

# Aggiunta delle rette verticali
plt.axvline(mean, color='green', linestyle='-', linewidth=1.5, label=f'Media: {mean:.2f}')
# plt.axvline(range_max, color='orange', linestyle='--', linewidth=1.5, label=f'Range max: {range_max:.2f}')
# plt.axvline(range_min, color='blue', linestyle='--', linewidth=1.5, label=f'Range min: {range_min:.2f}')
plt.axvline(range_min_ipo, color='blue', linestyle='--', linewidth=1.5, label=f'Range min IPO: {range_min_ipo:.2f}')
plt.axvline(range_max_ipo, color='orange', linestyle='--', linewidth=1.5, label=f'Range max IPO: {range_max_ipo:.2f}')
plt.axvline(range_min_app, color='blue', linestyle=':', linewidth=1.5, label=f'Range min APP: {range_min_app:.2f}')
plt.axvline(range_max_app, color='orange', linestyle=':', linewidth=1.5, label=f'Range max app: {range_max_app:.2f}')

# Aggiunta di titolo e etichette
plt.title("Distribuzione di b'Resp nasal'")
plt.xlabel("Valori")
plt.ylabel("Densità")
plt.legend()

# Mostrare il grafico
plt.grid()
plt.show()

print(f"Media della colonna {target_column}: {mean}")
print(f"Range max: {range_max}")
print(f"Range min: {range_min}")
print(f"Deviazione standard della colonna {target_column}: {std_dev}")
