from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

import json
from difflib import SequenceMatcher
import difflib
import subprocess

import time
import os
import re
import logging

import csv


# Configura il logger
logging.basicConfig(filename='error_log_moving.txt', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def estrai_json_da_stringa(stringa, percorso_file_json):
    # Usa una regex per estrarre il JSON dalla stringa
    match = re.search(r'\{.*\}', stringa, re.DOTALL)
    if match:
        json_string = match.group(0)
        
        
        # Rimuovi le virgole in eccesso prima di una chiusura di oggetto o array
        json_string_pulito = re.sub(r',\s*(\}|\])', r'\1', json_string)
        
        try:
            return json.loads(json_string_pulito)
        except json.JSONDecodeError as e:
            # Registra l'errore e la stringa che ha causato l'errore
            logging.error(f"Errore nel JSON: {percorso_file_json}")
            logging.error(f"Errore nel parsing del JSON: {e}")
            logging.error(f"Stringa problematica: {json_string_pulito}")
            logging.error(f"###")
            logging.error(f"###")
            logging.error(f"###")
            try:
                good_json_string = repair_json(json_string_pulito)
                logging.error(f"**********")
                logging.error(f"json riparato e inserito nel file")
                logging.error(f"**********")
                return json.loads(good_json_string)
            except:
                logging.error(f"$$$$$$$$$$")
                logging.error(f"json Nemmeno riparato")
                logging.error(f"$$$$$$$$$$")
                return None
    else:
        logging.error(f"Nessun JSON valido trovato nella stringa: {stringa}")
        return None

def aggiorna_file_json(nuovi_dati, percorso_file):
    # Se il file esiste, carica i dati esistenti
    if os.path.exists(percorso_file):
        with open(percorso_file, 'r', encoding='utf-8') as file:
            dati_esistenti = json.load(file)
    else:
        dati_esistenti = []

    # Aggiungi i nuovi dati ai dati esistenti
    dati_esistenti.append(nuovi_dati)

    # Salva il file aggiornato
    with open(percorso_file, 'w', encoding='utf-8') as file:
        json.dump(dati_esistenti, file, ensure_ascii=False, indent=4)        
        
directory= "dataset/"

listllms= ["phi4:14b-q8_0", "deepseek-r1:14b-qwen-distill-q8_0", "gemma2:9b-instruct-q8_0", "llama3:8b-instruct-q8_0", "mistral:7b-instruct-q8_0", "llama3.2:3b", "gemma2:2b", "gemma3:4b-it-q8_0", "gemma3:12b-it-q8_0"]
 

systemPrompt= """Retrieve the Named Entities in the text, considering only these types: Person, Location, GPE, Organization. For each of them, find the exact title corresponding to the Wikipedia page. The final result should be a json like this:

### Json
{
    "named_entities": [
        {
            "named_entity_in_the_text": "...",
            "wikipedia_title": "..."
        }
    ]
}

Answer only with the json
"""


for llmModel in listllms:
    # Cicla tutti i file nella cartella
    for filename in os.listdir(directory):
        # Controlla se il file ha estensione .csv
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            #print(f"Processing file: {filename}")
            
                
            # Apri il file CSV
            with open(filepath, newline='', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                #csvreader = csv.reader(csvfile, delimiter=';', quotechar='"')
                    
                # Salta la prima riga (i titoli)
                next(csvreader)
                    
                # Processa ogni riga
                for row in csvreader:
                    # Stampa il valore della seconda colonna (indice 1)
                    if len(row) > 1:
                        #print(row[1])
    
                        sen = row[1]
                        print(sen)
            
                        
                        llm = Ollama(
                            model=llmModel, 
                            system=systemPrompt, 
                            num_ctx=4096, 
                            temperature=0.01, 
                            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                        )
                        
                        events = llm(sen)
    
                        # Percorso del file JSON dove salvare i dati
                        os.makedirs("movingJson/"+llmModel, exist_ok=True)
                        percorso_file_json = 'movingJson/'+llmModel+'/'+filename+'.json'
                        
                        # Estrai il JSON dalla stringa
                        json_estratto = estrai_json_da_stringa(events, percorso_file_json)
                        
                        if json_estratto:
                            
                            # Aggiorna il file JSON con i nuovi dati
                            aggiorna_file_json(json_estratto, percorso_file_json)
                        else:
                            print("Nessun JSON trovato nella stringa.")

                            my_jsone = {
                                "named_entities": []
                                #"keywords": []
                            }
                            aggiorna_file_json(my_jsone, percorso_file_json)