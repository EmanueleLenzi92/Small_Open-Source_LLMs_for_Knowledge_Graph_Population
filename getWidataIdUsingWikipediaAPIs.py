import requests
import os
import json


import re

def sanitize_filename(filename):
    """
    Rimuove o sostituisce caratteri non validi per Windows
    """
    return re.sub(r'[(),<>:"/\\|?*]', '_', filename)


# Use Wikipedia APIs to find Wikidata QID from a Wikipedia title
def get_wikidata_entity_from_wikipedia_title(language, title):
    url = f"https://{language}.wikipedia.org/w/api.php"
    
    params = {
        "action": "query",
        "prop": "pageprops",
        "titles": title,
        "format": "json",
        "redirects": 1
    }

    headers = {
        # Wikipedia gradisce un user-agent identificabile
        "User-Agent": "WikiWikidataScript/1.0 (mailto:youremail@example.com)"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()  # solleva eccezione se status_code != 200
    except requests.RequestException as e:
        print(f"[ERRORE HTTP] Titolo '{title}': {e}")
        return None

    # Prova a decodificare come JSON
    try:
        data = response.json()
    except ValueError:
        print(f"[ERRORE JSON] Risposta non JSON per titolo '{title}'.")
        print("Primi 300 caratteri della risposta:")
        print(response.text[:300])
        return None

    pages = data.get("query", {}).get("pages", {})
    if pages:
        page = next(iter(pages.values()))
        if "pageprops" in page and "wikibase_item" in page["pageprops"]:
            return page["pageprops"]["wikibase_item"]
        else:
            return None
    else:
        return None


# Elaborate the JSON file (LLMs answers)

    
def process_json(input_json, language='en'):
    output_json = []
    
    for item in input_json:
        new_item = {"entities": []}

        # Controllo se l'item contiene "entities" o "keywords"
        if "entities" in item:
            source_list = item["entities"]
            wikipedia_field = "wikipedia_title"
            text_field = "entity_in_the_text"

        elif "keywords" in item:
            source_list = item["keywords"]
            wikipedia_field = "wikipedia_title"
            text_field = "keyword_in_the_text"
            
        elif "named_entities" in item:
            source_list = item["named_entities"]
            wikipedia_field = "wikipedia_title"
            text_field = "named_entity_in_the_text"

        else:
            # Nessun campo utile, aggiungo oggetto vuoto
            output_json.append(new_item)
            continue

        # Elaborazione di entities/keywords
        for entity in source_list:
            wikipedia_label = entity.get(wikipedia_field)
            keyword_in_the_text = entity.get(text_field)

            if wikipedia_label:
                wikidata_id = get_wikidata_entity_from_wikipedia_title(language, wikipedia_label)
            else:
                wikidata_id = None

            new_item["entities"].append({
                "originalKey": keyword_in_the_text,
                "original_value": wikipedia_label,
                "Wikidata_ID": wikidata_id
            })

        output_json.append(new_item)

    return output_json



# Read a JSON file
def read_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Save result in a JSON file
def save_json_to_file(output_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)


                
def process_all_json_files(input_folder, output_folder, language='en'):
    """
    Scorre ricorsivamente tutte le sottocartelle di input_folder,
    elabora ogni file .json trovato e salva il risultato in output_folder,
    mantenendo la stessa struttura di cartelle.

    Se il file di output esiste già, il file viene saltato
    (nessuna chiamata a Wikipedia).
    """

    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if not filename.endswith(".json"):
                continue

            input_file_path = os.path.join(root, filename)

            rel_path = os.path.relpath(root, input_folder)
            output_dir = os.path.join(output_folder, rel_path)
            os.makedirs(output_dir, exist_ok=True)

            # Prende solo la parte prima del primo "_"
            file_id = filename.split("_")[0]

            # Nome finale del file di output
            output_filename = f"{file_id}.json"
            output_file_path = os.path.join(output_dir, output_filename)


            # 🔴 CONTROLLO FONDAMENTALE
            if os.path.exists(output_file_path):
                print(f"[SKIP] Output già esistente: {output_file_path}")
                continue

            # Leggi il JSON di input
            try:
                input_json = read_json_from_file(input_file_path)
            except json.JSONDecodeError:
                print(f"[ERRORE] JSON non valido: {input_file_path}")
                continue

            # Elabora il JSON (qui partono le chiamate a Wikipedia)
            output_json = process_json(input_json, language)

            # Salva il risultato
            save_json_to_file(output_json, output_file_path)
            print(f"[OK] Elaborato: {input_file_path} -> {output_file_path}")




process_all_json_files("a", "output", language='en')
