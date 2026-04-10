import os
import csv
import json
import torch

from relik import Relik
from relik.inference.data.objects import RelikOutput
from kilt.knowledge_source import KnowledgeSource

# Inizializza il modello Relik
relik = Relik.from_pretrained(
    "sapienzanlp/relik-entity-linking-base",
    device="cuda" if torch.cuda.is_available() else "cpu",
    precision="fp16" if torch.cuda.is_available() else "fp32",
    skip_metadata=True
)

directory = "moving/"
output_directory = "movingjson/"
os.makedirs(output_directory, exist_ok=True)

# Inizializza KnowledgeSource una sola volta
ks = KnowledgeSource()
ks.get_num_pages()

# Cicla tutti i file nella cartella
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        output_path = os.path.join(output_directory, filename + ".json")

        new_data = []

        with open(filepath, newline="", encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)

            # Salta l'header
            next(csvreader, None)

            for row in csvreader:
                if len(row) > 1:
                    sen = row[1]

                    # Applica Relik alla frase
                    relik_out: RelikOutput = relik(sen)
                    print(relik_out.spans)

                    new_item = {
                        "sentence": sen,
                        "entities": []
                    }

                    
                    seen_entities = set()

                    for entityRelik in relik_out.spans:
                        titleEntity = entityRelik.label

                        try:
                            page = ks.get_page_by_title(titleEntity)

                            if page is not None:
                                wikipedia_title = page.get("wikipedia_title")
                                wikidata_info = page.get("wikidata_info", {}).get("wikidata_id")
                            else:
                                wikipedia_title = titleEntity
                                wikidata_info = None

                        except Exception as e:
                            print(f"Errore su entità '{titleEntity}': {e}")
                            wikipedia_title = titleEntity
                            wikidata_info = None

                        
                        key = (entityRelik.text, wikidata_info)

                        if key not in seen_entities:
                            seen_entities.add(key)

                            new_item["entities"].append({
                                "originalKey": entityRelik.text,
                                "original_value": wikipedia_title,
                                "Wikidata_ID": wikidata_info
                            })

                    # Append della frase, anche se entities è vuoto
                    new_data.append(new_item)

        # Salva il JSON UNA SOLA VOLTA alla fine
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)

        print(f"Creato: {output_path}")