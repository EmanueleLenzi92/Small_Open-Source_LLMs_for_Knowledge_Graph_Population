import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict

def jaccard_similarity(str1, str2):
    """
    Calculate the Jaccard infex between two string.
    """
    set1, set2 = set(str1.lower().split()), set(str2.lower().split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def extract_wikidata_id(link, nullTP =False):
    """
    Get Wikidata ID from the gold standard keywords Wikidata link
    """
    if nullTP == True:
        if not link:
            return None
        id_part = link.split("/")[-1]
        return None if id_part.lower() == "null" else id_part
    else:
        return link.split("/")[-1] if link else None

def load_json_files(folder_path):
    """
    Load all JSON files from a folder.
    """
    data = {}
    if not os.path.isdir(folder_path):
        return data
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                    data[filename] = json.load(file)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Errore nel caricamento del file {filename}: {e}")
    return data



def calculate_metrics(gold_data, predicted_data, jaccard_threshold, nullTP= False):
    """
    Calculate precision, recall e F1 score for keyword linking.
    """

    
# Metriche globali per l'keyword linking
    linking_true_positive_global = 0
    linking_false_positive_global = 0
    linking_false_negative_global = 0

    true_positives_per_file = {}
    false_positive_per_file = {}
    false_negative_per_file = {}

    for filename, gold_content in gold_data.items():
        predicted_content = predicted_data.get(filename.replace(".json", ".csv.json"))
        if not predicted_content:
            continue

        file_true_positive = []
        file_false_positive = []
        file_false_negative = []

        for gold_entities, pred_entities in zip(gold_content, predicted_content):
            gold_labels = [(entity["Wikipedia_label"], extract_wikidata_id(entity["Wikidata_ID"], nullTP)) for entity in gold_entities["entities"]]
            pred_entities_processed = [
                (entity["originalKey"], entity["Wikidata_ID"]) for entity in pred_entities["entities"]
            ]

            matched_gold = set()
            matched_pred = set()
            
            # TP 
            for i, (gold_label, gold_wikidata_id) in enumerate(gold_labels):
                for j, (pred_key, pred_wikidata_id) in enumerate(pred_entities_processed):
                
          
         
                    if jaccard_similarity(gold_label, pred_key) >= jaccard_threshold:
                        matched_gold.add(i)
                        matched_pred.add(j)
 


                       
                        if gold_wikidata_id == pred_wikidata_id:
                            linking_true_positive_global += 1
                            file_true_positive.append({
                                "gold_label": gold_label,
                                "pred_key": pred_key,
                                "gold_wikidata_id": gold_wikidata_id,
                                "pred_wikidata_id": pred_wikidata_id
                            })




                     
            #FP

            for pred_key, pred_wikidata_id in pred_entities_processed:
                isSimilar = False
                for gold_label, gold_wikidata_id in gold_labels:
                    if jaccard_similarity(gold_label, pred_key) >= jaccard_threshold:
                        isSimilar = True
                        # Se la stringa è simile ma l'ID non coincide → FP (linking errato)
                        if gold_wikidata_id != pred_wikidata_id:
                            linking_false_positive_global += 1
                            file_false_positive.append({
                                "gold_label": gold_label,
                                "pred_key": pred_key + " (id wrong)",
                                "gold_wikidata_id": gold_wikidata_id,
                                "pred_wikidata_id": pred_wikidata_id
                            })


                # Se nessuna stringa gold è simile → FP (entità inesistente)
                if not isSimilar:
                    linking_false_positive_global += 1
                    file_false_positive.append({
                        "gold_label": "",
                        "pred_key": pred_key + " (string not present in gold)",
                        "gold_wikidata_id": "",
                        "pred_wikidata_id": pred_wikidata_id
                    })
                        



                            

            #FN
            for i, (gold_label, gold_wikidata_id) in enumerate(gold_labels):
                best_match = None
                for j, (pred_key, pred_wikidata_id) in enumerate(pred_entities_processed):
                    if jaccard_similarity(gold_label, pred_key) >= jaccard_threshold:
                        best_match = pred_wikidata_id

                if best_match is None:
                    # mention gold non trovata
                    linking_false_negative_global += 1
                    file_false_negative.append({
                        "gold_label": gold_label +" (Mention Not found)",
                        "gold_wikidata_id": gold_wikidata_id
                    })
                elif best_match != gold_wikidata_id:
                    # mention trovata ma ID errato
                    linking_false_negative_global += 1
                    file_false_negative.append({
                        "gold_label": gold_label,
                        "gold_wikidata_id": gold_wikidata_id + " (Mention found with different id)"
                    })


        true_positives_per_file[filename] =  file_true_positive
        false_positive_per_file[filename] = file_false_positive
        false_negative_per_file[filename] =  file_false_negative
        
        

    # Metrichs for keyword linking
    linking_precision_global = linking_true_positive_global / (linking_true_positive_global + linking_false_positive_global) if (linking_true_positive_global + linking_false_positive_global) > 0 else 0
    linking_recall_global = linking_true_positive_global / (linking_true_positive_global + linking_false_negative_global) if (linking_true_positive_global + linking_false_negative_global) > 0 else 0
    linking_f1_score_global = (2 * linking_precision_global * linking_recall_global) / (linking_precision_global + linking_recall_global) if (linking_precision_global + linking_recall_global) > 0 else 0
                
                
    return (linking_precision_global, linking_recall_global, linking_f1_score_global, 
            true_positives_per_file, false_positive_per_file, false_negative_per_file)



def process_folders_recursively(
    gold_folder,
    root_folder,
    jaccard_threshold,
    stampaTP=False,
    stampaFP=False,
    stampaFN=False,
    model_name=None,
    file_name=None
):
    """
    Processa le sottocartelle di root_folder e calcola le metriche sui file JSON.
    
    Parametri opzionali:
    - model_name: se specificato, analizza solo quel modello (nome della sottocartella)
    - file_name: se specificato, mostra TP/FP/FN solo per quel file
    """

    # Carico una sola volta il gold
    gold_data = load_json_files(gold_folder)

    # Se voglio filtrare per file, tengo solo quel file nel gold
    if file_name is not None:
        if file_name in gold_data:
            gold_data = {file_name: gold_data[file_name]}
        else:
            print(f"File '{file_name}' non trovato nel gold standard.")
            return

    for dirpath, dirnames, filenames in os.walk(root_folder):
        json_files = [f for f in filenames if f.endswith(".json")]
        if not json_files:
            continue

        current_model = os.path.basename(dirpath)

        # Se è stato specificato un modello, salto tutti gli altri
        if model_name is not None and current_model != model_name:
            continue

        print(f"\nProcessando modello/cartella: {dirpath}")
        predicted_data = load_json_files(dirpath)

        if not gold_data or not predicted_data:
            print(f"Cartella {dirpath}: Nessun file JSON valido trovato.")
            continue

        # Se voglio filtrare per file, tengo solo il file corrispondente anche nelle predizioni
        if file_name is not None:
            pred_filename = file_name.replace(".json", ".csv.json")
            if pred_filename in predicted_data:
                predicted_data = {pred_filename: predicted_data[pred_filename]}
            else:
                print(f"Nel modello '{current_model}' il file predetto '{pred_filename}' non è stato trovato.")
                continue

        try:
            metrics = calculate_metrics(gold_data, predicted_data, jaccard_threshold)
            (
                linking_precision_global,
                linking_recall_global,
                linking_f1_score_global,
                true_positives_per_file,
                false_positives_per_file,
                false_negatives_per_file
            ) = metrics

            print(f"\nEntity Linking (Globale) - Precision: {linking_precision_global:.4f}")
            print(f"Entity Linking (Globale) - Recall: {linking_recall_global:.4f}")
            print(f"Entity Linking (Globale) - F1 Score: {linking_f1_score_global:.4f}")

            if stampaTP:
                print("\nTrue Positives per keyword Linking:")
                for filename, true_positives in true_positives_per_file.items():
                    print(f"\nFile: {filename}")
                    for tp in true_positives:
                        print(f"  Gold Label: {tp['gold_label']} | Predicted Key: {tp['pred_key']}")
                        print(f"    Gold Wikidata ID: {tp['gold_wikidata_id']} | Predicted Wikidata ID: {tp['pred_wikidata_id']}")

            if stampaFP:
                print("\nFalse Positives per keyword Linking:")
                for filename, false_positives in false_positives_per_file.items():
                    print(f"\nFile: {filename}")
                    for fp in false_positives:
                        print(f"  Gold Label: {fp['gold_label']} | Predicted Key: {fp['pred_key']}")
                        print(f"    Gold Wikidata ID: {fp['gold_wikidata_id']} | Predicted Wikidata ID: {fp['pred_wikidata_id']}")

            if stampaFN:
                print("\nFalse Negatives per keyword Linking:")
                for filename, false_negatives in false_negatives_per_file.items():
                    print(f"\nFile: {filename}")
                    for fn in false_negatives:
                        print(f"  Gold Label: {fn['gold_label']} | Gold id: {fn['gold_wikidata_id']}")

        except Exception as e:
            print(f"Errore durante il calcolo delle metriche per la cartella {dirpath}: {e}")


def sort_metrics(root_folder, gold_folder, metric_type, t):
    """
    Sort precision, recall e F1 score calculated for each folder, order by f1 score for the keyword extraction and keyword linking.
    """
    metrics_list = []

    for dirpath, _, filenames in os.walk(root_folder):
        json_files = [f for f in filenames if f.endswith(".json")]
        if not json_files:
            continue

        predicted_data = load_json_files(dirpath)
        gold_data = load_json_files(gold_folder)

        if not gold_data or not predicted_data:
            continue

        try:
            metrics = calculate_metrics(gold_data, predicted_data, t)
            (linking_precision_global, linking_recall_global, linking_f1_score_global, 
            _, _, _) = metrics

            # Seleziona precision, recall e F1 in base al tipo di metrica specificato
            if metric_type == "keyword extraction":
                precision = entity_precision
                recall = entity_recall
                f1_score = entity_f1_score
            elif metric_type == "keyword linking":
                precision = linking_precision_global
                recall = linking_recall_global
                f1_score = linking_f1_score_global
            else:
                raise ValueError(f"Tipo di metrica '{metric_type}' non riconosciuto. Usa 'entity extraction', 'global linking', o 'filtered linking'.")

            metrics_list.append((dirpath, precision, recall, f1_score))

        except Exception as e:
            print(f"Errore durante il calcolo delle metriche per la cartella {dirpath}: {e}")

    # Ordina le cartelle per F1 score in ordine decrescente
    metrics_list.sort(key=lambda x: x[3], reverse=True)  # x[3] è l'F1 score
    return metrics_list


def plot_all_metrics_trend(gold_folder, root_folder):
    """
    Mostra tre grafici (Precision, Recall, F1 score) per ogni modello,
    al variare della soglia di Jaccard.
    """
    # Carica i dati gold
    gold_data = load_json_files(gold_folder)
    thresholds = [round(x * 0.1, 1) for x in range(10, 0, -1)]

    # Prepara 3 subplot in un'unica figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=False)
    metrics = ["Precision", "Recall", "F1"]
    colors = ["tab:blue", "tab:orange", "tab:green"]  # opzionale, per coerenza visiva

    # Scorri tutte le sottocartelle del root_folder
    for model_name in sorted(os.listdir(root_folder)):
        model_path = os.path.join(root_folder, model_name)
        if not os.path.isdir(model_path):
            continue

        predicted_data = load_json_files(model_path)
        if not predicted_data:
            continue

        precision_values, recall_values, f1_values = [], [], []

        for t in thresholds:
            # Supponendo che calculate_metrics ritorni:
            # (TP, FP, FN, precision, recall, f1, altro)
            precision, recall, f1, _, _,_ = calculate_metrics(gold_data, predicted_data, t)

            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

        # Disegna le curve su ciascun subplot
        axes[0].plot(thresholds, precision_values, marker='o', label=model_name)
        axes[1].plot(thresholds, recall_values, marker='o', label=model_name)
        axes[2].plot(thresholds, f1_values, marker='o', label=model_name)

    # Personalizzazione dei subplot
    for ax, metric, color in zip(axes, metrics, colors):
        ax.set_xlabel("Jaccard threshold", fontsize=12)
        ax.set_ylabel(f"{metric} score", fontsize=12)
        ax.set_title(f"{metric} trend", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(thresholds)
        ax.legend(title="Models", fontsize=9)

    plt.tight_layout()
    plt.show()
    
def best_f1_per_model(gold_folder, root_folder, metric_type="keyword linking"):
    """
    Per ogni modello in root_folder:
    - calcola precision, recall, f1 per soglie da 1.0 a 0.1
    - seleziona la soglia con miglior f1
    - restituisce tabella ordinata per f1
    metric_type: 'keyword extraction' oppure 'keyword linking'
    """

    gold_data = load_json_files(gold_folder)
    thresholds = [round(x * 0.1, 1) for x in range(10, 0, -1)]
    results = []

    for model_name in sorted(os.listdir(root_folder)):
        model_path = os.path.join(root_folder, model_name)
        if not os.path.isdir(model_path):
            continue

        predicted_data = load_json_files(model_path)
        if not predicted_data:
            continue

        best_f1 = 0
        best_metrics = None
        best_threshold = None

        for t in thresholds:
            (link_precision, link_recall, link_f1,
             _, _, _) = calculate_metrics(gold_data, predicted_data, t)


            if metric_type == "keyword linking":
                precision, recall, f1 = link_precision, link_recall, link_f1
            else:
                raise ValueError("metric_type deve essere 'keyword extraction' o 'keyword linking'.")

            if f1 > best_f1:
                best_f1 = f1
                best_metrics = (precision, recall, f1)
                best_threshold = t

        if best_metrics:
            results.append({
                "LLM": model_name,
                "Precision": best_metrics[0],
                "Recall": best_metrics[1],
                "F1": best_metrics[2],
                "Threshold": best_threshold
            })

    # Crea tabella ordinata per F1
    if results:
        df = pd.DataFrame(results).sort_values(by="F1", ascending=False).reset_index(drop=True)
        print("\nTabella dei migliori modelli (ordinata per F1):")
        print(df.to_string(index=False))
        return df
    else:
        print("⚠️ Nessun modello valido trovato.")
        return pd.DataFrame(columns=["Modello", "Precision", "Recall", "F1", "Soglia"])


def fp_percentages_per_model(gold_folder, root_folder, jaccard_threshold, nullTP=False):
    """
    Calcola, per ogni modello (sottocartella di root_folder), le percentuali di
    False Positive dovuti a:
      - id errato (pred_key contiene ' (id wrong)')
      - stringa non presente nel gold (pred_key contiene ' (string not present in gold)')

    Usa la funzione calculate_metrics esistente, senza modificarne la logica.
    Ritorna un DataFrame con una riga per modello e disegna un grafico a torta
    per ciascun modello.
    """

    # Carico una sola volta il gold
    gold_data = load_json_files(gold_folder)
    if not gold_data:
        print("Nessun file gold valido trovato.")
        return pd.DataFrame()

    results = []

    # Ogni sottocartella di root_folder è un modello
    for model_name in sorted(os.listdir(root_folder)):
        model_path = os.path.join(root_folder, model_name)
        if not os.path.isdir(model_path):
            continue

        predicted_data = load_json_files(model_path)
        if not predicted_data:
            print(f"Modello '{model_name}': nessun file JSON di predizione trovato, salto.")
            continue

        try:
            # Riutilizzo ESATTAMENTE la tua funzione
            (_prec, _rec, _f1,
             _tp_per_file,
             false_positive_per_file,
             _fn_per_file) = calculate_metrics(gold_data, predicted_data, jaccard_threshold, nullTP=nullTP)
        except Exception as e:
            print(f"Errore nel modello '{model_name}': {e}")
            continue

        total_fp = 0
        fp_id_wrong = 0
        fp_string_not_in_gold = 0

        # Scorro tutti gli FP prodotti da calculate_metrics
        for filename, fps in false_positive_per_file.items():
            for fp in fps:
                total_fp += 1
                pred_key = fp.get("pred_key", "")

                if "(id wrong)" in pred_key:
                    fp_id_wrong += 1
                elif "(string not present in gold)" in pred_key:
                    fp_string_not_in_gold += 1
                # eventualmente qui potresti distinguere altri tipi di FP in futuro

        # Calcolo percentuali (se non ci sono FP → 0)
        if total_fp > 0:
            fp_id_ratio = fp_id_wrong / total_fp
            fp_string_ratio = fp_string_not_in_gold / total_fp
        else:
            fp_id_ratio = 0.0
            fp_string_ratio = 0.0

        # Salvo nei risultati tabellari
        results.append({
            "Model": model_name,
            "FP_total": total_fp,
            "FP_id_wrong_ratio": fp_id_ratio,
            "FP_string_not_in_gold_ratio": fp_string_ratio
        })

        # -------------------------------
        # GRAFICO A TORTA PER QUESTO MODELLO
        # -------------------------------
        if total_fp > 0:
            other_fp = total_fp - fp_id_wrong - fp_string_not_in_gold

            labels = []
            sizes = []

            if fp_id_wrong > 0:
                labels.append("Wrong QID")
                sizes.append(fp_id_wrong)
            if fp_string_not_in_gold > 0:
                labels.append("Wrong keywords")
                sizes.append(fp_string_not_in_gold)
            if other_fp > 0:
                labels.append("Altri FP")
                sizes.append(other_fp)

            # Disegno il pie chart
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # per farla rotonda
            ax.set_title(f"{model_name}")
            plt.show()
        else:
            print(f"'{model_name}': nessun FP, nessun grafico a torta.")

    if not results:
        print("Nessun modello valido trovato.")
        return pd.DataFrame(columns=["Model", "FP_total", "FP_id_wrong_ratio", "FP_string_not_in_gold_ratio"])

    df = pd.DataFrame(results).sort_values(by="Model").reset_index(drop=True)

    print("\nPercentuali di False Positive per modello:")
    print(df.to_string(index=False))

    return df
    
    
    
    
    
    
    
    
    
    
    
def most_recurrent_cases(
    gold_folder,
    root_folder,
    jaccard_threshold,
    case_type="FP",          # "TP", "FP", "FN"
    model_name=None,         # None = tutti i modelli, altrimenti es. "Relikk"
    nullTP=False,
    top_n=None,              # es. 20 per vedere solo i primi 20
    group_by="full_case"     # "full_case", "gold_label", "pred_key", "gold_wikidata_id", "pred_wikidata_id"
):
    """
    Calcola i TP / FP / FN più ricorrenti:
    - su tutti i modelli, oppure
    - su un solo modello se model_name è specificato.

    Parametri
    ---------
    case_type : str
        "TP", "FP" oppure "FN"
    model_name : str or None
        Nome del modello da filtrare. Se None, usa tutti i modelli.
    group_by : str
        Come raggruppare i casi:
        - "full_case"         -> usa tutti i campi del caso
        - "gold_label"
        - "pred_key"
        - "gold_wikidata_id"
        - "pred_wikidata_id"

    Ritorna
    -------
    pd.DataFrame
    """

    gold_data = load_json_files(gold_folder)
    if not gold_data:
        print("Nessun file gold valido trovato.")
        return pd.DataFrame()

    allowed_case_types = {"TP", "FP", "FN"}
    if case_type not in allowed_case_types:
        raise ValueError(f"case_type deve essere uno tra {allowed_case_types}")

    rows = []
    case_counter = Counter()
    case_models = defaultdict(set)
    case_files = defaultdict(set)

    for current_model in sorted(os.listdir(root_folder)):
        model_path = os.path.join(root_folder, current_model)
        if not os.path.isdir(model_path):
            continue

        if model_name is not None and current_model != model_name:
            continue

        predicted_data = load_json_files(model_path)
        if not predicted_data:
            continue

        try:
            (
                _prec,
                _rec,
                _f1,
                tp_per_file,
                fp_per_file,
                fn_per_file
            ) = calculate_metrics(gold_data, predicted_data, jaccard_threshold, nullTP=nullTP)
        except Exception as e:
            print(f"Errore nel modello '{current_model}': {e}")
            continue

        if case_type == "TP":
            selected = tp_per_file
        elif case_type == "FP":
            selected = fp_per_file
        else:
            selected = fn_per_file

        for filename, cases in selected.items():
            for case in cases:
                case_key = build_case_key(case, group_by=group_by, case_type=case_type)

                case_counter[case_key] += 1
                case_models[case_key].add(current_model)
                case_files[case_key].add(filename)

                row = {
                    "CaseType": case_type,
                    "Model": current_model,
                    "File": filename,
                    "OccurrenceKey": case_key
                }

                # Copia tutti i campi originali del caso
                row.update(case)
                rows.append(row)

    if not rows:
        print("Nessun caso trovato.")
        return pd.DataFrame()

    # Aggregazione finale
    aggregated_rows = []
    for case_key, count in case_counter.most_common():
        example_row = next(r for r in rows if r["OccurrenceKey"] == case_key)

        aggregated_row = {
            "CaseType": case_type,
            "OccurrenceKey": case_key,
            "Count": count,
            "NumModels": len(case_models[case_key]),
            "Models": ", ".join(sorted(case_models[case_key])),
            "NumFiles": len(case_files[case_key]),
            "Files": ", ".join(sorted(case_files[case_key]))
        }

        # aggiungo i campi descrittivi principali se esistono
        for field in ["gold_label", "pred_key", "gold_wikidata_id", "pred_wikidata_id"]:
            if field in example_row:
                aggregated_row[field] = example_row[field]

        aggregated_rows.append(aggregated_row)

    df = pd.DataFrame(aggregated_rows)

    # Ordino per frequenza
    df = df.sort_values(
        by=["Count", "NumModels", "NumFiles"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    if top_n is not None:
        df = df.head(top_n)

    print(f"\nCasi {case_type} più ricorrenti"
          + (f" per il modello '{model_name}'" if model_name else " su tutti i modelli")
          + f" (group_by='{group_by}'):\n")
    print(df.to_string(index=False))

    return df


def build_case_key(case, group_by="full_case", case_type="FP"):
    """
    Costruisce la chiave con cui contare i casi ricorrenti.
    """

    if group_by == "gold_label":
        return case.get("gold_label", "")

    if group_by == "pred_key":
        return case.get("pred_key", "")

    if group_by == "gold_wikidata_id":
        return case.get("gold_wikidata_id", "")

    if group_by == "pred_wikidata_id":
        return case.get("pred_wikidata_id", "")

    # group_by == "full_case"
    # raggruppa usando tutti i campi rilevanti
    if case_type in {"TP", "FP"}:
        return (
            case.get("gold_label", ""),
            case.get("pred_key", ""),
            case.get("gold_wikidata_id", ""),
            case.get("pred_wikidata_id", "")
        )
    elif case_type == "FN":
        return (
            case.get("gold_label", ""),
            case.get("gold_wikidata_id", "")
        )

    return tuple(sorted(case.items()))
    
def pretty_print_recurrent_cases(
    df,
    case_type="FP",
    model_name=None,
    show_files=False,
    max_files=10,
    show_models=True,
    max_models=10,
    top_n=20
):
    """
    Stampa in modo leggibile i casi ricorrenti.
    """

    if df.empty:
        print("Nessun caso da mostrare.")
        return

    df_to_print = df.copy().head(top_n)

    title = f"\nTop {len(df_to_print)} casi {case_type} più ricorrenti"
    if model_name:
        title += f" per il modello '{model_name}'"
    else:
        title += " su tutti i modelli"
    print(title)
    print("=" * len(title))

    for idx, row in df_to_print.iterrows():
        print(f"\n#{idx + 1}  |  Count: {row.get('Count', 0)}  |  File distinti: {row.get('NumFiles', 0)}  |  Modelli: {row.get('NumModels', 0)}")

        # 🔹 stampa lista modelli (NUOVO)
        if show_models:
            models_str = row.get("Models", "")
            if models_str:
                models_list = [m.strip() for m in models_str.split(",") if m.strip()]
                shown_models = models_list[:max_models]

                print(f"Models     : {', '.join(shown_models)}")
                if len(models_list) > max_models:
                    print(f"             ... +{len(models_list) - max_models} altri modelli")

        # campi principali
        gold_label = row.get("gold_label", "")
        pred_key = row.get("pred_key", "")
        gold_qid = row.get("gold_wikidata_id", "")
        pred_qid = row.get("pred_wikidata_id", "")
        fp_type = row.get("fp_type", "")

        if fp_type:
            print(f"FP type    : {fp_type}")

        if gold_label:
            print(f"Gold label : {gold_label}")
        if pred_key:
            print(f"Pred key   : {pred_key}")
        if gold_qid:
            print(f"Gold QID   : {gold_qid}")
        if pred_qid:
            print(f"Pred QID   : {pred_qid}")

        # 🔹 stampa file (come prima)
        if show_files:
            files_str = row.get("Files", "")
            files_list = [f.strip() for f in files_str.split(",") if f.strip()]
            shown_files = files_list[:max_files]

            if shown_files:
                print(f"Files      : {', '.join(shown_files)}")
                if len(files_list) > max_files:
                    print(f"             ... +{len(files_list) - max_files} altri file")


# parameters (change the root_folder for evaluating the other approaches)
metric_type = "keyword linking"
gold_folder = "goldStandard/"

root_folder = "Results/Results_with_Wikidata_QIDs/"

jaccard = 1


    
# print keywords for a jaccard threshold    
#process_folders_recursively(gold_folder, root_folder, jaccard, stampaFP=True,   model_name="Relik", file_name="1.json")
#process_folders_recursively(gold_folder, root_folder, jaccard, stampaFP=True, model_name="Relik")

#Print precision, recall and f1 of 1 approach
sorted_metrics = sort_metrics(root_folder, gold_folder, metric_type, jaccard)
print(f"\nResults order by F1 Score ({metric_type}):")
print("Model | Precision | Recall | F1 Score")
print("-" * 50)
for folder, precision, recall, f1_score in sorted_metrics:
    print(f"{folder} | {precision:.4f} | {recall:.4f} | {f1_score:.4f}")


# print results of the best jcaccard threshold  
df_results = best_f1_per_model(
    gold_folder=gold_folder,
    root_folder=root_folder,
    metric_type="keyword linking"   
)

# print plot of f1, precision and recall for each jaccard threshold 
plot_all_metrics_trend(gold_folder, root_folder)

f_fp = fp_percentages_per_model(gold_folder, root_folder, jaccard)
    

## conta FP/TP/FN
# df_fp_relik = most_recurrent_cases(
    # gold_folder=gold_folder,
    # root_folder=root_folder,
    # jaccard_threshold=jaccard,
    # case_type="FP",
    # model_name="Relik",
    # group_by="pred_key", 
    # top_n=20
# )
# pretty_print_recurrent_cases(
    # df_fp_relik,
    # case_type="FP",
    # model_name="Relik",
    # show_files=True,
    # max_files=5,
    # top_n=30
# )