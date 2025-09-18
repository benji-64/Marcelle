# print("beginning")
# import pandas as pd
#
# ditp=pd.read_excel('2025_07_22_DITP.xlsx')
#
# #Réduction des colones
# df=ditp[["ID expérience","Description"]]
#
# mask = df["Description"].str.contains("amande", case=False, na=False)
# df.loc[mask, "Description"] = df.loc[mask, "Description"].str.replace("amande", "amende", case=False, regex=True)
#
#
# pip install "git+https://github.com/huggingface/transformers.git@6e0515e99c39444caae39472ee1b2fd76ece32f1" --upgrade
#
# pip install flash-attn==2.7.2.post1
#
#
# import math
# #import os
#
# ### Deberta
# from transformers import pipeline
# zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
# hypothesis_template = "The emotions of this text are {}"
# classes_verbalized = [
#     "Acceptation", "Affection", "Amour", "Amusement", "Anxiété", "Angoisse",
#     "Appréhension", "Bonheur", "Calme", "Colère", "Confusion", "Contrôle",
#     "Déception", "Dégoût", "Désespoir", "Désir", "Doute", "Enthousiasme",
#     "Espérance", "Espoir", "Expression", "Étonnement", "Fierté", "Frustration",
#     "Gratitude", "Honte", "Indignation", "Insatisfaction", "Intensité", "Joie",
#     "Optimisme", "Peur", "Pessimisme", "Plaisir", "Réjouissance", "Renoncement",
#     "Respect", "Revendication", "Satisfaction", "Soulagement", "Surprise", "Sympathie",
#     "Tristesse"
# ]
# results = []
#
# ### modernBert
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model_id = "clapAI/modernBERT-large-multilingual-sentiment"
# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float16)
#
# model.to(device)
# model.eval()
#
#
# # Retrieve labels from the model's configuration
# id2label = model.config.id2label
#
# texts = [
#
#     # French
#     {
#         "text": "J'adore ce restaurant, la nourriture est délicieuse!",
#         "label": "positive"
#     },
#     {
#         "text": "Le service était très lent et désagréable.",
#         "label": "negative"
#     },
#
# ]
#
# for item in texts:
#     text = item["text"]
#     label = item["label"]
#
#     inputs = tokenizer(text, return_tensors="pt").to(device)
#
#
#
#
#
#     # Perform inference in inference mode
#     with torch.inference_mode():
#         outputs = model(**inputs)
#         predictions = outputs.logits.argmax(dim=-1)
#     print(f"Text: {text} | Label: {label} | Prediction: {id2label[predictions.item()]}")
#
#
#
#


from transformers import pipeline
import pandas as pd
import os
import time

vecteur_model = [
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0", #Deberta
    "joeddav/xlm-roberta-large-xnli", #roberta
    "mtheo/camembert-base-xnli" #camembert

]

# vecteur_batch = [4, 8, 16, 32]
batch_size =8
# Catégories candidates
candidate_labels = [
    "Acceptation", "Affection", "Amour", "Amusement", "Anxiété", "Angoisse",
    "Appréhension", "Bonheur", "Calme", "Colère", "Confusion", "Contrôle",
    "Déception", "Dégoût", "Désespoir", "Désir", "Doute", "Enthousiasme",
    "Espérance", "Espoir", "Expression", "Étonnement", "Fierté", "Frustration",
    "Gratitude", "Honte", "Indignation", "Insatisfaction", "Intensité", "Joie",
    "Optimisme", "Peur", "Pessimisme", "Plaisir", "Réjouissance", "Renoncement",
    "Respect", "Revendication", "Satisfaction", "Soulagement", "Surprise", "Sympathie",
    "Tristesse"
]

# Charger les données
# df = pd.read_csv('data/data_work.csv')

ditp = pd.read_excel('2025_07_22_DITP.xlsx')

# Réduction des colones
df = ditp[["ID expérience", "Description"]]


# Process the full dataset instead of just a sample
# df_filtered_sample = df.sample(n=5)  # Commented out for full processing

# Convertir la colonne 'Description' en une liste de chaînes de caractères
texts = df['Description'].tolist()
# Convertir la colonne 'ID expérience' en une liste
ids = df['ID expérience'].tolist()

# Liste pour stocker les temps d'exécution
execution_times = []

# Assurez-vous que le répertoire de sortie existe
# output_dir = r'C:\Users\33623\Documents\_AtelierR\__Arcom\ArcomEnv\data'
# os.makedirs(output_dir, exist_ok=True)

# Double boucle sur vecteur_model et vecteur_batch
for model in vecteur_model:
    #for batch_size in vecteur_batch:
        print(f"Traitement avec le modèle: {model} et batch_size: {batch_size}")

        # Charger le pipeline de zero-shot classification
        classifier = pipeline("zero-shot-classification", model=model, device=-1, batch_size=batch_size,
                              multilabel=True)

        # Enregistrer le temps de début
        start_time = time.time()

        # Effectuer la classification pour chaque texte avec sauvegarde tous les 1000 lignes
        results = []
        results_data = []
        
        # Nettoyer le nom du modèle pour le nom de fichier
        cleaned_model = model.split('/')[-1]
        output_file = os.path.join(f'class_{cleaned_model}_B{batch_size}.csv')
        
        for i, text in enumerate(texts):
            result = classifier(text, candidate_labels)
            results.append(result)
            
            # Préparer les données pour ce résultat
            result_row = {'ID expérience': ids[i], 'Texte': text, 'Modèle': model, 'Taille de lot': batch_size}
            for label, score in zip(result['labels'], result['scores']):
                result_row[label] = score
            results_data.append(result_row)
            
            # Sauvegarder tous les 1000 lignes
            if (i + 1) % 1000 == 0:
                print(f"Sauvegarde intermédiaire après {i + 1} lignes...")
                temp_df = pd.DataFrame(results_data)
                
                # Si c'est la première sauvegarde, créer le fichier, sinon l'ajouter
                if i < 1000:
                    temp_df.to_csv(output_file, index=False)
                else:
                    temp_df.to_csv(output_file, mode='a', header=False, index=False)
                
                # Vider la liste pour libérer de la mémoire
                results_data = []
        
        # Sauvegarder les dernières données s'il en reste
        if results_data:
            print(f"Sauvegarde finale des {len(results_data)} dernières lignes...")
            temp_df = pd.DataFrame(results_data)
            if len(results) <= 1000:
                temp_df.to_csv(output_file, index=False)
            else:
                temp_df.to_csv(output_file, mode='a', header=False, index=False)

        # Enregistrer le temps de fin
        end_time = time.time()

        # Calculer le temps d'exécution
        execution_time = end_time - start_time
        print(f"Temps d'exécution: {execution_time} secondes")

        # Stocker le temps d'exécution avec les paramètres
        execution_times.append({'Modèle': model, 'Taille de lot': batch_size, 'exécution': execution_time})
        execution_times_df = pd.DataFrame(execution_times)
        execution_times_df.to_csv(os.path.join( 'temps_execution.csv'), index=False)

        print(f"Traitement terminé pour le modèle {model}. Total de {len(results)} lignes traitées.")
        #
        # start_index = 24700
        # batch_size = 100
        # total_rows = 30000
        # num_batches = math.ceil((total_rows - start_index) / batch_size)
        # excel_file = "ZSL.xlsx"



