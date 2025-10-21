from transformers import pipeline
import torch
import pandas as pd
import os
import time

print("begin")

vecteur_model = [
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0", #Deberta
    "joeddav/xlm-roberta-large-xnli", #roberta
    "mtheo/camembert-base-xnli" #camembert

]

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')
    print ("GPU not found.")

# vecteur_batch = [4, 8, 16, 32]
batch_size =16

print("labels")
# Catégories candidates
candidate_labels = [
    "Amour", "Affection", "Calme", "Plaisir", "Gratitude", "Soulagement",
    "Surprise", "Confusion", "Dégoût", "Détresse", "Honte", "Colère", "Indignation",
    "Déception", "Frustration", "Tristesse", "Désespoir", "Doute", "Espoir",
    "Peur", "Bonheur", "Joie", "Anxiété", "Fierté", "Acceptation",
    "Excitation", "Amusement", "Désir", "Respect", "Appréhension", "Angoisse",
    "Étonnement", "Satisfaction", "Insatisfaction", "Réjouissance", "Revendication",
    "Espérance", "Optimisme", "Pessimisme", "Enthousiasme", "Renoncement",
    "Intensité", "Expression", "Maîtrise", "Regret", "Dépit", "Mépris", "Inquiétude"
]

# Charger les données
# df = pd.read_csv('data/data_work.csv')

print("import fichier")
ditp = pd.read_csv('2025_10_21_DITP.csv', encoding="utf8", sep=";")

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

# Double boucle sur vecteur_model et vecteur_batch / Boucle vecteur batch supprimée
print("loop")
for model in vecteur_model:
    print(f"Traitement avec le modèle: {model} et batch_size: {batch_size}")

    # Charger le pipeline de zero-shot classification
    classifier = pipeline("zero-shot-classification", model=model, device=device, batch_size=batch_size,
                          multilabel=True)

    # Enregistrer le temps de début
    start_time = time.time()
    print(start_time)

    # Effectuer la classification pour chaque texte avec sauvegarde tous les 1000 lignes
    results = []
    results_data = []

    # Nettoyer le nom du modèle pour le nom de fichier
    cleaned_model = model.split('/')[-1]
    output_file = os.path.join(f'class_{cleaned_model}_B{batch_size}.csv')
    

    for i, text in enumerate(texts):
        result = classifier(text, candidate_labels, hypothesis_template="L'émotion de ce texte est {}.", multilabel=True)
        results.append(result)

        print(i)

        # Préparer les données pour ce résultat
        result_row = {'ID expérience': ids[i], 'Texte': text, 'Modèle': model}
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



