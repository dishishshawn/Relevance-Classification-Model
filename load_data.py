import os

def load_data(relevant_folder, irrelevant_folder):
    texts = []
    labels = []

    # Load relevant texts
    for filename in os.listdir(relevant_folder):
        with open(os.path.join(relevant_folder, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
            labels.append(1)  # Relevant

    # Load irrelevant texts
    for filename in os.listdir(irrelevant_folder):
        with open(os.path.join(irrelevant_folder, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
            labels.append(0)  # Irrelevant

    return texts, labels