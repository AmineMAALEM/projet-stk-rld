import torch
import os
from sb3_contrib import QRDQN  # <--- Si tu as utilisé Rainbow
from stable_baselines3 import PPO  # <--- Si tu as utilisé PPO (décommente la ligne)

# Noms des fichiers
ZIP_SOURCE = "stk_actor/my_model.zip"      # Ton fichier zip
PTH_DESTINATION = "stk_actor/pystk_actor.pth" # Ce que le prof veut

print(f"🔄 Chargement de {ZIP_SOURCE}...")

# Charge le modèle (force l'utilisation du CPU pour éviter les erreurs)
# Si tu es en PPO, remplace QRDQN par PPO ci-dessous
model = PPO(ZIP_SOURCE, device="cpu") 

print("extracting policy...")
# On prend juste les poids du réseau de neurones
policy_weights = model.policy.state_dict()

print(f"💾 Sauvegarde dans {PTH_DESTINATION}...")
torch.save(policy_weights, PTH_DESTINATION)

print("✅ Terminé ! Le fichier .pth est prêt.")