import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn as nn

class Actor(Agent):
    """Le Cerveau : Estime la valeur de chaque action"""
    def __init__(self, observation_space, action_space):
        super().__init__()
        # On calcule la taille de l'entrée (taille de l'image aplatie)
        input_size = 1
        for d in observation_space.shape:
            input_size *= d
            
        self.n_actions = action_space.n
        
        # Un réseau de neurones simple (MLP)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions) # Sortie = nombre d'actions possibles
        )

    def forward(self, t: int, **kwargs):
        # On récupère l'observation au temps t
        obs = self.get(("env/env_obs", t))
        # On passe dans le réseau
        scores = self.model(obs)
        # On stocke les scores (logits)
        self.set(("action_logits", t), scores)

class ArgmaxActor(Agent):
    """Le Décideur : Choisit l'action avec le plus grand score"""
    def forward(self, t: int, **kwargs):
        # On récupère les scores calculés par l'Actor
        scores = self.get(("action_logits", t))
        # On prend l'index du max
        action = scores.argmax(dim=-1)
        # On définit l'action à jouer
        self.set(("action", t), action)