#!/usr/bin/env python3
"""
Script pour inspecter la structure du modèle PyTorch sauvegardé
"""

import torch
import torch.nn as nn

def inspect_model():
    """Inspecter le modèle sauvegardé"""
    
    try:
        # Charger le modèle
        print("Chargement du modèle best_brain_tumor_model.pth...")
        checkpoint = torch.load('best_brain_tumor_model.pth', map_location='cpu')
        
        print(f"Type du checkpoint: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print("\nClés disponibles dans le checkpoint:")
            for key in checkpoint.keys():
                print(f"  - {key}")
                
            # Si c'est un state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
        else:
            # Le checkpoint est directement le state_dict
            state_dict = checkpoint
            
        print(f"\nNombre de paramètres dans le state_dict: {len(state_dict)}")
        
        print("\nStructure des couches:")
        for name, param in state_dict.items():
            print(f"  {name}: {param.shape}")
            
        # Analyser la structure pour deviner l'architecture
        print("\n" + "="*50)
        print("ANALYSE DE L'ARCHITECTURE")
        print("="*50)
        
        conv_layers = []
        fc_layers = []
        
        for name, param in state_dict.items():
            if 'conv' in name and 'weight' in name:
                conv_layers.append((name, param.shape))
            elif 'fc' in name and 'weight' in name:
                fc_layers.append((name, param.shape))
                
        print(f"\nCouches de convolution détectées: {len(conv_layers)}")
        for name, shape in conv_layers:
            print(f"  {name}: {shape} -> (out_channels={shape[0]}, in_channels={shape[1]}, kernel_size={shape[2]}x{shape[3]})")
            
        print(f"\nCouches fully connected détectées: {len(fc_layers)}")
        for name, shape in fc_layers:
            print(f"  {name}: {shape} -> (out_features={shape[0]}, in_features={shape[1]})")
            
        # Déterminer le nombre de classes
        if fc_layers:
            last_fc = fc_layers[-1]
            num_classes = last_fc[1][0]  # out_features de la dernière couche FC
            print(f"\nNombre de classes détecté: {num_classes}")
            
        return state_dict, conv_layers, fc_layers
        
    except Exception as e:
        print(f"Erreur lors de l'inspection: {e}")
        return None, [], []

def create_model_architecture(conv_layers, fc_layers):
    """Créer l'architecture du modèle basée sur l'inspection"""
    
    print("\n" + "="*50)
    print("GÉNÉRATION DE L'ARCHITECTURE")
    print("="*50)
    
    if not conv_layers or not fc_layers:
        print("Impossible de déterminer l'architecture automatiquement")
        return None
        
    # Déterminer le nombre de classes
    num_classes = fc_layers[-1][1][0]
    
    print(f"Génération d'une architecture pour {num_classes} classes...")
    
    # Créer le code de l'architecture
    architecture_code = f"""
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes={num_classes}):
        super(BrainTumorCNN, self).__init__()
        
        # Couches de convolution"""
    
    for i, (name, shape) in enumerate(conv_layers):
        out_ch, in_ch, k_size, _ = shape
        architecture_code += f"""
        self.{name.split('.')[0]} = nn.Conv2d({in_ch}, {out_ch}, kernel_size={k_size}, padding=1)"""
    
    architecture_code += """
        
        # Couches de pooling et autres
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        
        # Couches fully connected"""
    
    for i, (name, shape) in enumerate(fc_layers):
        out_feat, in_feat = shape
        architecture_code += f"""
        self.{name.split('.')[0]} = nn.Linear({in_feat}, {out_feat})"""
    
    architecture_code += """
        
    def forward(self, x):
        # Appliquer les couches de convolution avec pooling"""
    
    for i, (name, _) in enumerate(conv_layers):
        layer_name = name.split('.')[0]
        architecture_code += f"""
        x = self.pool(self.relu(self.{layer_name}(x)))"""
    
    architecture_code += """
        
        # Aplatir et appliquer les couches FC
        x = self.flatten(x)"""
    
    for i, (name, _) in enumerate(fc_layers[:-1]):
        layer_name = name.split('.')[0]
        architecture_code += f"""
        x = self.dropout(self.relu(self.{layer_name}(x)))"""
    
    # Dernière couche sans activation
    last_layer = fc_layers[-1][0].split('.')[0]
    architecture_code += f"""
        x = self.{last_layer}(x)
        
        return x
"""
    
    print("Architecture générée:")
    print(architecture_code)
    
    return architecture_code

if __name__ == "__main__":
    print("=== INSPECTION DU MODÈLE PYTORCH ===")
    
    state_dict, conv_layers, fc_layers = inspect_model()
    
    if state_dict is not None:
        architecture_code = create_model_architecture(conv_layers, fc_layers)
        
        if architecture_code:
            # Sauvegarder l'architecture dans un fichier
            with open('model_architecture.py', 'w') as f:
                f.write("import torch.nn as nn\n")
                f.write(architecture_code)
            
            print(f"\nArchitecture sauvegardée dans 'model_architecture.py'")
        
        print("\n=== INSPECTION TERMINÉE ===")
    else:
        print("Échec de l'inspection du modèle")
