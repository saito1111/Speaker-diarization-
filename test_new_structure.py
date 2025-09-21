#!/usr/bin/env python3
"""
Test script pour valider la nouvelle structure VAD/OSD/VCN
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from voxconverse_dataset import VoxConverseDataset, create_voxconverse_dataloaders

def test_new_structure():
    """Test de la nouvelle structure frame-wise VAD/OSD/VCN."""
    print("🔧 Test de la nouvelle structure VAD/OSD/VCN...")
    
    try:
        # Créer un petit dataset pour tester
        dataset = VoxConverseDataset(
            split='dev',
            segment_duration=4.0,
            hop_duration=2.0
        )
        
        print(f"✅ Dataset créé avec {len(dataset)} segments")
        
        # Tester un échantillon
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n🔍 Structure d'un échantillon:")
            print(f"  Features shape: {sample['features'].shape}")
            print(f"  VAD labels shape: {sample['vad_labels'].shape}")
            print(f"  OSD labels shape: {sample['osd_labels'].shape}")
            print(f"  VCN labels shape: {sample['vcn_labels'].shape}")
            
            # Vérifier les valeurs
            vad_frames = torch.sum(sample['vad_labels'] > 0).item()
            osd_frames = torch.sum(sample['osd_labels'] > 0).item()
            vcn_frames = torch.sum(sample['vcn_labels'] > 0).item()
            
            print(f"\n📊 Activité frame-wise pour cet échantillon:")
            print(f"  VAD frames actives: {vad_frames}/{sample['vad_labels'].shape[0]} ({vad_frames/sample['vad_labels'].shape[0]*100:.1f}%)")
            print(f"  OSD frames actives: {osd_frames}/{sample['osd_labels'].shape[0]} ({osd_frames/sample['osd_labels'].shape[0]*100:.1f}%)")
            print(f"  VCN frames actives: {vcn_frames}/{sample['vcn_labels'].shape[0]} ({vcn_frames/sample['vcn_labels'].shape[0]*100:.1f}%)")
            
            # Tester la fonction de dataloader
            print(f"\n🔧 Test du DataLoader...")
            train_loader, val_loader = create_voxconverse_dataloaders(
                batch_size=4,
                num_workers=0,
                segment_duration=4.0,
                debug_overlap_stats=True
            )
            
            # Tester un batch
            batch = next(iter(train_loader))
            print(f"\n📦 Structure d'un batch:")
            print(f"  Features: {batch['features'].shape}")
            print(f"  VAD labels: {batch['vad_labels'].shape}")
            print(f"  OSD labels: {batch['osd_labels'].shape}")
            print(f"  VCN labels: {batch['vcn_labels'].shape}")
            
            # Statistiques du batch
            print(f"\n📊 Statistiques du batch:")
            vad_activity = torch.sum(batch['vad_labels'] > 0).item()
            osd_activity = torch.sum(batch['osd_labels'] > 0).item()
            vcn_activity = torch.sum(batch['vcn_labels'] > 0).item()
            total_frames = batch['vad_labels'].numel()
            
            print(f"  VAD activity: {vad_activity}/{total_frames} frames ({vad_activity/total_frames*100:.2f}%)")
            print(f"  OSD activity: {osd_activity}/{total_frames} frames ({osd_activity/total_frames*100:.2f}%)")
            print(f"  VCN activity: {vcn_activity}/{total_frames} frames ({vcn_activity/total_frames*100:.2f}%)")
            
            return True
            
    except Exception as e:
        print(f"❌ ERREUR lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_detailed_sample():
    """Test détaillé d'un échantillon avec overlap."""
    print(f"\n🔍 Test détaillé d'échantillons...")
    
    try:
        dataset = VoxConverseDataset(
            split='dev',
            segment_duration=4.0,
            hop_duration=2.0
        )
        
        # Chercher un échantillon avec OSD
        osd_sample = None
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            if torch.sum(sample['osd_labels'] > 0) > 0:
                osd_sample = sample
                print(f"✅ Trouvé échantillon avec OSD à l'index {i}")
                break
        
        if osd_sample is not None:
            print(f"\n📋 Analyse détaillée de l'échantillon OSD:")
            
            vad = osd_sample['vad_labels'].numpy()
            osd = osd_sample['osd_labels'].numpy()
            vcn = osd_sample['vcn_labels'].numpy()
            
            print(f"  Durée: {len(vad)} frames ({len(vad)*0.02:.1f}s)")
            print(f"  VAD frames: {np.sum(vad > 0)}")
            print(f"  OSD frames: {np.sum(osd > 0)}")
            print(f"  VCN frames: {np.sum(vcn > 0)}")
            
            # Afficher quelques frames détaillées
            osd_indices = np.where(osd > 0)[0]
            if len(osd_indices) > 0:
                print(f"  Premières frames OSD: {osd_indices[:5]}")
                print(f"  Temps des OSD: {osd_indices[:5] * 0.02}s")
        
        # Chercher un échantillon avec VCN
        vcn_sample = None
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            if torch.sum(sample['vcn_labels'] > 0) > 0:
                vcn_sample = sample
                print(f"✅ Trouvé échantillon avec VCN à l'index {i}")
                break
        
        if vcn_sample is not None:
            vcn = vcn_sample['vcn_labels'].numpy()
            vcn_indices = np.where(vcn > 0)[0]
            print(f"\n📋 Analyse VCN:")
            print(f"  VCN frames: {np.sum(vcn > 0)}")
            print(f"  Positions VCN: {vcn_indices[:10]}")
            print(f"  Temps VCN: {vcn_indices[:10] * 0.02}s")
            
        return True
        
    except Exception as e:
        print(f"❌ ERREUR lors du test détaillé: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_new_structure()
    success2 = test_detailed_sample()
    
    if success1 and success2:
        print("\n🎉 Tous les tests réussis! La nouvelle structure VAD/OSD/VCN fonctionne.")
    else:
        print("\n💥 Certains tests ont échoué. Il y a encore des problèmes à corriger.")