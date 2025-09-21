#!/usr/bin/env python3
"""
Analyse des résultats de benchmark VoxConverse
==============================================

Script pour analyser les résultats CSV et déterminer la configuration optimale.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_benchmark_results():
    """Analyser les résultats du benchmark et trouver la configuration optimale."""
    
    # Charger les données
    df = pd.read_csv('output/voxconverse_benchmark_results.csv')
    
    print("🎯 ANALYSE DES RÉSULTATS DE BENCHMARK VOXCONVERSE")
    print("=" * 60)
    print(f"📊 Total configurations testées: {len(df)}")
    
    # Trier par performance training
    df_sorted_train = df.sort_values('train_samples_per_sec', ascending=False)
    
    # Trier par performance validation
    df_sorted_val = df.sort_values('val_samples_per_sec', ascending=False)
    
    print("\n🏆 TOP 5 CONFIGURATIONS - TRAINING PERFORMANCE:")
    print("-" * 50)
    for i, row in df_sorted_train.head(5).iterrows():
        print(f"{row['config_name']:25s} | {row['train_samples_per_sec']:8.1f} sps | "
              f"Batch: {row['batch_size']:2d} | Workers: {row['num_workers']:2d} | "
              f"Pin: {str(row['pin_memory']):5s} | PW: {str(row['persistent_workers']):5s}")
    
    print("\n🏆 TOP 5 CONFIGURATIONS - VALIDATION PERFORMANCE:")
    print("-" * 50)
    for i, row in df_sorted_val.head(5).iterrows():
        print(f"{row['config_name']:25s} | {row['val_samples_per_sec']:8.1f} sps | "
              f"Batch: {row['batch_size']:2d} | Workers: {row['num_workers']:2d} | "
              f"Pin: {str(row['pin_memory']):5s} | PW: {str(row['persistent_workers']):5s}")
    
    # Configuration optimale basée sur un score combiné
    df['combined_score'] = (df['train_samples_per_sec'] + df['val_samples_per_sec']) / 2
    best_overall = df.loc[df['combined_score'].idxmax()]
    
    print("\n🎯 CONFIGURATION OPTIMALE (Score combiné train+val):")
    print("=" * 60)
    print(f"Config: {best_overall['config_name']}")
    print(f"  📈 Performance Training:   {best_overall['train_samples_per_sec']:8.1f} samples/sec")
    print(f"  📈 Performance Validation: {best_overall['val_samples_per_sec']:8.1f} samples/sec")
    print(f"  🔄 Score Combiné:          {best_overall['combined_score']:8.1f} samples/sec")
    print(f"  📊 Batch Size:             {best_overall['batch_size']}")
    print(f"  👥 Num Workers:            {best_overall['num_workers']}")
    print(f"  📌 Pin Memory:             {best_overall['pin_memory']}")
    print(f"  🔄 Persistent Workers:     {best_overall['persistent_workers']}")
    print(f"  📦 Prefetch Factor:        {best_overall['prefetch_factor']}")
    print(f"  ⏱️  Temps Batch Training:   {best_overall['train_avg_batch_time']:.3f}s")
    print(f"  ⏱️  Temps Batch Validation: {best_overall['val_avg_batch_time']:.3f}s")
    
    # Analyse par paramètres
    print("\n📈 ANALYSE PAR PARAMÈTRES:")
    print("-" * 40)
    
    # Impact du batch size
    print("\n🔹 Impact Batch Size:")
    batch_analysis = df.groupby('batch_size')['combined_score'].agg(['mean', 'std', 'max']).round(1)
    for bs, stats in batch_analysis.iterrows():
        print(f"  Batch {bs:2d}: {stats['mean']:6.1f}±{stats['std']:5.1f} sps (max: {stats['max']:6.1f})")
    
    # Impact des workers
    print("\n🔹 Impact Number of Workers:")
    worker_analysis = df.groupby('num_workers')['combined_score'].agg(['mean', 'std', 'max']).round(1)
    for nw, stats in worker_analysis.iterrows():
        print(f"  Workers {nw:2d}: {stats['mean']:6.1f}±{stats['std']:5.1f} sps (max: {stats['max']:6.1f})")
    
    # Impact pin memory
    print("\n🔹 Impact Pin Memory:")
    pin_analysis = df.groupby('pin_memory')['combined_score'].agg(['mean', 'std', 'max']).round(1)
    for pm, stats in pin_analysis.iterrows():
        print(f"  Pin Memory {str(pm):5s}: {stats['mean']:6.1f}±{stats['std']:5.1f} sps (max: {stats['max']:6.1f})")
    
    # Impact persistent workers
    print("\n🔹 Impact Persistent Workers:")
    pw_analysis = df.groupby('persistent_workers')['combined_score'].agg(['mean', 'std', 'max']).round(1)
    for pw, stats in pw_analysis.iterrows():
        print(f"  Persistent {str(pw):5s}: {stats['mean']:6.1f}±{stats['std']:5.1f} sps (max: {stats['max']:6.1f})")
    
    # Recommandations
    print("\n💡 RECOMMANDATIONS:")
    print("-" * 30)
    
    # Meilleur batch size
    best_batch = batch_analysis['mean'].idxmax()
    print(f"✅ Batch Size optimal: {best_batch}")
    
    # Meilleur nombre de workers
    best_workers = worker_analysis['mean'].idxmax()
    print(f"✅ Nombre de Workers optimal: {best_workers}")
    
    # Pin memory recommandation
    best_pin = pin_analysis['mean'].idxmax()
    print(f"✅ Pin Memory recommandé: {best_pin}")
    
    # Persistent workers recommandation
    best_pw = pw_analysis['mean'].idxmax()
    print(f"✅ Persistent Workers recommandé: {best_pw}")
    
    print("\n🎯 CONFIGURATION FINALE RECOMMANDÉE:")
    print("=" * 45)
    print("```python")
    print("# Configuration optimale pour VoxConverse")
    print("train_loader, val_loader = create_voxconverse_dataloaders(")
    print(f"    batch_size={best_overall['batch_size']},")
    print(f"    num_workers={best_overall['num_workers']},")
    print(f"    pin_memory={best_overall['pin_memory']},")
    print(f"    persistent_workers={best_overall['persistent_workers']},")
    print(f"    prefetch_factor={best_overall['prefetch_factor']},")
    print("    segment_duration=4.0,")
    print("    validation_split=0.1")
    print(")")
    print("```")
    
    print(f"\n💥 Performance attendue:")
    print(f"  🚀 Training:   ~{best_overall['train_samples_per_sec']:.0f} samples/sec")
    print(f"  🚀 Validation: ~{best_overall['val_samples_per_sec']:.0f} samples/sec")
    
    return best_overall

if __name__ == "__main__":
    best_config = analyze_benchmark_results()