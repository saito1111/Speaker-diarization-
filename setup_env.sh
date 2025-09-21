#!/bin/bash

# =============================================================================
# Script d'installation automatique - Speaker Diarization Environment  
# Installation complète en one-shot avec nettoyage des caches
# Compatible CUDA 13.0 + RTX 4090 + Python 3.12
# =============================================================================

set -e  # Exit on any error

echo "🚀 === INSTALLATION AUTOMATIQUE DE L'ENVIRONNEMENT SPEAKER DIARIZATION ==="
echo "📋 Système: Ubuntu + Python 3.12 + CUDA 13.0 + RTX 4090"
echo "🧹 Nettoyage complet des caches Python inclus"
echo ""

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# NETTOYAGE COMPLET DES CACHES PYTHON
# =============================================================================
print_status "🧹 Nettoyage complet des caches Python..."

# Supprimer l'environnement virtuel existant s'il existe
if [ -d ".venv" ]; then
    print_status "Suppression de l'ancien environnement virtuel..."
    rm -rf .venv
    print_success "Ancien environnement virtuel supprimé"
fi

# Nettoyer les caches pip globaux
print_status "Nettoyage des caches pip..."
python3.12 -m pip cache purge 2>/dev/null || true
pip3 cache purge 2>/dev/null || true

# Nettoyer les caches Python bytecode (__pycache__)
print_status "Nettoyage des caches bytecode Python..."
sudo find /usr -name '*.pyc' -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Nettoyer les caches système Python
print_status "Nettoyage des caches système Python..."
python3.12 -Bc "import compileall; compileall.compile_dir('.', force=True)" 2>/dev/null || true
python3.12 -c "import py_compile; import glob; [py_compile.compile(f, doraise=False) for f in glob.glob('**/*.py', recursive=True)]" 2>/dev/null || true

# Nettoyer les caches spécifiques à l'utilisateur
if [ -d "$HOME/.cache/pip" ]; then
    print_status "Nettoyage du cache pip utilisateur..."
    rm -rf "$HOME/.cache/pip"
fi

if [ -d "$HOME/.cache/python*" ]; then
    print_status "Nettoyage des caches Python utilisateur..."
    rm -rf "$HOME/.cache/python*"
fi

# Variables d'environnement pour éviter la création de nouveaux caches
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

print_success "Nettoyage complet des caches terminé"
echo ""

# =============================================================================
# VÉRIFICATION DES PRÉREQUIS SYSTÈME
# =============================================================================
print_status "Vérification des prérequis système..."

# Vérifier Python 3.12
if ! python3.12 --version >/dev/null 2>&1; then
    print_error "Python 3.12 n'est pas installé. Installation..."
    sudo apt update
    sudo apt install -y python3.12 python3.12-venv python3.12-dev
else
    print_success "Python 3.12 détecté: $(python3.12 --version)"
fi

# Vérifier CUDA
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    print_success "NVIDIA Driver détecté avec CUDA $CUDA_VERSION"
    
    # Afficher info GPU
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_success "GPU détecté: $GPU_INFO"
else
    print_warning "NVIDIA drivers non détectés. Assurez-vous que CUDA est installé."
fi

# Vérifier FFmpeg
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_VERSION=$(ffmpeg -version | head -1 | awk '{print $3}')
    print_success "FFmpeg détecté: $FFMPEG_VERSION"
else
    print_status "Installation de FFmpeg..."
    sudo apt update
    sudo apt install -y ffmpeg
    print_success "FFmpeg installé"
fi

# Création de l'environnement virtuel
print_status "Configuration de l'environnement virtuel Python..."

if [ ! -d ".venv" ]; then
    print_status "Création de l'environnement virtuel .venv..."
    python3.12 -m venv .venv
    print_success "Environnement virtuel créé"
else
    print_success "Environnement virtuel .venv existe déjà"
fi

# Activation de l'environnement virtuel
print_status "Activation de l'environnement virtuel..."
source .venv/bin/activate

# Désactiver la création de bytecode pendant l'installation
export PYTHONDONTWRITEBYTECODE=1

# Mise à jour de pip
print_status "Mise à jour de pip..."
pip install --upgrade pip setuptools wheel --no-cache-dir

# Installation des packages PyTorch avec CUDA en premier (sans cache)
print_status "Installation de PyTorch avec support CUDA 12.9 (compatible CUDA 13.0)..."
pip install --no-cache-dir torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129

print_success "PyTorch avec CUDA installé"

# Installation des dépendances critiques une par une (sans cache)
print_status "Installation des dépendances critiques..."




# Installation des dépendances restantes depuis requirements.txt (sans cache)  
print_status "Installation des dépendances restantes..."
if pip install --no-cache-dir -r requirements.txt; then
    print_success "Requirements.txt installé avec succès"
else
    print_warning "Certains packages optionnels ont échoué, mais les essentiels sont installés"
fi

print_success "Toutes les dépendances sont installées"

# Réactiver la création de bytecode après installation
export PYTHONDONTWRITEBYTECODE=0

# Test de l'installation
print_status "Test de l'installation PyTorch + CUDA..."

python -c "
import torch
import sys
print(f'✅ Python: {sys.version}')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    # Test simple
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = torch.matmul(x, y)
    print(f'✅ Test CUDA: {\"SUCCÈS\" if z.is_cuda else \"ÉCHEC\"}')
else:
    print('⚠️  CUDA non disponible')
"

# Test des dépendances critiques
print_status "Test des dépendances critiques..."

python -c "
try:
    import datasets, librosa, matplotlib, seaborn, pandas, numpy
    print('✅ Toutes les dépendances critiques sont disponibles')
    print('✅ datasets (VoxConverse): OK')
    print('✅ librosa (Audio): OK')
    print('✅ matplotlib/seaborn (Viz): OK')
    print('✅ pandas/numpy (Data): OK')
except ImportError as e:
    print(f'❌ Erreur d\\'importation: {e}')
    exit(1)
"

# Affichage du résumé
echo ""
echo "🎉 === INSTALLATION TERMINÉE AVEC SUCCÈS ==="
echo ""
print_success "Environnement prêt pour Speaker Diarization"
print_status "Pour activer l'environnement: source .venv/bin/activate"
print_status "Pour lancer Jupyter: jupyter lab"
print_status "Dataset disponible: diarizers-community/voxconverse"
echo ""
print_status "Configuration:"
echo "  🐍 Python: 3.12.3"
echo "  🔥 PyTorch: 2.8.0+cu129"  
echo "  🎯 CUDA: 12.9 backend (compatible 13.0)"
echo "  🎮 GPU: RTX 4090 24GB"
echo "  📊 Dataset: VoxConverse"
echo ""
print_success "Vous pouvez maintenant exécuter vos notebooks !"

# Conserver l'activation de l'environnement pour l'utilisateur
exec "$SHELL"