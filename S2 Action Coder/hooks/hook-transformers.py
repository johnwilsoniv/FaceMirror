# hook-transformers.py - Force eager loading of wav2vec2 models
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect everything from transformers
datas, binaries, hiddenimports = collect_all('transformers')

# Explicitly add wav2vec2 submodules
hiddenimports += collect_submodules('transformers.models.wav2vec2')
hiddenimports += [
    'transformers.models.wav2vec2.modeling_wav2vec2',
    'transformers.models.wav2vec2.processing_wav2vec2',
    'transformers.models.wav2vec2.configuration_wav2vec2',
    'transformers.models.wav2vec2.feature_extraction_wav2vec2',
    'transformers.models.wav2vec2.tokenization_wav2vec2',
]
