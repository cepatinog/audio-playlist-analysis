import os

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_FEATURES_DIR = os.path.join(DATA_DIR, 'processed', 'features')
MODELS_DIR = os.path.join(SRC_DIR, 'models')

# Audio file extensions to consider
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')

# Model file paths
TEMPO_MODEL_FILE         = os.path.join(MODELS_DIR, 'deeptemp-k16-3.pb')
EMB_DISCOGS_MODEL_FILE   = os.path.join(MODELS_DIR, 'discogs-effnet-bs64-1.pb')
EMB_MSD_MODEL_FILE       = os.path.join(MODELS_DIR, 'msd-musicnn-1.pb')
GENRE_MODEL_FILE         = os.path.join(MODELS_DIR, 'genre_discogs400-discogs-effnet-1.pb')
VOICE_MODEL_FILE         = os.path.join(MODELS_DIR, 'voice_instrumental-discogs-effnet-1.pb')
DANCEABILITY_MODEL_FILE  = os.path.join(MODELS_DIR, 'danceability-discogs-effnet-1.pb')
EMOTION_MODEL_FILE       = os.path.join(MODELS_DIR, 'emomusic-msd-musicnn-2.pb')

# Other constants
TEMPO_METHOD = 'tempocnn'
