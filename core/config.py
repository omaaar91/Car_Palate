import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
LOC_MODEL_PATH   = os.path.join(BASE_DIR, "best_char.pt")
RECOG_MODEL_PATH = os.path.join(BASE_DIR, "Tuning_char.pt")

# Character Mapping (Model English label -> Arabic character)
CHAR_MAP = {
    'alif': 'أ', 'baa': 'ب', 'taa': 'ت', 'thaa': 'ث', 'jeem': 'ج',
    '7aa': 'ح', 'khaa': 'خ', 'daal': 'د', 'zaal': 'ذ', 'raa': 'ر',
    'zay': 'ز', 'seen': 'س', 'sheen': 'ش', 'saad': 'ص', 'daad': 'ض',
    'Taa': 'ط', 'Thaa': 'ظ', 'ain': 'ع', 'ghayn': 'غ', 'faa': 'ف',
    'qaaf': 'ق', 'kaaf': 'ك', 'laam': 'ل', 'meem': 'م', 'noon': 'ن',
    'haa': 'هـ', 'waw': 'و', 'yaa': 'ي',
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
}

DIGITS_SET = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

# Prior calibration weights (rebalances dataset imbalance for underrepresented classes)
CLASS_WEIGHTS = {
    'qaaf': 25.0,
    'faa': 2.5
}

# Detection & Recognition Thresholds
PLATE_CONF = 0.35
LOC_CONF = 0.14
RECOG_CONF = 0.003
SCALE_FACTOR = 4.0
