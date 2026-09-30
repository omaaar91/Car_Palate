import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
LOC_MODEL_PATH   = os.path.join(BASE_DIR, "best_char.pt")
RECOG_MODEL_PATH = os.path.join(BASE_DIR, "best_char_afterTuning.pt")

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

# الحروف الرسمية المعتمدة قانوناً في اللوحات المصرية (17 حرفاً فقط)
# الحروف الممنوعة رسمياً لتجنب الالتباس: (ت، ث، ح، خ، ذ، ز، ش، ض، ظ، غ، ك) كما لا يوجد الرقم صفر
OFFICIAL_ALLOWED_LETTERS = {'أ', 'ب', 'ج', 'د', 'ر', 'س', 'ص', 'ط', 'ع', 'ف', 'ق', 'ل', 'م', 'ن', 'هـ', 'و', 'ي'}

# جدول التحويل الحتمي للحروف المتشابهة والمحظورة إلى نظيرها القانوني المعتمد
CANONICAL_LETTER_MAP = {
    'sheen': 'seen',  # ش ممنوعة -> س
    'daad': 'saad',   # ض ممنوعة -> ص
    'Thaa': 'Taa',    # ظ ممنوعة -> ط
    'ghayn': 'ain',   # غ ممنوعة -> ع
    'khaa': 'jeem',   # خ ممنوعة -> ج
    '7aa': 'jeem',    # ح ممنوعة -> ج
    'thaa': 'baa',    # ث ممنوعة -> ب
    'taa': 'baa',     # ت ممنوعة -> ب
    'zaal': 'daal',   # ذ ممنوعة -> د
    'zay': 'raa',     # ز ممنوعة -> ر
    'kaaf': 'laam'    # ك ممنوعة -> ل
}

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
