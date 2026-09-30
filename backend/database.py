import os
import sqlite3
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "alpr_records.db")
CROPS_DIR = os.path.join(DB_DIR, "crops")

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

class ALPRDatabase:
    """
    نظام إدارة قاعدة البيانات وسجلات مرور المركبات والقوائم السوداء
    (High-Performance SQLite with In-Memory Caching & Anti-Spam Cooldown)
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._blacklist_cache: Dict[str, dict] = {}
        self._recent_detections: Dict[str, float] = {}  # لمنع التكرار (Cooldown)
        self.init_db()
        self.reload_blacklist_cache()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def init_db(self):
        """تهيئة الجداول والفهارس في قاعدة البيانات"""
        with self.get_connection() as conn:
            # 1. جدول سجل مرور المركبات
            conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date_str TEXT NOT NULL,
                time_str TEXT NOT NULL,
                plate_text TEXT NOT NULL,
                digits TEXT NOT NULL,
                letters TEXT NOT NULL,
                governorate TEXT NOT NULL,
                confidence REAL NOT NULL,
                syntax_valid BOOLEAN NOT NULL,
                is_blacklist BOOLEAN NOT NULL DEFAULT 0,
                blacklist_reason TEXT DEFAULT '',
                crop_path TEXT DEFAULT ''
            );
            """)

            # 2. جدول القائمة السوداء (السيارات المطلوبة أمنياً / المحظورة)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS blacklist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_norm TEXT UNIQUE NOT NULL,
                plate_display TEXT NOT NULL,
                reason TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'danger',
                created_at TEXT NOT NULL
            );
            """)

            # فهارس للسرعة الفائقة في البحث
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON vehicle_logs(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_plate ON vehicle_logs(plate_text);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_norm ON blacklist(plate_norm);")

            # إضافة عينات تجريبية للقائمة السوداء إذا كان الجدول فارغاً
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM blacklist;")
            if cursor.fetchone()["cnt"] == 0:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sample_blacklists = [
                    ("8429_س_ب", "[٨ ٤ ٢ ٩] | [س ب]", "سيارة مسروقة - بلاغ قسم الهرم 2026", "danger", now_str),
                    ("2962_ق_ق", "[٢ ٩ ٦ ٢] | [ق ق]", "مطلوبة لنيابة المرور - مخالفات جسيمة", "warning", now_str),
                    ("125_س_س_ل", "[١ ٢ ٥] | [س س ل]", "مركبة محظور دخولها من إدارة الأمن", "danger", now_str)
                ]
                conn.executemany("""
                INSERT OR IGNORE INTO blacklist (plate_norm, plate_display, reason, severity, created_at)
                VALUES (?, ?, ?, ?, ?);
                """, sample_blacklists)
                conn.commit()

    def normalize_plate(self, digits: list, letters: list) -> str:
        """توحيد صيغة مفتاح اللوحة للمطابقة الصارمة (مثال: '8429_س_ب')"""
        # تحويل الأرقام العربية إلى الإنجليزية للتوحيد
        ar_to_en = {'٠':'0', '١':'1', '٢':'2', '٣':'3', '٤':'4', '٥':'5', '٦':'6', '٧':'7', '٨':'8', '٩':'9'}
        d_norm = "".join([ar_to_en.get(d, str(d)) for d in digits])
        l_norm = "_".join(letters)
        return f"{d_norm}_{l_norm}"

    def reload_blacklist_cache(self):
        """تحديث الكاش في الذاكرة للبحث الفوري O(1) في البث الحي"""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM blacklist;").fetchall()
            self._blacklist_cache = {
                row["plate_norm"]: {
                    "id": row["id"],
                    "plate_display": row["plate_display"],
                    "reason": row["reason"],
                    "severity": row["severity"]
                }
                for row in rows
            }

    def check_blacklist(self, digits: list, letters: list) -> Optional[dict]:
        """فحص لحظي فائق السرعة إذا كانت اللوحة مدرجة بالقائمة السوداء"""
        norm_key = self.normalize_plate(digits, letters)
        return self._blacklist_cache.get(norm_key)

    def log_vehicle(self, plate_data: dict, crop_img: Optional[np.ndarray] = None, cooldown_seconds: float = 20.0) -> Tuple[bool, Optional[dict]]:
        """
        تسجيل السيارة في قاعدة البيانات مع حماية ذكية ضد التكرار (Anti-Duplicate Cooldown)
        """
        digits = plate_data.get("digits", [])
        letters = plate_data.get("letters", [])
        text = plate_data.get("text", "")
        if not text or "?" in text or not digits:
            return False, None

        norm_key = self.normalize_plate(digits, letters)
        curr_t = time.time()

        # فحص كاش مانع التكرار (إذا ظهرت نفس السيارة خلال مدة الـ Cooldown)
        if norm_key in self._recent_detections:
            if curr_t - self._recent_detections[norm_key] < cooldown_seconds:
                return False, None  # سيارة مسجلة للتو، تجاهل التكرار

        self._recent_detections[norm_key] = curr_t

        # فحص القائمة السوداء
        bl_info = self.check_blacklist(digits, letters)
        is_bl = bl_info is not None
        bl_reason = bl_info["reason"] if is_bl else ""

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # حفظ صورة اللوحة المقصوصة كدليل إثبات مرئي
        crop_rel_path = ""
        if crop_img is not None:
            filename = f"crop_{int(curr_t)}_{norm_key}.jpg"
            full_crop_path = os.path.join(CROPS_DIR, filename)
            cv2.imwrite(full_crop_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            crop_rel_path = f"/crops/{filename}"

        with self.get_connection() as conn:
            cursor = conn.execute("""
            INSERT INTO vehicle_logs (
                timestamp, date_str, time_str, plate_text, digits, letters, 
                governorate, confidence, syntax_valid, is_blacklist, 
                blacklist_reason, crop_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                timestamp_str, date_str, time_str, text,
                " ".join(digits), " ".join(letters),
                plate_data.get("governorate", "غير محدد"),
                float(plate_data.get("confidence", 0.9)),
                bool(plate_data.get("syntax_valid", True)),
                is_bl, bl_reason, crop_rel_path
            ))
            conn.commit()
            log_id = cursor.lastrowid

        return True, {
            "id": log_id,
            "timestamp": timestamp_str,
            "plate_text": text,
            "is_blacklist": is_bl,
            "blacklist_reason": bl_reason,
            "crop_path": crop_rel_path
        }

    def get_logs(self, limit: int = 50, offset: int = 0, query: str = "", date_filter: str = "") -> List[dict]:
        """استرجاع سجلات المرور مع البحث والفلترة"""
        with self.get_connection() as conn:
            sql = "SELECT * FROM vehicle_logs WHERE 1=1"
            params = []

            if query:
                sql += " AND (plate_text LIKE ? OR digits LIKE ? OR letters LIKE ? OR governorate LIKE ?)"
                q_wild = f"%{query}%"
                params.extend([q_wild, q_wild, q_wild, q_wild])

            if date_filter:
                sql += " AND date_str = ?"
                params.append(date_filter)

            sql += " ORDER BY id DESC LIMIT ? OFFSET ?;"
            params.extend([limit, offset])

            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_stats_summary(self) -> dict:
        """حساب إحصائيات المرور اليومية والإجمالية للوحة التحكم"""
        today_str = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            # إجمالي اليوم
            today_count = conn.execute("SELECT COUNT(*) as c FROM vehicle_logs WHERE date_str = ?;", (today_str,)).fetchone()["c"]
            # إجمالي التنبيهات الأمنية اليوم
            today_alerts = conn.execute("SELECT COUNT(*) as c FROM vehicle_logs WHERE date_str = ? AND is_blacklist = 1;", (today_str,)).fetchone()["c"]
            # إجمالي كل السجلات
            total_count = conn.execute("SELECT COUNT(*) as c FROM vehicle_logs;").fetchone()["c"]
            # إجمالي سيارات القائمة السوداء
            total_blacklist = conn.execute("SELECT COUNT(*) as c FROM blacklist;").fetchone()["c"]

            # توزيع المحافظات اليوم
            gov_rows = conn.execute("""
            SELECT governorate, COUNT(*) as cnt 
            FROM vehicle_logs 
            WHERE date_str = ? 
            GROUP BY governorate 
            ORDER BY cnt DESC;
            """, (today_str,)).fetchall()
            gov_distribution = {r["governorate"]: r["cnt"] for r in gov_rows}

            return {
                "today_count": today_count,
                "today_alerts": today_alerts,
                "total_count": total_count,
                "total_blacklist": total_blacklist,
                "gov_distribution": gov_distribution
            }

    def get_blacklist_items(self) -> List[dict]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM blacklist ORDER BY id DESC;").fetchall()
            return [dict(r) for r in rows]

    def add_to_blacklist(self, digits: list, letters: list, reason: str, severity: str = "danger") -> bool:
        norm_key = self.normalize_plate(digits, letters)
        d_str = " ".join(digits)
        l_str = " ".join(letters)
        display_text = f"[{d_str}] | [{l_str}]"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with self.get_connection() as conn:
                conn.execute("""
                INSERT OR REPLACE INTO blacklist (plate_norm, plate_display, reason, severity, created_at)
                VALUES (?, ?, ?, ?, ?);
                """, (norm_key, display_text, reason, severity, now_str))
                conn.commit()
            self.reload_blacklist_cache()
            return True
        except Exception:
            return False

    def remove_from_blacklist(self, item_id: int) -> bool:
        try:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM blacklist WHERE id = ?;", (item_id,))
                conn.commit()
            self.reload_blacklist_cache()
            return True
        except Exception:
            return False

# متغير النظام الموحد
_db_instance: Optional[ALPRDatabase] = None

def get_alpr_database() -> ALPRDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = ALPRDatabase()
    return _db_instance
