"""
TruthScope Streamlit Application
محرك اكتشاف الحقيقة/الشائعات باستخدام الذكاء الاصطناعي
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import requests
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.tokenize import sent_tokenize
import re

# تحميل بيانات nltk المطلوبة
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="TruthScope - محرك اكتشاف الحقيقة",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E3A8A;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .credibility-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .credibility-high {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    
    .credibility-medium {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    
    .credibility-low {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    
    .credibility-fake {
        background-color: #721c24;
        color: white;
        border-left: 5px solid #dc3545;
    }
    
    .source-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 12px;
        margin: 2px;
    }
    
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    
    .article-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .article-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .rtl-text {
        text-align: right;
        direction: rtl;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown("""
<div class="main-header">
    <h1 style="color: white;">🔍 TruthScope - محرك اكتشاف الحقيقة</h1>
    <p style="color: white; font-size: 18px;">تحليل مصداقية الأخبار والشائعات باستخدام الذكاء الاصطناعي</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# 1. فئات إدارة الحالة ونظام التخزين
# ============================================

class DatabaseManager:
    """مدير قاعدة البيانات SQLite"""
    
    def __init__(self, db_name="truthscope.db"):
        self.db_name = db_name
        self.init_database()
    
    def get_connection(self):
        """إنشاء اتصال بقاعدة البيانات"""
        conn = sqlite3.connect(self.db_name, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """تهيئة جداول قاعدة البيانات"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # جدول المستخدمين
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reputation_score INTEGER DEFAULT 100
            )
        ''')
        
        # جدول الأخبار
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                url TEXT UNIQUE,
                source_domain TEXT,
                author TEXT,
                published_date TIMESTAMP,
                collected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                collection_source TEXT,
                credibility_score REAL DEFAULT 0.0,
                confidence_level REAL DEFAULT 0.0,
                credibility_category TEXT,
                ai_analysis TEXT,
                verified BOOLEAN DEFAULT 0
            )
        ''')
        
        # جدول تقييمات المستخدمين
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                article_id INTEGER NOT NULL,
                vote INTEGER,  -- 1 for agree, -1 for disagree, 0 for neutral
                comment TEXT,
                confidence INTEGER DEFAULT 3,  -- 1-5
                feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES news_articles (id)
            )
        ''')
        
        # جدول سجل التحليلات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                analysis_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result TEXT,
                FOREIGN KEY (article_id) REFERENCES news_articles (id)
            )
        ''')
        
        # إنشاء فهرس للبحث السريع
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_url ON news_articles(url)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_score ON news_articles(credibility_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_article ON user_feedback(article_id)')
        
        conn.commit()
        conn.close()
    
    def save_article(self, article_data: Dict) -> int:
        """حفظ مقال جديد"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO news_articles 
                (title, content, url, source_domain, author, published_date, collection_source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                article_data['title'],
                article_data['content'],
                article_data.get('url', ''),
                article_data.get('source_domain', ''),
                article_data.get('author', 'مجهول'),
                article_data.get('published_date', datetime.now()),
                article_data.get('collection_source', 'manual')
            ))
            
            article_id = cursor.lastrowid
            conn.commit()
            return article_id
            
        except Exception as e:
            st.error(f"خطأ في حفظ المقال: {str(e)}")
            return None
        finally:
            conn.close()
    
    def update_article_analysis(self, article_id: int, analysis: Dict):
        """تحديث تحليل المقال"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE news_articles 
                SET credibility_score = ?, 
                    confidence_level = ?,
                    credibility_category = ?,
                    ai_analysis = ?
                WHERE id = ?
            ''', (
                analysis['credibility_score'],
                analysis['confidence_level'],
                analysis['category'],
                json.dumps(analysis, ensure_ascii=False),
                article_id
            ))
            
            conn.commit()
            
            # تسجيل التحليل
            cursor.execute('''
                INSERT INTO analysis_logs (article_id, analysis_type, result)
                VALUES (?, ?, ?)
            ''', (article_id, 'ai_analysis', 'success'))
            conn.commit()
            
        except Exception as e:
            st.error(f"خطأ في تحديث التحليل: {str(e)}")
        finally:
            conn.close()
    
    def save_feedback(self, article_id: int, feedback_data: Dict):
        """حفظ تقييم المستخدم"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO user_feedback (article_id, vote, comment, confidence)
                VALUES (?, ?, ?, ?)
            ''', (
                article_id,
                feedback_data.get('vote', 0),
                feedback_data.get('comment', ''),
                feedback_data.get('confidence', 3)
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            st.error(f"خطأ في حفظ التقييم: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_article(self, article_id: int) -> Optional[Dict]:
        """جلب مقال بواسطة ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM news_articles WHERE id = ?', (article_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def search_articles(self, query: str = None, limit: int = 50) -> List[Dict]:
        """بحث في المقالات"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if query:
            cursor.execute('''
                SELECT * FROM news_articles 
                WHERE title LIKE ? OR content LIKE ?
                ORDER BY collected_date DESC
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', limit))
        else:
            cursor.execute('''
                SELECT * FROM news_articles 
                ORDER BY collected_date DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_statistics(self) -> Dict:
        """جلب إحصائيات النظام"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # عدد المقالات
        cursor.execute('SELECT COUNT(*) FROM news_articles')
        stats['total_articles'] = cursor.fetchone()[0]
        
        # متوسط المصداقية
        cursor.execute('SELECT AVG(credibility_score) FROM news_articles WHERE credibility_score > 0')
        stats['avg_credibility'] = cursor.fetchone()[0] or 0
        
        # توزيع التصنيفات
        cursor.execute('''
            SELECT credibility_category, COUNT(*) 
            FROM news_articles 
            WHERE credibility_category IS NOT NULL
            GROUP BY credibility_category
        ''')
        stats['category_distribution'] = dict(cursor.fetchall())
        
        # عدد التقييمات
        cursor.execute('SELECT COUNT(*) FROM user_feedback')
        stats['total_feedbacks'] = cursor.fetchone()[0]
        
        conn.close()
        return stats

# ============================================
# 2. وحدة جمع البيانات
# ============================================

class NewsCollector:
    """وحدة جمع البيانات من مصادر مختلفة"""
    
    def __init__(self):
        self.trusted_sources = {
            'العربية': ['alarabiya.net', 'alarabiya.com'],
            'الجزيرة': ['aljazeera.net', 'aljazeera.com'],
            'بي بي سي': ['bbc.com', 'bbc.co.uk', 'bbc.com/arabic'],
            'رويترز': ['reuters.com', 'reuters.tv'],
            'فرانس 24': ['france24.com', 'france24.com/ar'],
            'سكاي نيوز': ['skynewsarabia.com'],
            'سي إن إن': ['cnnarabic.com', 'arabic.cnn.com']
        }
    
    def extract_from_url(self, url: str) -> Optional[Dict]:
        """استخراج محتوى من رابط"""
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 TruthScope Bot'
            })
            response.raise_for_status()
            
            # استخراج البيانات (تبسيط)
            # في الإنتاج، نستخدم BeautifulSoup لتحليل HTML
            title = url.split('/')[-1].replace('-', ' ')
            content = "محتويات المقال المستخرج من الموقع..."
            
            return {
                'title': title,
                'content': content,
                'url': url,
                'source_domain': url.split('/')[2],
                'author': 'مجهول',
                'published_date': datetime.now(),
                'collection_source': 'manual'
            }
            
        except Exception as e:
            st.warning(f"تعذر استخراج المحتوى: {str(e)}")
            return None
    
    def check_source_reliability(self, domain: str) -> float:
        """فحص موثوقية المصدر"""
        domain = domain.lower()
        
        for source_name, source_domains in self.trusted_sources.items():
            for source_domain in source_domains:
                if source_domain in domain:
                    return 85.0 + np.random.uniform(0, 15)  # 85-100 للمصادر الموثوقة
        
        # تحليل النطاق
        score = 50.0
        
        # خصم للنطاقات المشبوهة
        suspicious = ['blog', 'free', 'click', 'buzz', 'viral', 'wordpress', 'blogspot']
        for keyword in suspicious:
            if keyword in domain:
                score -= 20
        
        return max(10.0, min(score, 100.0))

# ============================================
# 3. وحدة التحليل الذكي
# ============================================

class CredibilityAnalyzer:
    """محلل المصداقية باستخدام الذكاء الاصطناعي"""
    
    def __init__(self):
        self.model = None
        self.load_model()
        
        # قواعد اكتشاف الشائعات
        self.fake_news_patterns = [
            r'تأكيد\s+مصدر\s+مطلع',
            r'تكشف\s+وثائق\s+سرية',
            r'بعد\s+صدمة\s+كبيرة',
            r'مفاجأة\s+غير\s+متوقعة',
            r'ينشر\s+لأول\s+مرة',
            r'صورة\s+تثبت',
            r'فيديو\s+مدوّي',
            r'يكشف\s+الستار'
        ]
        
        self.trust_indicators = [
            r'نقلاً\s+عن\s+مصادر\s+رسمية',
            r'بحسب\s+بيان\s+صادر',
            r'صرح\s+مسؤول',
            r'أفادت\s+وكالة\s+الأنباء',
            r'نشرت\s+الجريدة\s+الرسمية',
            r'كشف\s+التقرير\s+الرسمي'
        ]
    
    def load_model(self):
        """تحميل نموذج الذكاء الاصطناعي"""
        try:
            # استخدام نموذج خفيف للاستخدام على Streamlit Cloud
            @st.cache_resource
            def load_ai_model():
                return SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
            self.model = load_ai_model()
            st.success("✅ تم تحميل نموذج الذكاء الاصطناعي بنجاح")
            
        except Exception as e:
            st.warning(f"⚠️ تعذر تحميل النموذج الكامل: {str(e)}")
            st.info("سيتم استخدام التحليل التقليدي")
    
    def analyze_article(self, article: Dict, collector: NewsCollector) -> Dict:
        """تحليل المقال وتقييم مصداقيته"""
        
        scores = {
            'content_score': 0.0,
            'source_score': 0.0,
            'style_score': 0.0,
            'verification_score': 0.0
        }
        
        # 1. تقييم المصدر
        if 'source_domain' in article:
            scores['source_score'] = collector.check_source_reliability(article['source_domain'])
        
        # 2. تحليل المحتوى
        scores['content_score'] = self.analyze_content(article.get('content', ''))
        
        # 3. تحليل أسلوب الكتابة
        scores['style_score'] = self.analyze_writing_style(article.get('content', ''))
        
        # 4. تحليل التناسق (نموذج AI)
        if self.model and article.get('content'):
            scores['verification_score'] = self.analyze_with_ai(article['content'])
        else:
            scores['verification_score'] = 50.0
        
        # 5. حساب النتيجة النهائية
        final_score = self.calculate_final_score(scores)
        
        # 6. تصنيف المصداقية
        category = self.categorize_credibility(final_score)
        
        # 7. توليد التفسيرات
        explanations = self.generate_explanations(scores, category, article)
        
        return {
            'credibility_score': final_score,
            'confidence_level': self.calculate_confidence(scores),
            'category': category,
            'component_scores': scores,
            'explanations': explanations,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def analyze_content(self, content: str) -> float:
        """تحليل محتوى النص"""
        if not content:
            return 50.0
        
        score = 70.0  # درجة أساسية
        
        # البحث عن مؤشرات الشائعات
        for pattern in self.fake_news_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score -= 15
        
        # البحث عن مؤشرات المصداقية
        for pattern in self.trust_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                score += 10
        
        # تحليل طول النص
        word_count = len(content.split())
        if word_count < 50:
            score -= 10  # النصوص القصيرة مشبوهة
        elif word_count > 500:
            score += 5   # النصوص الطويلة أكثر تفصيلاً
        
        return max(0.0, min(score, 100.0))
    
    def analyze_writing_style(self, content: str) -> float:
        """تحليل أسلوب الكتابة"""
        if not content:
            return 50.0
        
        score = 65.0
        
        # تحليل علامات التعجب والتكبير
        excl_count = content.count('!')
        ques_count = content.count('?')
        word_count = len(content.split())
        
        if word_count > 0:
            excl_ratio = excl_count / word_count
            if excl_ratio > 0.005:  # أكثر من 0.5%
                score -= excl_ratio * 1000
        
        # تحليل الصفات المبالغ فيها
        exaggeration_words = ['مذهل', 'صادم', 'مفاجئ', 'كارثة', 'خطير للغاية']
        for word in exaggeration_words:
            if word in content:
                score -= 5
        
        return max(20.0, min(score, 100.0))
    
    def analyze_with_ai(self, content: str) -> float:
        """تحليل النص باستخدام نموذج الذكاء الاصطناعي"""
        try:
            if len(content) > 1000:
                content = content[:1000]
            
            # ترميز النص
            embedding = self.model.encode(content, convert_to_tensor=True)
            
            # جمل مرجعية للتحقق (مثال)
            reference_texts = [
                "تقرير رسمي صادر عن الجهات المختصة يؤكد صحة المعلومات",
                "خبر غير مؤكد يحتاج إلى تحقق من مصادر أخرى",
                "معلومات مضللة تهدف إلى التضليل والإشاعة"
            ]
            
            reference_embeddings = self.model.encode(reference_texts, convert_to_tensor=True)
            
            # حساب التشابه
            cos_scores = util.pytorch_cos_sim(embedding, reference_embeddings)[0]
            
            # تحويل التشابه إلى درجة
            # فرضية: الجملة الأولى هي الأكثر مصداقية
            credibility_score = float(cos_scores[0]) * 100
            
            return max(0.0, min(credibility_score, 100.0))
            
        except Exception as e:
            st.warning(f"خطأ في تحليل AI: {str(e)}")
            return 50.0
    
    def calculate_final_score(self, scores: Dict) -> float:
        """حساب النتيجة النهائية"""
        weights = {
            'source_score': 0.35,
            'content_score': 0.30,
            'style_score': 0.20,
            'verification_score': 0.15
        }
        
        final_score = 0
        for key, weight in weights.items():
            final_score += scores[key] * weight
        
        return round(final_score, 2)
    
    def calculate_confidence(self, scores: Dict) -> float:
        """حساب مستوى الثقة"""
        values = list(scores.values())
        variance = np.var(values)
        
        # كلما قل التباين، زادت الثقة
        confidence = 100 - (variance * 2)
        
        return round(max(30.0, min(confidence, 100.0)), 2)
    
    def categorize_credibility(self, score: float) -> str:
        """تصنيف درجة المصداقية"""
        if score >= 80:
            return 'high'  # موثوق
        elif score >= 60:
            return 'medium'  # معقول
        elif score >= 40:
            return 'low'  # مشكوك فيه
        else:
            return 'fake'  # شائعة
    
    def generate_explanations(self, scores: Dict, category: str, article: Dict) -> Dict:
        """توليد تفسيرات للنتيجة"""
        explanations = []
        
        if scores['source_score'] < 40:
            explanations.append("المصدر غير معروف أو غير موثوق")
        elif scores['source_score'] > 80:
            explanations.append("المصدر معروف وموثوق")
        
        if scores['content_score'] < 40:
            explanations.append("المحتوى يحتوي على مؤشرات الشائعات")
        elif scores['content_score'] > 70:
            explanations.append("المحتوى متوازن وواقعي")
        
        if scores['style_score'] < 40:
            explanations.append("أسلوب الكتابة درامي ومبالغ فيه")
        
        if scores['verification_score'] < 40:
            explanations.append("التحليل الآلي يشير إلى عدم الاتساق")
        
        return {
            'summary': '، '.join(explanations) if explanations else 'التحليل يشير إلى نتيجة متوسطة',
            'recommendation': self.get_recommendation(category),
            'source_evaluation': f"تقييم المصدر: {scores['source_score']}/100",
            'content_evaluation': f"تقييم المحتوى: {scores['content_score']}/100"
        }
    
    def get_recommendation(self, category: str) -> str:
        """توليد توصية بناءً على التصنيف"""
        recommendations = {
            'high': '✅ يمكن الوثوق بهذا الخبر، لكن يفضل التحقق من مصادر أخرى',
            'medium': '⚠️ الخبر معقول ولكن يحتاج إلى تأكيد من مصادر إضافية',
            'low': '🚨 الشكوك حول هذا الخبر كبيرة، تجنب نشره دون تحقق',
            'fake': '⛔ هذه على الأرجح شائعة، لا تنشر ولا تشارك'
        }
        return recommendations.get(category, 'تحتاج إلى مزيد من التحقق')

# ============================================
# 4. واجهة المستخدم - Streamlit App
# ============================================

def main():
    # تهيئة الحالة الجلسة
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()
    
    if 'collector' not in st.session_state:
        st.session_state.collector = NewsCollector()
    
    if 'analyzer' not in st.session_state:
        with st.spinner('جاري تحميل نموذج الذكاء الاصطناعي...'):
            st.session_state.analyzer = CredibilityAnalyzer()
    
    # شريط جانبي للتنقل
    st.sidebar.markdown("## 🧭 التنقل")
    page = st.sidebar.radio(
        "اختر الصفحة:",
        ["🏠 الصفحة الرئيسية", "🔍 تحليل خبر", "📊 الإحصائيات", "📜 قاعدة البيانات", "⚙️ الإعدادات"]
    )
    
    # الصفحة الرئيسية
    if page == "🏠 الصفحة الرئيسية":
        show_home_page()
    
    # صفحة تحليل الخبر
    elif page == "🔍 تحليل خبر":
        show_analysis_page()
    
    # صفحة الإحصائيات
    elif page == "📊 الإحصائيات":
        show_statistics_page()
    
    # صفحة قاعدة البيانات
    elif page == "📜 قاعدة البيانات":
        show_database_page()
    
    # صفحة الإعدادات
    elif page == "⚙️ الإعدادات":
        show_settings_page()

def show_home_page():
    """عرض الصفحة الرئيسية"""
    
    st.markdown("""
    <div class="rtl-text">
        <h2>مرحباً بك في TruthScope</h2>
        <p>أداة متقدمة لاكتشاف الشائعات وتحليل مصداقية الأخبار باستخدام الذكاء الاصطناعي.</p>
        
        <h3>🎯 كيف تعمل الأداة؟</h3>
        <ol>
            <li>أدخل نص الخبر أو الرابط المراد تحليله</li>
            <li>يقوم النظام بتحليل المصدر والمحتوى والأسلوب</li>
            <li>يستخدم الذكاء الاصطناعي لتقييم المصداقية</li>
            <li>تحصل على تقرير مفصل مع درجة موثوقية</li>
        </ol>
        
        <h3>🔍 مجالات الاستخدام</h3>
        <ul>
            <li>التحقق من الأخبار قبل نشرها</li>
            <li>مراقبة الشائعات على وسائل التواصل</li>
            <li>مساعدة الصحفيين في التحقق من المعلومات</li>
            <li>البحث الأكاديمي في تحليل الشائعات</li>
        </ul>
        
        <h3>📈 إحصائيات سريعة</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # عرض إحصائيات سريعة
    stats = st.session_state.db.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("?? إجمالي المقالات", stats['total_articles'])
    
    with col2:
        st.metric("⭐ متوسط المصداقية", f"{stats['avg_credibility']:.1f}")
    
    with col3:
        st.metric("💬 عدد التقييمات", stats['total_feedbacks'])
    
    with col4:
        categories = stats.get('category_distribution', {})
        verified_count = sum(categories.get(cat, 0) for cat in ['high', 'medium'])
        st.metric("✅ أخبار موثوقة", verified_count)
    
    # أمثلة على تحليلات حديثة
    st.markdown("### 📰 أحدث التحليلات")
    
    articles = st.session_state.db.search_articles(limit=5)
    
    if articles:
        for article in articles:
            display_article_card(article, show_analysis=False)
    else:
        st.info("لا توجد تحليلات سابقة. ابدأ بتحليل أول خبر!")

def show_analysis_page():
    """صفحة تحليل الخبر"""
    
    st.markdown("""
    <div class="rtl-text">
        <h2>🔍 تحليل مصداقية الخبر</h2>
        <p>أدخل نص الخبر أو الرابط المراد تحليل مصداقيته.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # خيارات الإدخال
    input_method = st.radio(
        "طريقة الإدخال:",
        ["📝 إدخال نص مباشر", "🔗 إدخال رابط", "🔎 البحث في قاعدة البيانات"],
        horizontal=True
    )
    
    article_data = None
    
    if input_method == "📝 إدخال نص مباشر":
        article_data = get_text_input()
    
    elif input_method == "🔗 إدخال رابط":
        article_data = get_url_input()
    
    elif input_method == "🔎 البحث في قاعدة البيانات":
        article_data = search_database()
    
    # تحليل الخبر
    if article_data and st.button("🚀 بدء التحليل", type="primary", use_container_width=True):
        with st.spinner("جاري التحليل... قد يستغرق بضع ثوانٍ"):
            # تحليل المصداقية
            analysis = st.session_state.analyzer.analyze_article(
                article_data, 
                st.session_state.collector
            )
            
            # حفظ في قاعدة البيانات
            if 'url' in article_data and article_data['url']:
                article_id = st.session_state.db.save_article(article_data)
                if article_id:
                    st.session_state.db.update_article_analysis(article_id, analysis)
                    article_data['id'] = article_id
            
            # عرض النتائج
            display_analysis_results(article_data, analysis)
            
            # قسم التقييم
            st.markdown("---")
            st.markdown("### 💬 شاركنا رأيك في التحليل")
            
            if 'id' in article_data:
                display_feedback_section(article_data['id'])
            else:
                st.info("تعذر حفظ التقييم لأن المقال غير محفوظ في قاعدة البيانات")

def show_statistics_page():
    """صفحة الإحصائيات"""
    
    st.markdown("""
    <div class="rtl-text">
        <h2>📊 إحصائيات النظام</h2>
        <p>نظرة عامة على تحليلات المصداقية في النظام.</p>
    </div>
    """, unsafe_allow_html=True)
    
    stats = st.session_state.db.get_statistics()
    
    # مخطط توزيع المصداقية
    st.markdown("### 📈 توزيع درجات المصداقية")
    
    articles = st.session_state.db.search_articles(limit=100)
    
    if articles:
        df = pd.DataFrame(articles)
        
        if 'credibility_score' in df.columns:
            # مخطط توزيع الدرجات
            fig = px.histogram(
                df, 
                x='credibility_score',
                nbins=20,
                title='توزيع درجات المصداقية',
                labels={'credibility_score': 'درجة المصداقية', 'count': 'عدد المقالات'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            # مخطط تصنيفات المصداقية
            if 'credibility_category' in df.columns:
                category_counts = df['credibility_category'].value_counts()
                
                fig2 = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title='توزيع تصنيفات المصداقية',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig2.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig2, use_container_width=True)
            
            # عرض جدول إحصائي
            st.markdown("### 📋 ملخص إحصائي")
            
            stats_summary = {
                'المتوسط': df['credibility_score'].mean(),
                'الوسيط': df['credibility_score'].median(),
                'الانحراف المعياري': df['credibility_score'].std(),
                'الأعلى': df['credibility_score'].max(),
                'الأدنى': df['credibility_score'].min()
            }
            
            stats_df = pd.DataFrame(list(stats_summary.items()), columns=['المقياس', 'القيمة'])
            st.dataframe(stats_df, use_container_width=True)
    
    else:
        st.info("لا توجد بيانات كافية لعرض الإحصائيات")

def show_database_page():
    """صفحة قاعدة البيانات"""
    
    st.markdown("""
    <div class="rtl-text">
        <h2>📜 قاعدة البيانات</h2>
        <p>تصفح جميع المقالات التي تم تحليلها.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # خيارات البحث والتصفية
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input("🔍 بحث في العناوين والمحتوى")
    
    with col2:
        credibility_filter = st.selectbox(
            "تصفية حسب المصداقية",
            ["الكل", "موثوق (80-100)", "معقول (60-80)", "مشكوك فيه (40-60)", "شائعة (أقل من 40)"]
        )
    
    # جلب المقالات
    articles = st.session_state.db.search_articles(query=search_query if search_query else None)
    
    # تطبيق الفلتر
    if credibility_filter != "الكل":
        if credibility_filter == "موثوق (80-100)":
            articles = [a for a in articles if a.get('credibility_score', 0) >= 80]
        elif credibility_filter == "معقول (60-80)":
            articles = [a for a in articles if 60 <= a.get('credibility_score', 0) < 80]
        elif credibility_filter == "مشكوك فيه (40-60)":
            articles = [a for a in articles if 40 <= a.get('credibility_score', 0) < 60]
        elif credibility_filter == "شائعة (أقل من 40)":
            articles = [a for a in articles if a.get('credibility_score', 0) < 40]
    
    st.markdown(f"### 📊 النتائج ({len(articles)} مقالة)")
    
    if articles:
        # خيارات العرض
        view_mode = st.radio(
            "طريقة العرض:",
            ["📋 عرض جدولي", "🪧 عرض بطاقات"],
            horizontal=True
        )
        
        if view_mode == "📋 عرض جدولي":
            # تحضير البيانات للجدول
            table_data = []
            for article in articles:
                table_data.append({
                    'ID': article.get('id'),
                    'العنوان': article.get('title', '')[:100] + '...' if len(article.get('title', '')) > 100 else article.get('title', ''),
                    'المصدر': article.get('source_domain', ''),
                    'التاريخ': article.get('collected_date', ''),
                    'المصداقية': f"{article.get('credibility_score', 0):.1f}",
                    'التصنيف': article.get('credibility_category', 'غير محدد')
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            
            # خيار التصدير
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 تصدير البيانات كـ CSV",
                data=csv,
                file_name="truthscope_database.csv",
                mime="text/csv"
            )
        
        else:
            # عرض البطاقات
            for article in articles:
                display_article_card(article, show_analysis=True)
    
    else:
        st.info("لا توجد مقالات تطابق معايير البحث")

def show_settings_page():
    """صفحة الإعدادات"""
    
    st.markdown("""
    <div class="rtl-text">
        <h2>⚙️ إعدادات النظام</h2>
        <p>تخصيص إعدادات محرك التحقق من الشائعات.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # إعدادات عامة
    st.markdown("### ⚙️ الإعدادات العامة")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_save = st.checkbox("حفظ التحليلات تلقائياً", value=True)
        show_details = st.checkbox("عرض التفاصيل التقنية", value=False)
    
    with col2:
        default_language = st.selectbox("اللغة الافتراضية", ["العربية", "الإنجليزية"])
        theme = st.selectbox("السمة", ["فاتحة", "داكنة"])
    
    # إعدادات التحليل
    st.markdown("### 🔧 إعدادات التحليل")
    
    st.info("""
    **معايير تقييم المصداقية:**
    - **موثوق:** 80-100 نقطة
    - **معقول:** 60-80 نقطة  
    - **مشكوك فيه:** 40-60 نقطة
    - **شائعة:** أقل من 40 نقطة
    """)
    
    # مصادق API (للتطوير المستقبلي)
    st.markdown("### 🔑 مفاتيح API (اختياري)")
    
    with st.expander("إضافة مفاتيح API للخدمات الخارجية"):
        newsapi_key = st.text_input("مفتاح NewsAPI", type="password")
        twitter_key = st.text_input("مفتاح Twitter API", type="password")
        
        if st.button("حفظ مفاتيح API"):
            st.success("تم حفظ المفاتيح (هذا مثال، في التطبيق الحقيقي سيتم تخزينها في قاعدة بيانات آمنة)")
    
    # إدارة قاعدة البيانات
    st.markdown("### 🗄️ إدارة قاعدة البيانات")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 تحديث قاعدة البيانات", type="secondary"):
            st.session_state.db.init_database()
            st.success("تم تحديث قاعدة البيانات")
    
    with col2:
        if st.button("🧹 مسح البيانات المؤقتة", type="secondary"):
            st.warning("هذا الإجراء سيحذف جميع البيانات. هل أنت متأكد؟")
            confirm = st.checkbox("نعم، أنا متأكد")
            if confirm:
                # كود حذف البيانات المؤقتة
                st.info("في التطبيق الحقيقي، سيتم حذف البيانات المؤقتة هنا")
    
    # معلومات النظام
    st.markdown("### ℹ️ معلومات النظام")
    
    sys_info = {
        "إصدار التطبيق": "1.0.0",
        "عدد المقالات": st.session_state.db.get_statistics()['total_articles'],
        "حالة نموذج AI": "محمّل" if st.session_state.analyzer.model else "غير محمّل",
        "تاريخ التحديث": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for key, value in sys_info.items():
        st.text(f"{key}: {value}")

# ============================================
# 5. دوال مساعدة للواجهة
# ============================================

def get_text_input() -> Optional[Dict]:
    """الحصول على نص من المستخدم"""
    
    st.markdown("#### 📝 أدخل نص الخبر")
    
    title = st.text_input("عنوان الخبر", placeholder="أدخل عنوان الخبر هنا...")
    
    content = st.text_area(
        "نص الخبر",
        height=200,
        placeholder="الصق أو اكتب نص الخبر هنا..."
    )
    
    source = st.text_input("المصدر (اختياري)", placeholder="اسم الموقع أو الصحيفة")
    
    if st.button("تحليل النص", type="primary") and content:
        return {
            'title': title if title else "خبر بدون عنوان",
            'content': content,
            'source_domain': source if source else "مدخل يدوي",
            'author': "مستخدم",
            'published_date': datetime.now(),
            'collection_source': 'manual_text'
        }
    
    return None

def get_url_input() -> Optional[Dict]:
    """الحصول على رابط من المستخدم"""
    
    st.markdown("#### 🔗 أدخل رابط الخبر")
    
    url = st.text_input(
        "رابط الخبر",
        placeholder="https://example.com/news/..."
    )
    
    if st.button("جمع وتحليل", type="primary") and url:
        with st.spinner("جاري جمع المحتوى من الرابط..."):
            article_data = st.session_state.collector.extract_from_url(url)
            
            if article_data:
                return article_data
            else:
                st.error("تعذر استخراج المحتوى من الرابط. يرجى المحاولة يدوياً.")
                return None
    
    return None

def search_database() -> Optional[Dict]:
    """البحث في قاعدة البيانات"""
    
    st.markdown("#### 🔎 بحث في التحليلات السابقة")
    
    search_query = st.text_input("كلمات البحث")
    
    if search_query:
        articles = st.session_state.db.search_articles(query=search_query)
        
        if articles:
            st.markdown(f"**تم العثور على {len(articles)} مقالة**")
            
            # عرض قائمة للمختصر
            article_options = {f"{a['title'][:50]}... (مصداقية: {a.get('credibility_score', 0):.1f})": a for a in articles[:10]}
            
            selected = st.selectbox("اختر مقالة:", list(article_options.keys()))
            
            if selected and st.button("عرض التحليل", type="primary"):
                return article_options[selected]
        else:
            st.info("لا توجد نتائج مطابقة للبحث")
    
    return None

def display_analysis_results(article: Dict, analysis: Dict):
    """عرض نتائج التحليل"""
    
    st.markdown("---")
    st.markdown("## 📊 نتائج التحليل")
    
    # بطاقة النتيجة الرئيسية
    score = analysis['credibility_score']
    category = analysis['category']
    confidence = analysis['confidence_level']
    
    # تحديد لون البطاقة
    if category == 'high':
        css_class = 'credibility-high'
        emoji = "✅"
        label = "موثوق"
    elif category == 'medium':
        css_class = 'credibility-medium'
        emoji = "⚠️"
        label = "معقول"
    elif category == 'low':
        css_class = 'credibility-low'
        emoji = "🚨"
        label = "مشكوك فيه"
    else:
        css_class = 'credibility-fake'
        emoji = "⛔"
        label = "شائعة"
    
    st.markdown(f"""
    <div class="credibility-card {css_class}">
        <h3>{emoji} درجة المصداقية: {score}/100</h3>
        <h4>التصنيف: {label}</h4>
        <p>مستوى الثقة في التحليل: {confidence}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # شريط التقدم
    st.progress(score / 100)
    
    # التوصية
    st.markdown(f"### 💡 التوصية")
    st.info(analysis['explanations']['recommendation'])
    
    # التفاصيل التقنية
    with st.expander("📈 التفاصيل التقنية للتحليل"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### نتائج المكونات")
            scores = analysis['component_scores']
            
            for key, value in scores.items():
                st.metric(
                    label=get_score_label(key),
                    value=f"{value:.1f}/100"
                )
        
        with col2:
            st.markdown("#### تفسيرات التحليل")
            explanations = analysis['explanations']
            
            st.write(f"**ملخص:** {explanations['summary']}")
            st.write(f"**تقييم المصدر:** {explanations.get('source_evaluation', 'غير متوفر')}")
            st.write(f"**تقييم المحتوى:** {explanations.get('content_evaluation', 'غير متوفر')}")
        
        # عرض النسب المئوية
        st.markdown("#### الأوزان المستخدمة")
        weights = {
            "موثوقية المصدر": 35,
            "تحليل المحتوى": 30,
            "أسلوب الكتابة": 20,
            "التحقق الآلي": 15
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=.3,
            textinfo='label+percent'
        )])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # عرض نص المقال
    with st.expander("📄 عرض نص المقال"):
        st.markdown(f"**العنوان:** {article.get('title', 'بدون عنوان')}")
        st.markdown(f"**المصدر:** {article.get('source_domain', 'غير معروف')}")
        st.markdown("**المحتوى:**")
        st.write(article.get('content', ''))

def display_article_card(article: Dict, show_analysis: bool = True):
    """عرض بطاقة مقال"""
    
    title = article.get('title', 'بدون عنوان')
    source = article.get('source_domain', 'غير معروف')
    date = article.get('collected_date', '')
    score = article.get('credibility_score', 0)
    category = article.get('credibility_category', 'unknown')
    
    # تحديد لون البطاقة
    if category == 'high':
        border_color = "#28a745"
        badge_color = "success"
    elif category == 'medium':
        border_color = "#ffc107"
        badge_color = "warning"
    elif category == 'low':
        border_color = "#dc3545"
        badge_color = "danger"
    else:
        border_color = "#721c24"
        badge_color = "dark"
    
    st.markdown(f"""
    <div class="article-card" style="border-left: 5px solid {border_color};">
        <h4>{title[:100]}{'...' if len(title) > 100 else ''}</h4>
        <p style="color: #666; font-size: 14px;">
            <strong>المصدر:</strong> {source} | 
            <strong>التاريخ:</strong> {str(date)[:10]}
        </p>
    """, unsafe_allow_html=True)
    
    if show_analysis and score > 0:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.progress(score / 100)
        
        with col2:
            st.metric("المصداقية", f"{score:.1f}")
        
        with col3:
            st.button(
                "عرض التفاصيل", 
                key=f"view_{article.get('id')}",
                on_click=lambda a=article: display_article_details(a)
            )
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_article_details(article: Dict):
    """عرض تفاصيل مقال (في صفحة منفصلة أو مودال)"""
    
    # في تطبيق Streamlit حقيقي، قد نستخدم صفحة منفصلة أو مودال
    st.session_state['selected_article'] = article
    st.rerun()

def display_feedback_section(article_id: int):
    """عرض قسم تقييم المستخدم"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vote = st.radio(
            "هل تتفق مع التحليل؟",
            ["👍 أوافق", "👎 لا أوافق", "🤷 محايد"],
            horizontal=True
        )
    
    with col2:
        confidence = st.slider(
            "مدى ثقتك في رأيك",
            min_value=1,
            max_value=5,
            value=3
        )
    
    with col3:
        comment = st.text_input("تعليقك (اختياري)")
    
    if st.button("إرسال التقييم", type="secondary"):
        vote_map = {"👍 أوافق": 1, "👎 لا أوافق": -1, "🤷 محايد": 0}
        
        feedback_data = {
            'vote': vote_map[vote],
            'comment': comment,
            'confidence': confidence
        }
        
        success = st.session_state.db.save_feedback(article_id, feedback_data)
        
        if success:
            st.success("شكراً لتقييمك! تم حفظ رأيك.")
            
            # تحديث درجة المصداقية بناءً على التقييمات
            update_credibility_based_on_feedback(article_id)
        else:
            st.error("حدث خطأ في حفظ التقييم.")

def update_credibility_based_on_feedback(article_id: int):
    """تحديث درجة المصداقية بناءً على تقييمات المستخدمين"""
    # هذا دالة تجميلية - في التطبيق الحقيقي سيتم حساب متوسط التقييمات
    pass

def get_score_label(key: str) -> str:
    """تحويل مفتاح النتيجة إلى تسمية عربية"""
    labels = {
        'source_score': 'موثوقية المصدر',
        'content_score': 'تحليل المحتوى',
        'style_score': 'أسلوب الكتابة',
        'verification_score': 'التحقق الآلي'
    }
    return labels.get(key, key)

# ============================================
# 6. تشغيل التطبيق
# ============================================

if __name__ == "__main__":
    # تشغيل التطبيق الرئيسي
    main()
    
    # تذييل الصفحة
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px;">
        <p>تم تطويره باستخدام ❤️ و Streamlit</p>
        <p>TruthScope v1.0 | محرك اكتشاف الحقيقة</p>
        <p>⚠️ هذا التطبيق للأغراض التعليمية والبحثية</p>
    </div>
    """, unsafe_allow_html=True)