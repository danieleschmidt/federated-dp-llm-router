"""
Internationalization (I18n) Manager for Global Federated DP-LLM Router
Provides comprehensive multi-language support with cultural adaptation.
"""

import json
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import locale


class Language(Enum):
    """Supported languages with ISO codes"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    RUSSIAN = "ru"
    HINDI = "hi"
    THAI = "th"
    MALAY = "ms"
    DUTCH = "nl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"


class TextDirection(Enum):
    """Text direction for language support"""
    LTR = "ltr"  # Left to right
    RTL = "rtl"  # Right to left


@dataclass
class LanguageConfig:
    """Configuration for a specific language"""
    language_code: str
    language_name: str
    native_name: str
    text_direction: TextDirection
    date_format: str
    time_format: str
    number_format: str
    currency_symbol: str = ""
    decimal_separator: str = "."
    thousands_separator: str = ","
    regions: List[str] = field(default_factory=list)
    fallback_languages: List[str] = field(default_factory=list)


@dataclass
class CulturalContext:
    """Cultural context for language adaptation"""
    language_code: str
    formal_address: bool = True
    privacy_sensitivity: str = "medium"  # low, medium, high
    data_protection_awareness: str = "medium"
    healthcare_terminology_preference: str = "standard"  # simple, standard, technical
    cultural_considerations: List[str] = field(default_factory=list)


@dataclass
class TranslationRequest:
    """Request for text translation"""
    text: str
    source_language: str
    target_language: str
    context: str = "general"
    formality_level: str = "neutral"  # formal, neutral, informal
    domain: str = "healthcare"
    preserve_formatting: bool = True


class I18nManager:
    """
    Advanced internationalization manager with cultural adaptation,
    context-aware translations, and privacy-sensitive messaging.
    """
    
    def __init__(self):
        self.languages: Dict[str, LanguageConfig] = {}
        self.cultural_contexts: Dict[str, CulturalContext] = {}
        self.translations: Dict[str, Dict[str, str]] = {}
        self.message_templates: Dict[str, Dict[str, str]] = {}
        
        self._initialize_languages()
        self._initialize_cultural_contexts()
        self._initialize_message_templates()
    
    def _initialize_languages(self):
        """Initialize supported languages with their configurations"""
        language_configs = [
            LanguageConfig("en", "English", "English", TextDirection.LTR, "%m/%d/%Y", "%I:%M %p", "#,##0.00", "$", ".", ",", ["US", "CA", "GB", "AU"], []),
            LanguageConfig("es", "Spanish", "Español", TextDirection.LTR, "%d/%m/%Y", "%H:%M", "#.##0,00", "€", ",", ".", ["ES", "MX", "AR", "CO"], ["en"]),
            LanguageConfig("fr", "French", "Français", TextDirection.LTR, "%d/%m/%Y", "%H:%M", "# ##0,00", "€", ",", " ", ["FR", "CA", "BE", "CH"], ["en"]),
            LanguageConfig("de", "German", "Deutsch", TextDirection.LTR, "%d.%m.%Y", "%H:%M", "#.##0,00", "€", ",", ".", ["DE", "AT", "CH"], ["en"]),
            LanguageConfig("it", "Italian", "Italiano", TextDirection.LTR, "%d/%m/%Y", "%H:%M", "#.##0,00", "€", ",", ".", ["IT"], ["en"]),
            LanguageConfig("zh-CN", "Chinese Simplified", "简体中文", TextDirection.LTR, "%Y/%m/%d", "%H:%M", "#,##0.00", "¥", ".", ",", ["CN"], ["en"]),
            LanguageConfig("zh-TW", "Chinese Traditional", "繁體中文", TextDirection.LTR, "%Y/%m/%d", "%H:%M", "#,##0.00", "NT$", ".", ",", ["TW", "HK"], ["zh-CN", "en"]),
            LanguageConfig("ja", "Japanese", "日本語", TextDirection.LTR, "%Y/%m/%d", "%H:%M", "#,##0", "¥", ".", ",", ["JP"], ["en"]),
            LanguageConfig("ko", "Korean", "한국어", TextDirection.LTR, "%Y.%m.%d", "%H:%M", "#,##0", "₩", ".", ",", ["KR"], ["en"]),
            LanguageConfig("ar", "Arabic", "العربية", TextDirection.RTL, "%d/%m/%Y", "%H:%M", "#,##0.00", "", ".", ",", ["SA", "AE", "EG"], ["en"]),
            LanguageConfig("hi", "Hindi", "हिन्दी", TextDirection.LTR, "%d/%m/%Y", "%H:%M", "#,##0.00", "₹", ".", ",", ["IN"], ["en"]),
            LanguageConfig("th", "Thai", "ไทย", TextDirection.LTR, "%d/%m/%Y", "%H:%M", "#,##0.00", "฿", ".", ",", ["TH"], ["en"]),
            LanguageConfig("ms", "Malay", "Bahasa Melayu", TextDirection.LTR, "%d/%m/%Y", "%H:%M", "#,##0.00", "RM", ".", ",", ["MY", "SG"], ["en"])
        ]
        
        for config in language_configs:
            self.languages[config.language_code] = config
    
    def _initialize_cultural_contexts(self):
        """Initialize cultural contexts for different languages"""
        cultural_contexts = [
            CulturalContext("en", formal_address=False, privacy_sensitivity="medium", data_protection_awareness="high", 
                          healthcare_terminology_preference="standard", cultural_considerations=["Direct communication", "Individual privacy"]),
            CulturalContext("de", formal_address=True, privacy_sensitivity="high", data_protection_awareness="very_high", 
                          healthcare_terminology_preference="technical", cultural_considerations=["Formal address", "GDPR compliance", "Precision"]),
            CulturalContext("fr", formal_address=True, privacy_sensitivity="high", data_protection_awareness="high", 
                          healthcare_terminology_preference="standard", cultural_considerations=["Formal politeness", "Cultural sensitivity"]),
            CulturalContext("ja", formal_address=True, privacy_sensitivity="very_high", data_protection_awareness="medium", 
                          healthcare_terminology_preference="simple", cultural_considerations=["Respectful language", "Indirect communication"]),
            CulturalContext("zh-CN", formal_address=True, privacy_sensitivity="medium", data_protection_awareness="medium", 
                          healthcare_terminology_preference="standard", cultural_considerations=["Hierarchical respect", "Group harmony"]),
            CulturalContext("ar", formal_address=True, privacy_sensitivity="very_high", data_protection_awareness="medium", 
                          healthcare_terminology_preference="simple", cultural_considerations=["Religious sensitivity", "Family privacy"]),
            CulturalContext("es", formal_address=True, privacy_sensitivity="medium", data_protection_awareness="medium", 
                          healthcare_terminology_preference="standard", cultural_considerations=["Regional variations", "Warm communication"])
        ]
        
        for context in cultural_contexts:
            self.cultural_contexts[context.language_code] = context
    
    def _initialize_message_templates(self):
        """Initialize message templates for different contexts"""
        self.message_templates = {
            "privacy_notice": {
                "en": "Your privacy is protected. This system uses differential privacy to ensure your data remains secure.",
                "es": "Su privacidad está protegida. Este sistema utiliza privacidad diferencial para garantizar que sus datos permanezcan seguros.",
                "fr": "Votre vie privée est protégée. Ce système utilise la confidentialité différentielle pour garantir la sécurité de vos données.",
                "de": "Ihre Privatsphäre ist geschützt. Dieses System verwendet differenzielle Privatsphäre, um die Sicherheit Ihrer Daten zu gewährleisten.",
                "zh-CN": "您的隐私受到保护。该系统使用差分隐私来确保您的数据安全。",
                "ja": "お客様のプライバシーは保護されています。このシステムは差分プライバシーを使用してデータの安全性を確保します。",
                "ar": "خصوصيتك محمية. يستخدم هذا النظام الخصوصية التفاضلية لضمان أمان بياناتك."
            },
            "error_message": {
                "en": "We encountered an issue processing your request. Please try again or contact support.",
                "es": "Encontramos un problema al procesar su solicitud. Inténtelo de nuevo o contacte al soporte.",
                "fr": "Nous avons rencontré un problème lors du traitement de votre demande. Veuillez réessayer ou contacter le support.",
                "de": "Bei der Bearbeitung Ihrer Anfrage ist ein Problem aufgetreten. Bitte versuchen Sie es erneut oder wenden Sie sich an den Support.",
                "zh-CN": "处理您的请求时遇到问题。请重试或联系支持团队。",
                "ja": "リクエストの処理中に問題が発生しました。もう一度お試しいただくか、サポートにお問い合わせください。",
                "ar": "واجهنا مشكلة في معالجة طلبك. يرجى المحاولة مرة أخرى أو الاتصال بالدعم."
            },
            "consent_request": {
                "en": "Do you consent to processing your data for healthcare analytics while maintaining privacy protection?",
                "es": "¿Consiente el procesamiento de sus datos para análisis de atención médica manteniendo la protección de la privacidad?",
                "fr": "Consentez-vous au traitement de vos données pour l'analyse des soins de santé tout en maintenant la protection de la vie privée?",
                "de": "Stimmen Sie der Verarbeitung Ihrer Daten für Gesundheitsanalysen unter Wahrung des Datenschutzes zu?",
                "zh-CN": "您是否同意在保持隐私保护的情况下处理您的数据用于医疗分析？",
                "ja": "プライバシー保護を維持しながら、医療分析のためのデータ処理に同意されますか？",
                "ar": "هل توافق على معالجة بياناتك لتحليلات الرعاية الصحية مع الحفاظ على حماية الخصوصية؟"
            },
            "processing_status": {
                "en": "Processing your request securely...",
                "es": "Procesando su solicitud de forma segura...",
                "fr": "Traitement sécurisé de votre demande en cours...",
                "de": "Sichere Bearbeitung Ihrer Anfrage...",
                "zh-CN": "正在安全处理您的请求...",
                "ja": "リクエストを安全に処理中...",
                "ar": "معالجة طلبك بأمان..."
            }
        }
    
    async def translate_text(self, request: TranslationRequest) -> str:
        """
        Translate text with cultural and contextual awareness
        """
        # Check if we have a direct translation
        direct_translation = self._get_direct_translation(request)
        if direct_translation:
            return direct_translation
        
        # Apply cultural adaptations
        adapted_text = self._apply_cultural_adaptations(request)
        
        # Apply domain-specific terminology
        domain_adapted = self._apply_domain_terminology(adapted_text, request)
        
        # Apply formality adjustments
        formality_adapted = self._apply_formality(domain_adapted, request)
        
        return formality_adapted
    
    def _get_direct_translation(self, request: TranslationRequest) -> Optional[str]:
        """Get direct translation from templates if available"""
        # Check message templates first
        for template_key, translations in self.message_templates.items():
            if request.text.lower() in translations.get(request.source_language, "").lower():
                return translations.get(request.target_language, request.text)
        
        return None
    
    def _apply_cultural_adaptations(self, request: TranslationRequest) -> str:
        """Apply cultural adaptations based on target language"""
        text = request.text
        target_context = self.cultural_contexts.get(request.target_language)
        
        if not target_context:
            return text
        
        # Apply formal address if required
        if target_context.formal_address and request.formality_level != "informal":
            text = self._make_formal(text, request.target_language)
        
        # Apply privacy sensitivity adaptations
        if target_context.privacy_sensitivity == "very_high":
            text = self._enhance_privacy_language(text, request.target_language)
        
        # Apply healthcare terminology preferences
        if request.domain == "healthcare":
            text = self._adapt_healthcare_terminology(text, target_context.healthcare_terminology_preference, request.target_language)
        
        return text
    
    def _make_formal(self, text: str, language: str) -> str:
        """Make text more formal based on language conventions"""
        formal_patterns = {
            "de": {
                "you": "Sie",
                "your": "Ihr",
                "please": "bitte"
            },
            "fr": {
                "you": "vous",
                "your": "votre"
            },
            "es": {
                "you": "usted",
                "your": "su"
            },
            "ja": {
                # Add honorific patterns
                "です": "でございます",
                "ます": "ございます"
            }
        }
        
        patterns = formal_patterns.get(language, {})
        for informal, formal in patterns.items():
            text = re.sub(rf'\b{informal}\b', formal, text, flags=re.IGNORECASE)
        
        return text
    
    def _enhance_privacy_language(self, text: str, language: str) -> str:
        """Enhance privacy-related language for sensitive cultures"""
        privacy_enhancements = {
            "de": {
                "data": "Ihre vertraulichen Daten",
                "information": "Ihre persönlichen Informationen",
                "secure": "höchst sicher"
            },
            "ja": {
                "data": "お客様の大切なデータ",
                "privacy": "プライバシーの厳重な保護",
                "secure": "最高レベルのセキュリティ"
            },
            "ar": {
                "data": "بياناتك الشخصية المحمية",
                "privacy": "الخصوصية المضمونة",
                "secure": "آمن تماماً"
            }
        }
        
        enhancements = privacy_enhancements.get(language, {})
        for basic, enhanced in enhancements.items():
            text = re.sub(rf'\b{basic}\b', enhanced, text, flags=re.IGNORECASE)
        
        return text
    
    def _adapt_healthcare_terminology(self, text: str, preference: str, language: str) -> str:
        """Adapt healthcare terminology based on preference"""
        if preference == "simple":
            text = self._simplify_medical_terms(text, language)
        elif preference == "technical":
            text = self._use_technical_terms(text, language)
        
        return text
    
    def _simplify_medical_terms(self, text: str, language: str) -> str:
        """Simplify medical terminology for better understanding"""
        simplifications = {
            "en": {
                "differential privacy": "data protection",
                "federated learning": "secure data sharing",
                "machine learning": "computer analysis"
            },
            "es": {
                "privacidad diferencial": "protección de datos",
                "aprendizaje federado": "intercambio seguro de datos"
            },
            "fr": {
                "confidentialité différentielle": "protection des données",
                "apprentissage fédéré": "partage sécurisé des données"
            }
        }
        
        terms = simplifications.get(language, {})
        for technical, simple in terms.items():
            text = re.sub(technical, simple, text, flags=re.IGNORECASE)
        
        return text
    
    def _use_technical_terms(self, text: str, language: str) -> str:
        """Use more technical terminology when appropriate"""
        # Implementation for technical terminology enhancement
        return text
    
    def _apply_domain_terminology(self, text: str, request: TranslationRequest) -> str:
        """Apply domain-specific terminology"""
        # Healthcare-specific term adaptations
        if request.domain == "healthcare":
            return self._apply_healthcare_terminology_adaptations(text, request.target_language)
        
        return text
    
    def _apply_healthcare_terminology_adaptations(self, text: str, language: str) -> str:
        """Apply healthcare-specific terminology adaptations"""
        healthcare_terms = {
            "en": {
                "patient": "patient",
                "medical record": "medical record",
                "diagnosis": "diagnosis"
            },
            "es": {
                "patient": "paciente",
                "medical record": "historial médico",
                "diagnosis": "diagnóstico"
            },
            "fr": {
                "patient": "patient",
                "medical record": "dossier médical",
                "diagnosis": "diagnostic"
            }
        }
        
        # Apply healthcare terminology
        return text
    
    def _apply_formality(self, text: str, request: TranslationRequest) -> str:
        """Apply formality level adjustments"""
        if request.formality_level == "formal":
            return self._make_formal(text, request.target_language)
        elif request.formality_level == "informal":
            return self._make_informal(text, request.target_language)
        
        return text
    
    def _make_informal(self, text: str, language: str) -> str:
        """Make text more informal"""
        # Implementation for informal language patterns
        return text
    
    async def get_localized_message(self, message_key: str, language: str, 
                                  context: Dict[str, Any] = None) -> str:
        """Get localized message with context substitution"""
        if message_key in self.message_templates:
            template = self.message_templates[message_key].get(language)
            if template:
                if context:
                    # Simple context substitution
                    for key, value in context.items():
                        template = template.replace(f"{{{key}}}", str(value))
                return template
        
        # Fallback to English if available
        return self.message_templates.get(message_key, {}).get("en", f"Message not found: {message_key}")
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of input text
        Returns (language_code, confidence)
        """
        # Simple language detection based on character patterns
        # In production, would use a proper language detection library
        
        # Check for specific character sets
        if re.search(r'[\u4e00-\u9fff]', text):  # Chinese characters
            if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):  # Japanese hiragana/katakana
                return ("ja", 0.9)
            elif len(re.findall(r'[\u4e00-\u9fff]', text)) > len(text) * 0.3:
                return ("zh-CN", 0.8)
        
        if re.search(r'[\u0600-\u06ff]', text):  # Arabic
            return ("ar", 0.9)
        
        if re.search(r'[\u0900-\u097f]', text):  # Hindi
            return ("hi", 0.9)
        
        if re.search(r'[\u0e00-\u0e7f]', text):  # Thai
            return ("th", 0.9)
        
        # European languages - simple keyword detection
        german_words = ['der', 'die', 'das', 'und', 'ich', 'sie', 'ist', 'werden']
        french_words = ['le', 'la', 'les', 'et', 'je', 'il', 'elle', 'être']
        spanish_words = ['el', 'la', 'los', 'las', 'y', 'yo', 'él', 'ella', 'ser']
        
        text_lower = text.lower()
        
        german_matches = sum(1 for word in german_words if word in text_lower)
        french_matches = sum(1 for word in french_words if word in text_lower)
        spanish_matches = sum(1 for word in spanish_words if word in text_lower)
        
        if german_matches > 0:
            return ("de", 0.7)
        elif french_matches > 0:
            return ("fr", 0.7)
        elif spanish_matches > 0:
            return ("es", 0.7)
        
        # Default to English
        return ("en", 0.6)
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Get list of supported languages with metadata"""
        return [
            {
                "code": lang.language_code,
                "name": lang.language_name,
                "native_name": lang.native_name,
                "text_direction": lang.text_direction.value,
                "regions": lang.regions,
                "has_cultural_context": lang.language_code in self.cultural_contexts
            }
            for lang in self.languages.values()
        ]
    
    def get_regional_languages(self, region_code: str) -> List[str]:
        """Get languages supported in a specific region"""
        regional_languages = []
        
        for lang in self.languages.values():
            if region_code.upper() in lang.regions:
                regional_languages.append(lang.language_code)
        
        return regional_languages
    
    def format_number(self, number: float, language: str) -> str:
        """Format number according to language conventions"""
        lang_config = self.languages.get(language)
        if not lang_config:
            return str(number)
        
        # Simple number formatting
        decimal_places = 2 if '.' in str(number) else 0
        formatted = f"{number:,.{decimal_places}f}"
        
        # Apply language-specific separators
        if lang_config.decimal_separator != ".":
            formatted = formatted.replace(".", lang_config.decimal_separator)
        if lang_config.thousands_separator != ",":
            formatted = formatted.replace(",", lang_config.thousands_separator)
        
        return formatted
    
    def format_datetime(self, dt: datetime, language: str) -> str:
        """Format datetime according to language conventions"""
        lang_config = self.languages.get(language)
        if not lang_config:
            return dt.isoformat()
        
        try:
            return dt.strftime(f"{lang_config.date_format} {lang_config.time_format}")
        except:
            return dt.isoformat()
    
    def get_language_statistics(self) -> Dict[str, Any]:
        """Get comprehensive language support statistics"""
        return {
            "total_languages": len(self.languages),
            "rtl_languages": len([l for l in self.languages.values() if l.text_direction == TextDirection.RTL]),
            "ltr_languages": len([l for l in self.languages.values() if l.text_direction == TextDirection.LTR]),
            "regions_covered": len(set().union(*(l.regions for l in self.languages.values()))),
            "message_templates": len(self.message_templates),
            "cultural_contexts": len(self.cultural_contexts),
            "languages_by_region": {
                region: self.get_regional_languages(region)
                for region in set().union(*(l.regions for l in self.languages.values()))
            }
        }


# Global instance
global_i18n_manager = I18nManager()