"""
Indonesian Financial Statement Analyzer using LLM and NLP
========================================================

A comprehensive tool for analyzing Indonesian stock company financial statements using
Large Language Models and Natural Language Processing techniques.

Features:
- PDF text extraction using multiple methods (pdfplumber, PyMuPDF)
- Indonesian Rupiah (IDR) currency support
- Indonesian accounting standards (PSAK) compliance
- LLM-powered intelligent data interpretation with Indonesian language support (Qwen 2.5VL 72B via OpenRouter)
- Automatic financial data categorization for Indonesian companies
- Validation and sanity checks for Indonesian financial statements
- Structured output in JSON/tabular format
- Edge case handling and error detection
- Web interface with file upload functionality

Designed for: Indonesian Public Companies (Tbk) Financial Statements
Author: Kevin Cliff Gunawan
Date: 21 October 2025
"""

import json
import re
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# PDF processing libraries
try:
    import pdfplumber
    import fitz  # PyMuPDF
except ImportError as e:
    print(f"Warning: PDF processing libraries not installed. {e}")

# LLM integration (using OpenRouter API via OpenAI SDK)
from openai import OpenAI

# NLP libraries
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    print("Warning: NLTK not installed. Install with: pip install nltk")

# Numerical processing
import numpy as np
from decimal import Decimal, InvalidOperation


@dataclass
class FinancialItem:
    """Data class for individual financial items"""
    name: str
    value: float
    category: str
    subcategory: str
    confidence_score: float
    source_text: str
    page_number: int
    validation_status: str = "pending"
    notes: str = ""


@dataclass
class ValidationResult:
    """Data class for validation results"""
    is_valid: bool
    confidence_score: float
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class FinancialStatementAnalyzer:
    """
    Main class for analyzing financial statements using LLM and NLP
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "openai/gpt-4o-mini", max_pages: int = 50):
        """
        Initialize the Financial Statement Analyzer
        
        Args:
            openai_api_key: OpenRouter API key for LLM integration
            model: LLM model to use for analysis (default: GPT-4o-mini for reliability)
            max_pages: Maximum pages to process (default: 50, set to None for all)
        """
        self.openai_api_key = openai_api_key or "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"
        self.model = model
        self.max_pages = max_pages
        self.extracted_data = []
        self.validation_results = {}
        self.use_regex_fallback = False  # Disable regex fallback for academic requirements
        
        # Initialize OpenAI client for OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openai_api_key
        )
        
        # Setup logging
        self._setup_logging()
        
        # Indonesian Financial categories mapping (PSAK compliant)
        self.financial_categories = {
            'aset_lancar': [
                'kas', 'kas dan setara kas', 'cash and cash equivalents', 'investasi jangka pendek',
                'piutang usaha', 'piutang', 'persediaan', 'inventory', 'beban dibayar dimuka',
                'aset lancar', 'current assets', 'deposito berjangka', 'investasi pada instrumen ekuitas',
                'investasi pada obligasi', 'pajak dibayar dimuka'
            ],
            'aset_tidak_lancar': [
                'aset tetap', 'property plant equipment', 'ppe', 'investasi jangka panjang',
                'aset tidak berwujud', 'intangible assets', 'goodwill', 'aset pajak tangguhan',
                'aset tidak lancar', 'non-current assets', 'fixed assets', 'aset hak guna usaha',
                'aset takberwujud', 'investasi pada entitas asosiasi', 'uang muka perolehan aset tetap'
            ],
            'liabilitas_jangka_pendek': [
                'utang usaha', 'utang bank jangka pendek', 'beban akrual', 'utang pajak',
                'liabilitas jangka pendek', 'current liabilities', 'utang dividen',
                'liabilitas imbalan kerja jangka pendek', 'utang lain-lain', 'liabilitas sewa'
            ],
            'liabilitas_jangka_panjang': [
                'utang bank jangka panjang', 'liabilitas pajak tangguhan', 'liabilitas imbalan kerja',
                'liabilitas jangka panjang', 'non-current liabilities', 'utang obligasi',
                'liabilitas sewa', 'liabilitas pajak tangguhan'
            ],
            'ekuitas': [
                'ekuitas', 'modal saham', 'saham biasa', 'saham preferen', 'tambahan modal disetor',
                'saldo laba', 'retained earnings', 'equity', 'ekuitas yang dapat diatribusikan',
                'kepentingan nonpengendali', 'komponen ekuitas lain', 'surplus revaluasi aset tetap'
            ],
            'pendapatan': [
                'penjualan neto', 'pendapatan', 'revenue', 'penjualan', 'pendapatan usaha',
                'pendapatan operasi', 'net sales', 'total pendapatan', 'laba bruto'
            ],
            'beban': [
                'beban pokok penjualan', 'cost of goods sold', 'beban usaha', 'beban operasi',
                'beban penjualan', 'beban administrasi', 'beban penyusutan', 'beban amortisasi',
                'beban bunga', 'biaya keuangan'
            ],
            'pendapatan_lain': [
                'pendapatan lain-lain', 'pendapatan non operasi', 'pendapatan investasi',
                'keuntungan penjualan aset', 'bagian atas laba rugi neto entitas asosiasi',
                'pendapatan keuangan', 'laba sebelum pajak penghasilan'
            ]
        }
        
        # Regex patterns for Indonesian financial figures
        self.financial_patterns = [
            r'Rp\s*[\d.,]+',  # Indonesian Rupiah amounts
            r'[\d.,]+\.[\d]{3}\.[\d]{3}',  # Indonesian number format (e.g., 123.456.789)
            r'[\d.,]+\.[\d]{3}',  # Indonesian thousands format (e.g., 123.456)
            r'[\d,]+\.?\d*\s*juta',  # Millions in Indonesian
            r'[\d,]+\.?\d*\s*miliar',  # Billions in Indonesian
            r'[\d,]+\.?\d*\s*million',  # Millions in English
            r'[\d,]+\.?\d*\s*billion',  # Billions in English
            r'\(\s*[\d.,]+\s*\)',  # Negative amounts in parentheses
            r'[\d.,]+\s*%',  # Percentages
            r'[\d]{1,3}(?:\.[\d]{3})*(?:,[\d]{2})?',  # Indonesian decimal format
        ]
        
        self.logger.info("Financial Statement Analyzer initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('financial_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using multiple methods for robustness
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        self.logger.info(f"Starting text extraction from: {pdf_path}")
        
        extracted_data = {
            'text_by_page': {},
            'tables': [],
            'metadata': {},
            'extraction_method': 'hybrid'
        }
        
        try:
            # Method 1: pdfplumber (better for tables and structured data)
            with pdfplumber.open(pdf_path) as pdf:
                extracted_data['metadata'] = {
                    'total_pages': len(pdf.pages),
                    'title': pdf.metadata.get('Title', ''),
                    'author': pdf.metadata.get('Author', ''),
                    'creation_date': pdf.metadata.get('CreationDate', '')
                }
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        extracted_data['text_by_page'][page_num] = text
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            extracted_data['tables'].append({
                                'page': page_num,
                                'data': table
                            })
                
                self.logger.info(f"pdfplumber extraction completed: {len(extracted_data['text_by_page'])} pages")
        
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
            
            # Fallback Method 2: PyMuPDF
            try:
                doc = fitz.open(pdf_path)
                extracted_data['metadata']['total_pages'] = len(doc)
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text:
                        extracted_data['text_by_page'][page_num + 1] = text
                
                doc.close()
                extracted_data['extraction_method'] = 'pymupdf_fallback'
                self.logger.info(f"PyMuPDF fallback extraction completed: {len(extracted_data['text_by_page'])} pages")
                
            except Exception as e2:
                self.logger.error(f"Both extraction methods failed: {e2}")
                raise Exception(f"PDF extraction failed: {e}, {e2}")
        
        return extracted_data
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess extracted text for better LLM analysis
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{4}', '', text)  # Remove dates in headers
        
        # Normalize financial notation
        text = re.sub(r'\$\s+', '$', text)  # Remove space after dollar sign
        text = re.sub(r'\(\s*(\d)', r'(\1', text)  # Remove space in negative numbers
        
        return text.strip()
    
    def extract_financial_figures_regex(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract financial figures using regex patterns
        
        Args:
            text: Text to extract figures from
            
        Returns:
            List of extracted financial figures with context
        """
        figures = []
        seen_positions = set()
        
        for pattern in self.financial_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Avoid duplicate extractions at same position
                if match.start() in seen_positions:
                    continue
                seen_positions.add(match.start())
                
                # Get context around the match (more context for better name extraction)
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Try to extract item name from text before the value
                before_text = text[start:match.start()].strip()
                
                # Look for the last line or sentence before the number
                lines = before_text.split('\n')
                item_name = lines[-1].strip() if lines else ""
                
                # If name is too short or too long, try to find a better one
                if len(item_name) < 3 or len(item_name) > 100:
                    # Try to find words just before the number
                    words_before = before_text.split()
                    item_name = ' '.join(words_before[-5:]) if len(words_before) >= 5 else ' '.join(words_before[-3:])
                
                figures.append({
                    'value': match.group(),
                    'context': context,
                    'position': match.span(),
                    'pattern_used': pattern,
                    'item_name': item_name.strip()
                })
        
        return figures
    
    def query_llm(self, prompt: str, max_tokens: int = 1000, retry_count: int = 0) -> str:
        """
        Query the LLM for intelligent text interpretation using OpenRouter API via OpenAI SDK
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in response
            retry_count: Current retry attempt (for exponential backoff)
            
        Returns:
            LLM response text
        """
        if not self.openai_api_key:
            self.logger.warning("No OpenRouter API key provided, using mock response")
            return "Mock LLM response - Please provide API key for actual analysis"
        
        try:
            self.logger.debug(f"Sending request to OpenRouter with model: {self.model}")
            
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/indonesian-financial-analyzer",
                    "X-Title": "Indonesian Financial Analyzer",
                },
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a financial analyst expert at extracting and interpreting financial data from Indonesian financial documents. Always respond with valid JSON when requested. Never include markdown formatting or explanations."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent results
                timeout=30.0  # Add timeout to prevent hanging
            )
            
            content = completion.choices[0].message.content
            self.logger.debug(f"LLM response received: {len(content)} characters")
            return content
                
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"LLM query failed: {error_msg}")
            
            # Check for rate limit error
            if "429" in error_msg or "rate limit" in error_msg.lower():
                self.logger.warning("Rate limit hit - falling back to regex extraction")
                return "[]"  # Return empty array to trigger regex fallback
            
            # For other errors, also fall back gracefully
            self.logger.error(f"LLM error: {error_msg}")
            return "[]"
    
    def categorize_financial_item(self, item_name: str, context: str = "") -> Tuple[str, str, float]:
        """
        Categorize financial items using rule-based NLP logic (NO LLM calls for speed)
        
        Args:
            item_name: Name of the financial item
            context: Surrounding context for better categorization
            
        Returns:
            Tuple of (category, subcategory, confidence_score)
        """
        item_name_lower = item_name.lower()
        context_lower = context.lower()
        combined_text = f"{item_name_lower} {context_lower}"
        
        best_category = "other"
        best_subcategory = "uncategorized"
        best_score = 0.0
        
        # Enhanced rule-based categorization with context
        for category, keywords in self.financial_categories.items():
            for keyword in keywords:
                # Check in both item name and context
                name_match = keyword in item_name_lower
                context_match = keyword in context_lower
                
                if name_match or context_match:
                    # Calculate score based on match quality
                    if name_match:
                        score = len(keyword) / max(len(item_name_lower), 1)
                        score *= 1.2  # Boost for name match
                    else:
                        score = len(keyword) / max(len(context_lower), 1)
                        score *= 0.8  # Lower score for context-only match
                    
                    # Boost score for longer keyword matches
                    if len(keyword) > 10:
                        score *= 1.1
                    
                    if score > best_score:
                        best_score = score
                        best_category = category
                        best_subcategory = keyword
        
        # If still no good match, try partial matching
        if best_score < 0.3:
            for category, keywords in self.financial_categories.items():
                for keyword in keywords:
                    keyword_words = keyword.split()
                    if any(word in combined_text for word in keyword_words if len(word) > 3):
                        score = 0.4  # Moderate confidence for partial match
                        if score > best_score:
                            best_score = score
                            best_category = category
                            best_subcategory = keyword
                            break
        
        return best_category, best_subcategory, min(best_score, 1.0)
    
    def parse_financial_value(self, value_str: str) -> Tuple[float, str]:
        """
        Parse Indonesian financial value strings into numerical values
        
        Args:
            value_str: String representation of financial value (Indonesian format)
            
        Returns:
            Tuple of (numerical_value, notes)
        """
        notes = ""
        original_value = value_str
        
        # Clean the value string but preserve Indonesian formatting
        value_str = re.sub(r'[^\d.,()-]', '', value_str)
        
        # Handle negative values in parentheses
        is_negative = False
        if '(' in original_value and ')' in original_value:
            is_negative = True
            value_str = value_str.replace('(', '').replace(')', '')
            notes += "Nilai negatif (dalam kurung); "
        
        # Handle Indonesian multipliers
        multiplier = 1
        original_lower = original_value.lower()
        if 'miliar' in original_lower or 'billion' in original_lower:
            multiplier = 1_000_000_000
            notes += "Nilai dalam miliar; "
        elif 'juta' in original_lower or 'million' in original_lower:
            multiplier = 1_000_000
            notes += "Nilai dalam juta; "
        elif 'ribu' in original_lower or 'thousand' in original_lower:
            multiplier = 1_000
            notes += "Nilai dalam ribu; "
        
        # Parse the numerical value (Indonesian format uses . as thousands separator, , as decimal)
        try:
            # Handle Indonesian number format (e.g., 123.456.789,50)
            if ',' in value_str and '.' in value_str:
                # Split by comma to separate decimal part
                parts = value_str.split(',')
                if len(parts) == 2:
                    # Remove dots from integer part (thousands separators)
                    integer_part = parts[0].replace('.', '')
                    decimal_part = parts[1]
                    value_str = f"{integer_part}.{decimal_part}"
                else:
                    # Multiple commas, treat as error
                    value_str = value_str.replace('.', '').replace(',', '.')
            elif '.' in value_str and not ',' in value_str:
                # Check if it's thousands separator or decimal point
                dot_parts = value_str.split('.')
                if len(dot_parts) > 2:
                    # Multiple dots = thousands separators
                    value_str = value_str.replace('.', '')
                elif len(dot_parts) == 2 and len(dot_parts[1]) == 3:
                    # Likely thousands separator (e.g., 123.456)
                    value_str = value_str.replace('.', '')
                # else: treat as decimal point
            elif ',' in value_str and not '.' in value_str:
                # Comma as decimal separator
                value_str = value_str.replace(',', '.')
            
            numerical_value = float(value_str) * multiplier
            
            if is_negative:
                numerical_value = -numerical_value
                
            return numerical_value, notes.strip()
            
        except (ValueError, InvalidOperation):
            self.logger.warning(f"Tidak dapat mengurai nilai keuangan: {original_value}")
            return 0.0, f"Error parsing: {original_value}"
    
    def validate_financial_data(self, financial_items: List[FinancialItem]) -> ValidationResult:
        """
        Validate extracted financial data for consistency and accuracy
        Based on Indonesian accounting standards (PSAK)
        
        Args:
            financial_items: List of extracted financial items
            
        Returns:
            ValidationResult object with validation status and details
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Group items by category
        categories = {}
        for item in financial_items:
            if item.category not in categories:
                categories[item.category] = []
            categories[item.category].append(item)
        
        # PSAK-based validation rules
        
        # 1. Check balance sheet equation: Assets = Liabilities + Equity
        total_assets = sum(item.value for item in categories.get('aset_lancar', []) + 
                          categories.get('aset_tidak_lancar', []))
        total_liabilities = sum(item.value for item in categories.get('liabilitas_jangka_pendek', []) + 
                               categories.get('liabilitas_jangka_panjang', []))
        total_equity = sum(item.value for item in categories.get('ekuitas', []))
        
        balance_difference = abs(total_assets - (total_liabilities + total_equity))
        balance_tolerance = max(total_assets * 0.02, 1000000)  # 2% tolerance or Rp 1,000,000
        
        if balance_difference > balance_tolerance:
            errors.append(f"Persamaan neraca tidak seimbang (PSAK 1): Aset (Rp {total_assets:,.0f}) ≠ Liabilitas + Ekuitas (Rp {total_liabilities + total_equity:,.0f})")
        elif balance_difference > balance_tolerance * 0.5:
            warnings.append(f"Persamaan neraca mendekati batas toleransi: selisih Rp {balance_difference:,.0f}")
        
        # 2. Income statement validation (PSAK 1)
        revenues = sum(item.value for item in categories.get('pendapatan', []))
        expenses = sum(abs(item.value) for item in categories.get('beban', []))
        
        if revenues > 0 and expenses == 0:
            warnings.append("Pendapatan ditemukan tanpa beban - periksa kelengkapan laporan laba rugi")
        elif revenues == 0 and expenses > 0:
            warnings.append("Beban ditemukan tanpa pendapatan - periksa kelengkapan laporan laba rugi")
        
        # 3. Check for negative values where they shouldn't be (PSAK standards)
        for category, items in categories.items():
            if category in ['aset_lancar', 'aset_tidak_lancar', 'pendapatan']:
                negative_items = [item for item in items if item.value < 0]
                if negative_items:
                    warnings.append(f"Nilai negatif ditemukan di kategori {category}: {len(negative_items)} item(s) - periksa penyajian")
            
            # Check for unusually large values (potential data entry errors)
            for item in items:
                if abs(item.value) > 1e12:  # > 1 trillion IDR
                    warnings.append(f"Nilai sangat besar ditemukan: {item.name} = Rp {item.value:,.0f}")
        
        # 4. Check confidence scores
        low_confidence_items = [item for item in financial_items if item.confidence_score < 0.6]
        if low_confidence_items:
            warnings.append(f"Ditemukan {len(low_confidence_items)} item dengan confidence score rendah (<60%)")
        
        very_low_confidence = [item for item in financial_items if item.confidence_score < 0.4]
        if very_low_confidence:
            errors.append(f"Ditemukan {len(very_low_confidence)} item dengan confidence score sangat rendah (<40%)")
        
        # 5. Check for missing key categories (PSAK 1 requirements)
        required_categories = ['aset_lancar', 'liabilitas_jangka_pendek', 'ekuitas']
        missing_categories = [cat for cat in required_categories if cat not in categories or len(categories[cat]) == 0]
        if missing_categories:
            errors.append(f"Kategori wajib tidak ditemukan (PSAK 1): {', '.join(missing_categories)}")
        
        # 6. Check for duplicate items
        item_names = [item.name.lower() for item in financial_items]
        duplicates = set([name for name in item_names if item_names.count(name) > 1])
        if duplicates:
            warnings.append(f"Potensi duplikasi item: {', '.join(list(duplicates)[:3])}{'...' if len(duplicates) > 3 else ''}")
        
        # 7. Indonesian-specific validations
        
        # Check for proper asset classification (PSAK 1)
        if 'aset_lancar' in categories and 'aset_tidak_lancar' in categories:
            current_ratio = total_assets / max(total_liabilities, 1) if total_liabilities > 0 else float('inf')
            if current_ratio < 0.5:
                warnings.append("Rasio aset lancar terhadap total liabilitas rendah - periksa likuiditas perusahaan")
        
        # 8. Currency consistency check
        currency_inconsistencies = []
        for item in financial_items:
            if item.notes and 'currency_mismatch' in item.notes:
                currency_inconsistencies.append(item.name)
        
        if currency_inconsistencies:
            warnings.append(f"Inkonsistensi mata uang ditemukan pada: {', '.join(currency_inconsistencies[:3])}{'...' if len(currency_inconsistencies) > 3 else ''}")
        
        # 9. Calculate overall confidence
        if financial_items:
            avg_confidence = sum(item.confidence_score for item in financial_items) / len(financial_items)
            
            # Adjust confidence based on validation results
            if len(errors) > 0:
                avg_confidence *= 0.7  # Reduce confidence if there are errors
            if len(warnings) > 3:
                avg_confidence *= 0.9  # Slightly reduce confidence for many warnings
        else:
            avg_confidence = 0.0
            errors.append("Tidak ada data keuangan yang berhasil diekstrak")
        
        # 10. Generate suggestions
        if len(financial_items) < 10:
            suggestions.append("Jumlah item keuangan yang diekstrak sedikit - pertimbangkan untuk memeriksa kualitas PDF atau menggunakan API key untuk analisis LLM")
        
        if avg_confidence < 0.7:
            suggestions.append("Confidence score rendah - pertimbangkan untuk menggunakan OpenRouter API key untuk analisis yang lebih akurat")
        
        if balance_difference > 0:
            suggestions.append("Untuk meningkatkan akurasi, pastikan semua item neraca tercakup dalam ekstraksi")
        
        # 11. Determine if validation passes
        is_valid = len(errors) == 0 and avg_confidence >= 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=avg_confidence,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _is_financial_page(self, text: str, page_num: int) -> Tuple[bool, float]:
        """
        Determine if a page contains financial data
            
        Returns:
            Tuple of (is_financial, relevance_score)
        """
        text_lower = text.lower()
        score = 0.0
        
        # Financial statement keywords
        financial_keywords = [
            'aset', 'asset', 'liabilitas', 'liabilities', 'ekuitas', 'equity',
            'pendapatan', 'revenue', 'beban', 'expense', 'laba', 'profit',
            'rugi', 'loss', 'kas', 'cash', 'piutang', 'receivable',
            'utang', 'payable', 'modal', 'capital', 'neraca', 'balance sheet',
            'laporan keuangan', 'financial statement', 'posisi keuangan',
            'comprehensive income', 'arus kas', 'cash flow'
        ]
        
        # Count keyword matches
        for keyword in financial_keywords:
            if keyword in text_lower:
                score += 1.0
        
        # Check for numbers (financial pages have lots of numbers)
        import re
        numbers = re.findall(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?', text)
        if len(numbers) > 10:
            score += 2.0
        elif len(numbers) > 5:
            score += 1.0
        
        # Check for table-like structures (multiple spaces or tabs)
        if '  ' in text or '\t' in text:
            score += 0.5
        
        # First 30 pages more likely to contain financial statements
        if page_num <= 30:
            score += 0.5
        
        # Penalize very short pages (likely cover/separator pages)
        if len(text) < 200:
            score -= 2.0
        
        # Penalize pages far from the beginning (financial statements usually in first 50 pages)
        if page_num > 50:
            score -= 1.0
        if page_num > 70:
            score -= 2.0
        
        # Page is financial if score > 4.0 (stricter threshold to filter better)
        return (score > 4.0, score)
    
    def extract_structured_financial_data(self, text: str, statement_type: str) -> List[Dict]:
        """
        Extract financial data using structured LLM prompts per statement type
        
        Args:
            text: Text from financial statement pages
            statement_type: Type of statement (balance_sheet, income_statement, cash_flow, equity)
            
        Returns:
            List of extracted financial items as dictionaries
        """
        if statement_type == "balance_sheet":
            prompt = f"""Extract ALL line items from this Indonesian Balance Sheet (Neraca/Laporan Posisi Keuangan).

REQUIRED JSON FORMAT - Return ONLY valid JSON array:
[
  {{"item": "Kas dan Setara Kas", "value": 150000000, "category": "aset_lancar", "year": 2024}},
  {{"item": "Piutang Usaha", "value": 75000000, "category": "aset_lancar", "year": 2024}}
]

CATEGORIES (use exact names):
- aset_lancar (Current Assets)
- aset_tidak_lancar (Non-current Assets)
- liabilitas_jangka_pendek (Current Liabilities)
- liabilitas_jangka_panjang (Non-current Liabilities)
- ekuitas (Equity)

RULES:
1. Extract EVERY line item with a value
2. Convert Indonesian formatted numbers (e.g., "1.234.567") to plain numbers (1234567)
3. Keep original Indonesian item names
4. Include year if visible
5. Return ONLY the JSON array, no explanations

TEXT:
{text[:4000]}

JSON Array:"""
        
        elif statement_type == "income_statement":
            prompt = f"""Extract ALL line items from this Indonesian Income Statement (Laporan Laba Rugi).

REQUIRED JSON FORMAT - Return ONLY valid JSON array:
[
  {{"item": "Pendapatan", "value": 500000000, "category": "pendapatan", "year": 2024}},
  {{"item": "Beban Pokok Penjualan", "value": -300000000, "category": "beban", "year": 2024}}
]

CATEGORIES (use exact names):
- pendapatan (Revenue)
- beban (Expenses)
- pendapatan_lain (Other Income)
- beban_lain (Other Expenses)
- laba_rugi (Profit/Loss)

RULES:
1. Extract EVERY line item with a value
2. Convert numbers to plain format
3. Use negative for expenses/losses
4. Keep original Indonesian names
5. Return ONLY the JSON array

TEXT:
{text[:4000]}

JSON Array:"""
        
        else:  # general extraction
            prompt = f"""Extract financial data from this Indonesian financial statement.

REQUIRED JSON FORMAT:
[
  {{"item": "Item Name", "value": 123456, "category": "category_name", "year": 2024}}
]

Extract all line items with values. Return ONLY valid JSON array.

TEXT:
{text[:4000]}

JSON Array:"""
        
        try:
            response = self.query_llm(prompt, max_tokens=4000)
            cleaned = self._clean_llm_response(response)
            items = json.loads(cleaned)
            
            if isinstance(items, list):
                self.logger.info(f"Extracted {len(items)} items from {statement_type}")
                return items
            else:
                self.logger.warning(f"LLM didn't return array for {statement_type}")
                return []
        except Exception as e:
            self.logger.error(f"Error extracting {statement_type}: {e}")
            return []
    
    def _clean_llm_response(self, llm_response: str) -> str:
        """Clean LLM response to extract valid JSON"""
        cleaned = llm_response.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith('```'):
            first_newline = cleaned.find('\n')
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        
        # Extract JSON array if embedded in text
        if not cleaned.startswith('['):
            start_idx = cleaned.find('[')
            end_idx = cleaned.rfind(']')
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx:end_idx + 1]
        
        return cleaned
    
    def _process_llm_items(self, llm_items: List[Dict], page_numbers: List[int]) -> List[FinancialItem]:
        """Process items extracted by LLM into FinancialItem objects"""
        financial_items = []
        
        for item_data in llm_items:
            try:
                # Ensure required fields exist
                if not all(key in item_data for key in ['name', 'value']):
                    continue
                
                # Get page number (use from LLM response or default to first page in batch)
                page_num = item_data.get('page', page_numbers[0] if page_numbers else 1)
                context = item_data.get('context', '')
                
                # Parse the value
                numerical_value, notes = self.parse_financial_value(str(item_data['value']))
                
                if numerical_value == 0.0 and 'Error' in notes:
                    continue  # Skip items with parsing errors
                
                # Categorize the item (using enhanced rule-based method, no LLM)
                category, subcategory, confidence = self.categorize_financial_item(
                    item_data['name'], context
                )
                
                # Create FinancialItem
                financial_item = FinancialItem(
                    name=item_data['name'],
                    value=numerical_value,
                    category=category,
                    subcategory=subcategory,
                    confidence_score=confidence,
                    source_text=context[:300],
                    page_number=page_num,
                    notes=notes
                )
                
                financial_items.append(financial_item)
                
            except Exception as e:
                self.logger.warning(f"Error processing LLM item: {e}")
                continue
        
        return financial_items
    
    def _regex_fallback_extraction(self, page_numbers: List[int], pages_text: Dict[int, str]) -> List[FinancialItem]:
        """Fallback extraction using regex when LLM fails"""
        financial_items = []
        
        for page_num in page_numbers:
            if page_num not in pages_text:
                continue
                
            clean_text = pages_text[page_num]
            regex_figures = self.extract_financial_figures_regex(clean_text)
            
            self.logger.info(f"Regex fallback for page {page_num}: found {len(regex_figures)} figures")
            
            # Process up to 20 items per page
            for figure in regex_figures[:20]:
                item_name = figure.get('item_name', '')
                if not item_name or len(item_name) < 2:
                    continue
                
                numerical_value, notes = self.parse_financial_value(figure['value'])
                if numerical_value == 0.0:
                    continue
                
                category, subcategory, confidence = self.categorize_financial_item(
                    item_name, figure['context']
                )
                
                financial_item = FinancialItem(
                    name=item_name,
                    value=numerical_value,
                    category=category,
                    subcategory=subcategory,
                    confidence_score=confidence * 0.7,  # Lower confidence for regex-only
                    source_text=figure['context'][:200],
                    page_number=page_num,
                    notes=notes + " (Regex extraction)"
                )
                
                financial_items.append(financial_item)
            
        return financial_items
    
    def analyze_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main method to analyze a financial statement document
        
        Args:
            pdf_path: Path to the PDF financial statement
            
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info(f"Starting analysis of document: {pdf_path}")
        
        # Step 1: Extract text from PDF
        extracted_data = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Process pages in batches to reduce API calls
        financial_items = []
        pages_text = {}
        financial_pages = []
        
        # Preprocess all pages and filter for financial content
        self.logger.info("Filtering pages for financial content...")
        for page_num, text in extracted_data['text_by_page'].items():
            clean_text = self.preprocess_text(text)
            pages_text[page_num] = clean_text
            
            # Check if page contains financial data
            is_financial, score = self._is_financial_page(clean_text, page_num)
            if is_financial:
                financial_pages.append((page_num, score))
                self.logger.info(f"✓ Page {page_num}: Financial data detected (score: {score:.1f}, {len(clean_text)} chars)")
            else:
                self.logger.debug(f"✗ Page {page_num}: Skipped (score: {score:.1f}, not financial)")
        
        # Sort by relevance score (highest first) and limit
        financial_pages.sort(key=lambda x: x[1], reverse=True)
        
        # Apply max_pages limit
        if self.max_pages and len(financial_pages) > self.max_pages:
            self.logger.info(f"Limiting to top {self.max_pages} most relevant pages (out of {len(financial_pages)})")
            financial_pages = financial_pages[:self.max_pages]
        
        # Get page numbers to process (sorted by page number for logical order)
        page_numbers = sorted([pnum for pnum, score in financial_pages])
        
        self.logger.info(f"Processing {len(page_numbers)} financial pages out of {len(pages_text)} total pages")
        
        # Batch process pages with LLM (3 pages at a time to reduce API calls)
        if self.openai_api_key:
            batch_size = 3
            total_batches = (len(page_numbers) + batch_size - 1) // batch_size
            
            for batch_idx, i in enumerate(range(0, len(page_numbers), batch_size), 1):
                batch_pages = page_numbers[i:i+batch_size]
                self.logger.info(f"📊 Processing batch {batch_idx}/{total_batches}: pages {batch_pages}")
                
                # Combine text from multiple pages
                combined_text = ""
                page_markers = {}
                current_pos = 0
                
                for pnum in batch_pages:
                    page_text = pages_text[pnum][:1500]  # Limit per page to fit in context
                    page_markers[pnum] = (current_pos, current_pos + len(page_text))
                    combined_text += f"\n--- PAGE {pnum} ---\n{page_text}\n"
                    current_pos = len(combined_text)
                
                # Single LLM call for multiple pages
                llm_prompt = f"""You are a financial data extraction expert. Extract ALL financial line items with their values from the Indonesian financial statement text below.

RULES:
1. Extract every line item that has a name and a numerical value
2. Include the category context (e.g., "ASET LANCAR", "LIABILITAS", "EKUITAS") 
3. Keep the original Indonesian names
4. Keep values exactly as shown (with dots and commas)
5. Include the page number for each item
6. Return ONLY a valid JSON array, NO markdown formatting, NO explanations

EXAMPLE OUTPUT FORMAT:
[{{"name":"Kas dan Setara Kas","value":"150.000.000","context":"ASET LANCAR","page":1}},{{"name":"Piutang Usaha","value":"75.500.000","context":"ASET LANCAR","page":1}}]

TEXT TO EXTRACT FROM:
{combined_text[:8000]}

Return JSON array:"""
                
                try:
                    llm_response = self.query_llm(llm_prompt, max_tokens=6000)
                    self.logger.info(f"LLM batch response preview: {llm_response[:200]}")
                    
                    # Parse response
                    cleaned_response = self._clean_llm_response(llm_response)
                    llm_items = json.loads(cleaned_response)
                    
                    if isinstance(llm_items, list) and len(llm_items) > 0:
                        self.logger.info(f"Successfully extracted {len(llm_items)} items from batch")
                        financial_items.extend(self._process_llm_items(llm_items, batch_pages))
                    else:
                        self.logger.warning(f"Empty LLM response for batch {batch_pages}, using regex fallback")
                        financial_items.extend(self._regex_fallback_extraction(batch_pages, pages_text))
                        
                except Exception as e:
                    self.logger.error(f"LLM batch processing failed: {e}")
                    financial_items.extend(self._regex_fallback_extraction(batch_pages, pages_text))
        else:
            # No API key - use regex extraction for all pages
            self.logger.info("No API key provided, using regex extraction only")
            financial_items.extend(self._regex_fallback_extraction(page_numbers, pages_text))
        
        # Step 3: Validate the extracted data
        validation_result = self.validate_financial_data(financial_items)
        
        # Step 4: Structure the output
        analysis_result = {
            'document_metadata': extracted_data['metadata'],
            'extraction_summary': {
                'total_items_extracted': len(financial_items),
                'pages_processed': len(extracted_data['text_by_page']),
                'extraction_method': extracted_data['extraction_method'],
                'analysis_timestamp': datetime.now().isoformat()
            },
            'financial_data': {
                'items': [asdict(item) for item in financial_items],
                'categorized_summary': self._create_categorized_summary(financial_items)
            },
            'validation': asdict(validation_result),
            'tables_extracted': len(extracted_data['tables'])
        }
        
        self.extracted_data = financial_items
        self.validation_results = validation_result
        
        self.logger.info(f"Analysis completed. Extracted {len(financial_items)} financial items")
        
        return analysis_result
    
    def _create_categorized_summary(self, financial_items: List[FinancialItem]) -> Dict[str, Any]:
        """Create a summary of financial items by category"""
        summary = {}
        
        for item in financial_items:
            if item.category not in summary:
                summary[item.category] = {
                    'total_value': 0,
                    'item_count': 0,
                    'items': [],
                    'avg_confidence': 0
                }
            
            summary[item.category]['total_value'] += item.value
            summary[item.category]['item_count'] += 1
            summary[item.category]['items'].append({
                'name': item.name,
                'value': item.value,
                'confidence': item.confidence_score
            })
        
        # Calculate average confidence for each category
        for category in summary:
            if summary[category]['item_count'] > 0:
                summary[category]['avg_confidence'] = sum(
                    item['confidence'] for item in summary[category]['items']
                ) / summary[category]['item_count']
        
        return summary
    
    def export_to_json(self, output_path: str, analysis_result: Dict[str, Any]) -> None:
        """Export analysis results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Results exported to JSON: {output_path}")
    
    def export_to_excel(self, output_path: str, analysis_result: Dict[str, Any]) -> None:
        """Export analysis results to Excel file"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Financial items sheet
            items_df = pd.DataFrame(analysis_result['financial_data']['items'])
            items_df.to_excel(writer, sheet_name='Financial_Items', index=False)
            
            # Summary by category sheet
            summary_data = []
            for category, data in analysis_result['financial_data']['categorized_summary'].items():
                summary_data.append({
                    'Category': category,
                    'Total_Value': data['total_value'],
                    'Item_Count': data['item_count'],
                    'Avg_Confidence': data['avg_confidence']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Category_Summary', index=False)
            
            # Validation results sheet
            validation_df = pd.DataFrame([{
                'Metric': 'Overall_Valid',
                'Value': analysis_result['validation']['is_valid']
            }, {
                'Metric': 'Confidence_Score',
                'Value': analysis_result['validation']['confidence_score']
            }, {
                'Metric': 'Error_Count',
                'Value': len(analysis_result['validation']['errors'])
            }, {
                'Metric': 'Warning_Count',
                'Value': len(analysis_result['validation']['warnings'])
            }])
            validation_df.to_excel(writer, sheet_name='Validation', index=False)
        
        self.logger.info(f"Results exported to Excel: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Financial Statement Analyzer")
    print("=" * 50)
    print("This tool analyzes financial statements using LLM and NLP.")
    print("Please see the example usage script for detailed implementation.")