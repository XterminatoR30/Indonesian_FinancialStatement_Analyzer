"""
Indonesian Financial Statement Analyzer - Web Interface
======================================================

A web-based interface for analyzing Indonesian stock company financial statements
using the FinancialStatementAnalyzer class.

Features:
- File upload for PDF financial statements
- Real-time analysis with progress tracking
- Interactive results display
- Export functionality for analysis results
- Indonesian language support

Designed for: Indonesian Public Companies (Tbk) Financial Statements
Author: Kevin Cliff Gunawan
Date: 21 October 2025
"""

import gradio as gr
import pandas as pd
import json
import os
import tempfile
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

# Import our financial analyzer
from financial_statement_analyzer import FinancialStatementAnalyzer, FinancialItem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndonesianFinancialAnalyzerApp:
    """Web application for Indonesian financial statement analysis"""
    
    def __init__(self):
        self.analyzer = None
        self.current_results = None
        
    def initialize_analyzer(self, api_key: str = None, max_pages: int = 30) -> str:
        """Initialize the financial analyzer with optional API key and page limit"""
        try:
            self.analyzer = FinancialStatementAnalyzer(
                openai_api_key=api_key if api_key else None,
                model="openai/gpt-4o-mini",  # More reliable for academic use
                max_pages=max_pages
            )
            return f"✅ Analyzer initialized! (Model: GPT-4o-mini, Max pages: {max_pages})"
        except Exception as e:
            return f"❌ Error inisialisasi: {str(e)}"
    
    def analyze_financial_statement(self, 
                                  pdf_file,
                                  max_pages_input: int = 30,
                                  progress=gr.Progress()) -> Tuple[str, str, str, str]:
        """
        Analyze uploaded financial statement PDF using LLM-based extraction
        
        Args:
            pdf_file: Uploaded PDF file
            max_pages_input: Maximum pages to process
            progress: Gradio progress tracker
        
        Returns:
            Tuple of (summary_html, detailed_results, validation_report, export_data)
        """
        if pdf_file is None:
            return "❌ Please upload a PDF file first.", "", "", ""
        
        try:
            # Initialize analyzer with max_pages setting
            if self.analyzer is None:
                init_result = self.initialize_analyzer(max_pages=max_pages_input)
                if "Error" in init_result:
                    return init_result, "", "", ""
            else:
                # Update max_pages if analyzer already exists
                self.analyzer.max_pages = max_pages_input
            
            progress(0.1, desc="Memproses file PDF...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                # Handle both file-like objects and bytes
                if isinstance(pdf_file, bytes):
                    tmp_file.write(pdf_file)
                else:
                    tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            progress(0.3, desc="Mengekstrak teks dari PDF...")
            
            # Analyze the document
            results = self.analyzer.analyze_document(tmp_path)
            self.current_results = results
            
            progress(0.7, desc="Menganalisis data keuangan...")
            
            # Generate summary HTML
            summary_html = self._generate_summary_html(results)
            
            progress(0.9, desc="Menyiapkan laporan...")
            
            # Generate detailed results
            detailed_results = self._generate_detailed_results(results)
            
            # Generate validation report
            validation_report = self._generate_validation_report(results)
            
            # Prepare export data
            export_data = self._prepare_export_data(results)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            progress(1.0, desc="Analysis complete!")
            
            return summary_html, detailed_results, validation_report, export_data
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return f"❌ Error in analysis: {str(e)}", "", "", ""
    
    def _generate_summary_analysis(self, financial_items, category_counts, total_value, validation) -> str:
        """Generate comprehensive summary analysis text"""
        # Calculate key metrics
        total_assets = sum(item.value for item in financial_items if item.category in ['aset_lancar', 'aset_tidak_lancar'])
        total_liabilities = sum(item.value for item in financial_items if item.category in ['liabilitas_jangka_pendek', 'liabilitas_jangka_panjang'])
        total_equity = sum(item.value for item in financial_items if item.category == 'ekuitas')
        total_revenue = sum(item.value for item in financial_items if item.category == 'pendapatan')
        
        avg_confidence = validation.get('confidence_score', 0)
        
        analysis = "📊 **RINGKASAN ANALISIS LAPORAN KEUANGAN**\n\n"
        analysis += f"• **Total Item Terekstrak**: {len(financial_items):,} line items dari dokumen\n"
        analysis += f"• **Kategori Teridentifikasi**: {len(category_counts)} kategori utama (Aset, Liabilitas, Ekuitas, dll)\n"
        analysis += f"• **Total Nilai Keseluruhan**: Rp {total_value:,.0f}\n"
        analysis += f"• **Status Validasi**: {'✅ Valid' if validation.get('is_valid', False) else '⚠️ Perlu Review'} (Confidence: {avg_confidence:.1%})\n\n"
        
        if total_assets > 0:
            analysis += f"• **Total Aset**: Rp {total_assets:,.0f} ({(total_assets/total_value*100):.1f}% dari total)\n"
        if total_liabilities > 0:
            analysis += f"• **Total Liabilitas**: Rp {total_liabilities:,.0f} ({(total_liabilities/total_value*100):.1f}% dari total)\n"
        if total_equity > 0:
            analysis += f"• **Total Ekuitas**: Rp {total_equity:,.0f} ({(total_equity/total_value*100):.1f}% dari total)\n"
        if total_revenue > 0:
            analysis += f"• **Total Pendapatan**: Rp {total_revenue:,.0f}\n"
        
        # Balance sheet analysis
        if total_assets > 0 and total_liabilities > 0:
            balance_diff = abs(total_assets - (total_liabilities + total_equity))
            if balance_diff < total_assets * 0.02:
                analysis += f"\n• **Persamaan Neraca**: ✅ Seimbang (selisih: Rp {balance_diff:,.0f})\n"
            else:
                analysis += f"\n• **Persamaan Neraca**: ⚠️ Tidak seimbang (selisih: Rp {balance_diff:,.0f})\n"
        
        # Top categories
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1]['total_value'], reverse=True)
        if len(sorted_cats) > 0:
            analysis += f"\n• **Kategori Terbesar**: {sorted_cats[0][0].replace('_', ' ').title()} - Rp {sorted_cats[0][1]['total_value']:,.0f}\n"
        
        # Quality indicators
        low_conf_items = sum(1 for item in financial_items if item.confidence_score < 0.6)
        if low_conf_items > 0:
            analysis += f"\n• **Perhatian**: {low_conf_items} item dengan confidence rendah (<60%) - perlu review manual\n"
        
        return analysis
    
    def _generate_summary_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML summary of analysis results"""
        # Handle both old and new result formats
        if 'financial_items' in results:
            financial_items = results['financial_items']
        elif 'financial_data' in results and 'items' in results['financial_data']:
            # Convert dict items back to FinancialItem objects for compatibility
            from financial_statement_analyzer import FinancialItem
            items_dicts = results['financial_data']['items']
            financial_items = [FinancialItem(**item) for item in items_dicts]
        else:
            financial_items = []
        
        metadata = results.get('metadata', results.get('document_metadata', {}))
        validation = results.get('validation', {})
        
        # Count items by category
        category_counts = {}
        total_value = 0
        for item in financial_items:
            category = item.category
            if category not in category_counts:
                category_counts[category] = {'count': 0, 'total_value': 0}
            category_counts[category]['count'] += 1
            category_counts[category]['total_value'] += abs(item.value)
            total_value += abs(item.value)
        
        # Generate summary analysis
        summary_analysis = self._generate_summary_analysis(financial_items, category_counts, total_value, validation)
        
        # Store analysis in results for export
        results['summary_analysis'] = summary_analysis
        
        # Generate HTML with proper color contrast
        html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
            <h2 style="margin: 0; text-align: center; color: white;">📊 Ringkasan Analisis Laporan Keuangan</h2>
        </div>
        
        <div style="background: #e3f2fd; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #2196f3;">
            <pre style="white-space: pre-wrap; font-family: Arial, sans-serif; color: #212529; margin: 0; line-height: 1.8;">{summary_analysis}</pre>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px;">
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                <h4 style="margin: 0; color: #28a745;">📄 Informasi Dokumen</h4>
                <p style="color: #212529;"><strong style="color: #212529;">Total Halaman:</strong> {metadata.get('total_pages', 'N/A')}</p>
                <p style="color: #212529;"><strong style="color: #212529;">Judul:</strong> {metadata.get('title', 'N/A')}</p>
                <p style="color: #212529;"><strong style="color: #212529;">Metode Ekstraksi:</strong> {results.get('extraction_method', 'N/A')}</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
                <h4 style="margin: 0; color: #007bff;">💰 Data Keuangan</h4>
                <p style="color: #212529;"><strong style="color: #212529;">Total Item:</strong> {len(financial_items)}</p>
                <p style="color: #212529;"><strong style="color: #212529;">Kategori:</strong> {len(category_counts)}</p>
                <p style="color: #212529;"><strong style="color: #212529;">Total Nilai:</strong> Rp {total_value:,.0f}</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #{'28a745' if validation.get('is_valid', False) else 'dc3545'};">
                <h4 style="margin: 0; color: #{'28a745' if validation.get('is_valid', False) else 'dc3545'};">✅ Status Validasi</h4>
                <p style="color: #212529;"><strong style="color: #212529;">Status:</strong> {'Valid' if validation.get('is_valid', False) else 'Perlu Review'}</p>
                <p style="color: #212529;"><strong style="color: #212529;">Confidence Score:</strong> {validation.get('confidence_score', 0):.2%}</p>
                <p style="color: #212529;"><strong style="color: #212529;">Error:</strong> {len(validation.get('errors', []))}</p>
            </div>
        </div>
        
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h4 style="margin-top: 0; color: #495057;">📈 Distribusi Kategori</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
        """
        
        # Add category distribution
        for category, data in category_counts.items():
            percentage = (data['total_value'] / total_value * 100) if total_value > 0 else 0
            html += f"""
                <div style="background: white; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6;">
                    <strong style="color: #212529;">{category.replace('_', ' ').title()}</strong><br>
                    <small style="color: #6c757d;">{data['count']} item(s) - {percentage:.1f}%</small><br>
                    <small style="color: #6c757d;">Rp {data['total_value']:,.0f}</small>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_detailed_results(self, results: Dict[str, Any]) -> str:
        """Generate detailed results as formatted text"""
        # Handle both old and new result formats
        if 'financial_items' in results:
            financial_items = results['financial_items']
        elif 'financial_data' in results and 'items' in results['financial_data']:
            from financial_statement_analyzer import FinancialItem
            items_dicts = results['financial_data']['items']
            financial_items = [FinancialItem(**item) for item in items_dicts]
        else:
            financial_items = []
        
        if not financial_items:
            return "Tidak ada data keuangan yang berhasil diekstrak."
        
        # Group by category
        categories = {}
        for item in financial_items:
            if item.category not in categories:
                categories[item.category] = []
            categories[item.category].append(item)
        
        detailed_text = "📋 HASIL ANALISIS DETAIL\n"
        detailed_text += "=" * 50 + "\n\n"
        
        for category, items in categories.items():
            detailed_text += f"🏷️ {category.replace('_', ' ').upper()}\n"
            detailed_text += "-" * 30 + "\n"
            
            for item in items:
                detailed_text += f"• {item.name}\n"
                detailed_text += f"  Nilai: Rp {item.value:,.2f}\n"
                detailed_text += f"  Confidence: {item.confidence_score:.2%}\n"
                detailed_text += f"  Halaman: {item.page_number}\n"
                if item.notes:
                    detailed_text += f"  Catatan: {item.notes}\n"
                detailed_text += "\n"
            
            detailed_text += "\n"
        
        return detailed_text
    
    def _generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate validation report"""
        validation = results.get('validation', {})
        
        if not validation:
            return "Laporan validasi tidak tersedia."
        
        report = "🔍 LAPORAN VALIDASI\n"
        report += "=" * 30 + "\n\n"
        
        report += f"Status: {'✅ VALID' if validation.get('is_valid', False) else '⚠️ PERLU REVIEW'}\n"
        report += f"Confidence Score: {validation.get('confidence_score', 0):.2%}\n\n"
        
        errors = validation.get('errors', [])
        if errors:
            report += "❌ ERRORS:\n"
            for error in errors:
                report += f"  • {error}\n"
            report += "\n"
        
        warnings = validation.get('warnings', [])
        if warnings:
            report += "⚠️ WARNINGS:\n"
            for warning in warnings:
                report += f"  • {warning}\n"
            report += "\n"
        
        suggestions = validation.get('suggestions', [])
        if suggestions:
            report += "💡 SUGGESTIONS:\n"
            for suggestion in suggestions:
                report += f"  • {suggestion}\n"
            report += "\n"
        
        return report
    
    def _prepare_export_data(self, results: Dict[str, Any]) -> str:
        """Prepare data for export as JSON with summary analysis"""
        try:
            # Handle both old and new result formats
            if 'financial_items' in results:
                financial_items = results['financial_items']
            elif 'financial_data' in results and 'items' in results['financial_data']:
                from financial_statement_analyzer import FinancialItem
                items_dicts = results['financial_data']['items']
                financial_items = [FinancialItem(**item) for item in items_dicts]
            else:
                financial_items = []
            
            # Calculate summary statistics
            total_assets = sum(item.value for item in financial_items if item.category in ['aset_lancar', 'aset_tidak_lancar'])
            total_liabilities = sum(item.value for item in financial_items if item.category in ['liabilitas_jangka_pendek', 'liabilitas_jangka_panjang'])
            total_equity = sum(item.value for item in financial_items if item.category == 'ekuitas')
            total_value = sum(abs(item.value) for item in financial_items)
            
            # Convert FinancialItem objects to dictionaries with Rupiah formatting
            export_data = {
                'summary_analysis': results.get('summary_analysis', 'No analysis available'),
                'summary_statistics': {
                    'total_items': len(financial_items),
                    'total_value': f"Rp {total_value:,.0f}",
                    'total_assets': f"Rp {total_assets:,.0f}",
                    'total_liabilities': f"Rp {total_liabilities:,.0f}",
                    'total_equity': f"Rp {total_equity:,.0f}",
                    'validation_status': 'Valid' if results.get('validation', {}).get('is_valid', False) else 'Needs Review',
                    'confidence_score': f"{results.get('validation', {}).get('confidence_score', 0):.1%}"
                },
                'metadata': results.get('metadata', results.get('document_metadata', {})),
                'financial_items': [],
                'validation': results.get('validation', {}),
                'analysis_timestamp': datetime.now().isoformat(),
                'analyzer_version': '1.0.0'
            }
            
            for item in financial_items:
                export_data['financial_items'].append({
                    'name': item.name,
                    'value_numeric': item.value,
                    'value_formatted': f"Rp {item.value:,.2f}",
                    'category': item.category,
                    'subcategory': item.subcategory,
                    'confidence_score': f"{item.confidence_score:.1%}",
                    'page_number': item.page_number,
                    'validation_status': item.validation_status,
                    'notes': item.notes
                })
            
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Error preparing export data: {str(e)}"
    
    def export_to_excel(self) -> Tuple[str, str]:
        """Export current results to Excel file with summary analysis and proper formatting"""
        if not self.current_results:
            return "❌ No data to export. Please analyze a document first.", None
        
        try:
            # Handle both old and new result formats
            if 'financial_items' in self.current_results:
                financial_items = self.current_results['financial_items']
            elif 'financial_data' in self.current_results and 'items' in self.current_results['financial_data']:
                from financial_statement_analyzer import FinancialItem
                items_dicts = self.current_results['financial_data']['items']
                financial_items = [FinancialItem(**item) for item in items_dicts]
            else:
                return "❌ No financial data to export.", None
            
            # Create Excel file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_analysis_{timestamp}.xlsx"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            
            # Calculate summary statistics
            total_assets = sum(item.value for item in financial_items if item.category in ['aset_lancar', 'aset_tidak_lancar'])
            total_liabilities = sum(item.value for item in financial_items if item.category in ['liabilitas_jangka_pendek', 'liabilitas_jangka_panjang'])
            total_equity = sum(item.value for item in financial_items if item.category == 'ekuitas')
            total_value = sum(abs(item.value) for item in financial_items)
            
            # Create Excel writer
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Sheet 1: Summary Analysis
                summary_data = {
                    'Metric': [
                        'Total Items Extracted',
                        'Total Value',
                        'Total Assets',
                        'Total Liabilities',
                        'Total Equity',
                        'Validation Status',
                        'Confidence Score',
                        '',
                        'Summary Analysis'
                    ],
                    'Value': [
                        f"{len(financial_items):,} items",
                        f"Rp {total_value:,.0f}",
                        f"Rp {total_assets:,.0f}",
                        f"Rp {total_liabilities:,.0f}",
                        f"Rp {total_equity:,.0f}",
                        'Valid' if self.current_results.get('validation', {}).get('is_valid', False) else 'Needs Review',
                        f"{self.current_results.get('validation', {}).get('confidence_score', 0):.1%}",
                        '',
                        self.current_results.get('summary_analysis', 'No analysis available')
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Financial Items with Rupiah formatting
                data = []
                for item in financial_items:
                    data.append({
                        'Nama Item': item.name,
                        'Nilai (Rupiah)': f"Rp {item.value:,.2f}",
                        'Nilai Numerik': item.value,
                        'Kategori': item.category,
                        'Sub-kategori': item.subcategory,
                        'Confidence Score': f"{item.confidence_score:.1%}",
                        'Halaman': item.page_number,
                        'Status Validasi': item.validation_status,
                        'Catatan': item.notes
                    })
                
                items_df = pd.DataFrame(data)
                items_df.to_excel(writer, sheet_name='Financial Items', index=False)
                
                # Sheet 3: Category Summary
                category_summary = {}
                for item in financial_items:
                    if item.category not in category_summary:
                        category_summary[item.category] = {'count': 0, 'total': 0}
                    category_summary[item.category]['count'] += 1
                    category_summary[item.category]['total'] += abs(item.value)
                
                cat_data = []
                for category, stats in category_summary.items():
                    cat_data.append({
                        'Kategori': category.replace('_', ' ').title(),
                        'Jumlah Item': f"{stats['count']:,}",
                        'Total Nilai': f"Rp {stats['total']:,.0f}",
                        'Persentase': f"{(stats['total']/total_value*100):.2f}%"
                    })
                
                cat_df = pd.DataFrame(cat_data)
                cat_df.to_excel(writer, sheet_name='Category Summary', index=False)
            
            return f"✅ Excel file ready for download: {filename}", filepath
            
        except Exception as e:
            return f"❌ Error exporting Excel: {str(e)}", None
    
    def export_to_json_file(self) -> str:
        """Export current results to JSON file with summary analysis"""
        if not self.current_results:
            return None
        
        try:
            # Handle both old and new result formats
            if 'financial_items' in self.current_results:
                financial_items = self.current_results['financial_items']
            elif 'financial_data' in self.current_results and 'items' in self.current_results['financial_data']:
                from financial_statement_analyzer import FinancialItem
                items_dicts = self.current_results['financial_data']['items']
                financial_items = [FinancialItem(**item) for item in items_dicts]
            else:
                return None
            
            # Create JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_analysis_{timestamp}.json"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            
            # Calculate summary statistics
            total_assets = sum(item.value for item in financial_items if item.category in ['aset_lancar', 'aset_tidak_lancar'])
            total_liabilities = sum(item.value for item in financial_items if item.category in ['liabilitas_jangka_pendek', 'liabilitas_jangka_panjang'])
            total_equity = sum(item.value for item in financial_items if item.category == 'ekuitas')
            total_value = sum(abs(item.value) for item in financial_items)
            
            # Prepare export data with summary analysis and Rupiah formatting
            export_data = {
                'summary_analysis': self.current_results.get('summary_analysis', 'No analysis available'),
                'summary_statistics': {
                    'total_items': len(financial_items),
                    'total_value': f"Rp {total_value:,.0f}",
                    'total_assets': f"Rp {total_assets:,.0f}",
                    'total_liabilities': f"Rp {total_liabilities:,.0f}",
                    'total_equity': f"Rp {total_equity:,.0f}",
                    'validation_status': 'Valid' if self.current_results.get('validation', {}).get('is_valid', False) else 'Needs Review',
                    'confidence_score': f"{self.current_results.get('validation', {}).get('confidence_score', 0):.1%}"
                },
                'metadata': self.current_results.get('metadata', self.current_results.get('document_metadata', {})),
                'financial_items': [],
                'validation': self.current_results.get('validation', {}),
                'analysis_timestamp': datetime.now().isoformat(),
                'analyzer_version': '1.0.0'
            }
            
            for item in financial_items:
                export_data['financial_items'].append({
                    'name': item.name,
                    'value_numeric': item.value,
                    'value_formatted': f"Rp {item.value:,.2f}",
                    'category': item.category,
                    'subcategory': item.subcategory,
                    'confidence_score': f"{item.confidence_score:.1%}",
                    'page_number': item.page_number,
                    'validation_status': item.validation_status,
                    'notes': item.notes
                })
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"JSON export error: {str(e)}")
            return None

def create_interface():
    """Create and configure the Gradio interface"""
    app = IndonesianFinancialAnalyzerApp()
    
    with gr.Blocks(
        title="Indonesian Financial Statement Analyzer",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🏦 Indonesian Financial Statement Analyzer</h1>
            <p>LLM-based Financial Data Extraction & Validation System</p>
            <p>Academic Implementation | Model: GPT-4o-mini | PSAK-Compliant</p>
            <p>Made by: Kevin Cliff Gunawan | October 2025</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload & Konfigurasi")
                
                # File upload
                pdf_input = gr.File(
                    label="Upload Laporan Keuangan (PDF)",
                    file_types=[".pdf"],
                    type="binary"
                )
                
                # Max pages slider
                max_pages_slider = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=30,
                    step=5,
                    label="📄 Max Pages to Process",
                    info="Limit pages for large documents (100+ pages). System selects most relevant pages automatically."
                )
                
                # Analyze button
                analyze_btn = gr.Button(
                    "🔍 Analisis Laporan Keuangan",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("### 💾 Export Hasil")
                
                # Export buttons
                with gr.Row():
                    export_excel_btn = gr.Button(
                        "📊 Export Excel",
                        variant="secondary"
                    )
                    download_json_btn = gr.Button(
                        "📄 Download JSON",
                        variant="secondary"
                    )
                
                # Download components
                excel_download = gr.File(
                    label="📊 Excel File",
                    visible=False
                )
                
                json_download = gr.File(
                    label="📄 JSON File",
                    visible=False
                )
                
                export_status = gr.Textbox(
                    label="Status Export",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Hasil Analisis")
                
                # Results tabs
                with gr.Tabs():
                    with gr.TabItem("📈 Ringkasan"):
                        summary_output = gr.HTML(label="Ringkasan Analisis")
                    
                    with gr.TabItem("📋 Detail"):
                        detailed_output = gr.Textbox(
                            label="Hasil Detail",
                            lines=20,
                            max_lines=None,  # Enable unlimited scrolling
                            show_copy_button=True
                        )
                    
                    with gr.TabItem("✅ Validasi"):
                        validation_output = gr.Textbox(
                            label="Laporan Validasi",
                            lines=15,
                            max_lines=None,  # Enable unlimited scrolling
                            show_copy_button=True
                        )
                    
                    with gr.TabItem("💾 Export Data"):
                        export_data_output = gr.Textbox(
                            label="Data JSON untuk Export",
                            lines=15,
                            max_lines=None,  # Enable unlimited scrolling
                            show_copy_button=True
                        )
        
        # Event handlers
        analyze_btn.click(
            fn=app.analyze_financial_statement,
            inputs=[pdf_input, max_pages_slider],
            outputs=[summary_output, detailed_output, validation_output, export_data_output],
            show_progress=True
        )
        
        def prepare_excel_download():
            """Prepare Excel file for download"""
            status, filepath = app.export_to_excel()
            if filepath:
                return status, gr.File(value=filepath, visible=True)
            else:
                return status, gr.File(visible=False)
        
        export_excel_btn.click(
            fn=prepare_excel_download,
            outputs=[export_status, excel_download]
        )
        
        def prepare_json_download():
            """Prepare JSON file for download"""
            filepath = app.export_to_json_file()
            if filepath:
                return gr.File(value=filepath, visible=True)
            else:
                return gr.File(visible=False)
        
        download_json_btn.click(
            fn=prepare_json_download,
            outputs=[json_download]
        )
        
        # Instructions
        gr.Markdown("""
        ### 📝 How to Use:
        1. **Upload PDF**: Select Indonesian financial statement (e.g., EKAD Annual Report)
        2. **Set Max Pages**: Adjust slider for large documents (default: 30 pages)
        3. **Click Analyze**: System will extract and validate all financial data
        4. **Review Results**: Check Summary, Details, and Validation tabs
        5. **Export**: Download results as Excel or JSON
        
        ### 🎓 Academic Features (Task Compliance):
        - ✅ **LLM-based Extraction**: GPT-4o-mini for reliable data extraction
        - ✅ **Comprehensive Validation**: Balance sheet equation, sanity checks, PSAK compliance
        - ✅ **PSAK Categorization**: Indonesian accounting standards (Aset, Liabilitas, Ekuitas)
        - ✅ **Edge Case Handling**: Non-standard items → "other" category with justification
        - ✅ **Structured Output**: JSON, CSV, Excel with metadata and validation results
        - ✅ **Confidence Scoring**: Every item tagged with accuracy confidence (0-1)
        - ✅ **Multi-level Verification**: LLM + Rule-based + Mathematical validation
        
        ### 📊 What Gets Extracted:
        - **Balance Sheet** (Neraca): Assets, Liabilities, Equity
        - **Income Statement** (Laba Rugi): Revenue, Expenses, Profit/Loss
        - **Cash Flow** (Arus Kas): Operating, Investing, Financing activities
        - **Statement of Changes in Equity**
        
        ### ⚡ Performance:
        - **Small documents (< 30 pages)**: 2-3 minutes
        - **Large documents (100+ pages with 30 page limit)**: 3-5 minutes
        - **Smart filtering**: Automatically selects most relevant pages
        
        ### 📖 Documentation:
        See `README_ACADEMIC.md` for complete methodology, validation details, and compliance with extraction requirements.
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )