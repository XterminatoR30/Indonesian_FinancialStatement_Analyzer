# Indonesian Financial Statement Analyzer

## Project Overview

The Indonesian Financial Statement Analyzer is an AI-powered tool designed to automatically extract, analyze, and validate financial data from Indonesian public company (Tbk) financial statements. Built with a modern web interface, it streamlines the tedious process of manual data entry and financial analysis while ensuring compliance with Indonesian accounting standards (PSAK).

## Key Features

This comprehensive solution combines PDF processing, natural language processing, and large language models to intelligently extract financial information from complex documents. The system automatically categorizes financial items (assets, liabilities, equity, revenues, expenses) using both rule-based algorithms and AI-powered categorization through OpenRouter's Qwen 2.5VL 72B model—completely free to use.

The analyzer handles Indonesian-specific formatting including Rupiah currency notation, Indonesian number formats (dot as thousands separator), and bilingual terminology (Bahasa Indonesia and English). It performs sophisticated validation checks including balance sheet equation verification, ratio analysis, duplicate detection, and consistency checks across financial statements.

## Technology Stack

Built with Python and Gradio, the application features a beautiful, intuitive web interface accessible through any browser. The backend leverages pdfplumber and PyMuPDF for robust PDF text extraction, pandas for data manipulation, and integrates with OpenRouter's API for intelligent text interpretation—eliminating the need for expensive OpenAI subscriptions.

## Architecture

The project follows clean software design principles with two main components: a core analyzer engine (`financial_statement_analyzer.py`) containing all business logic, and a web interface (`indonesian_financial_analyzer_app.py`) built with Gradio. This modular architecture allows the analyzer to be used independently in scripts, notebooks, or other applications beyond the web UI.

## Output & Export

Results are presented through an interactive dashboard with real-time analysis, confidence scoring for each extracted item, categorized summaries, and detailed validation reports. Users can export findings to Excel or JSON formats for further analysis or integration with existing workflows.

**Target Users**: Financial analysts, accountants, auditors, investors, and researchers working with Indonesian corporate financial statements who need efficient, accurate data extraction and validation.

