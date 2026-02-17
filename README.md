🛒 Grocery Voice AI – Speech-to-Text Engine
📌 Project Overview

Grocery Voice AI is a domain-specific Speech-to-Text (STT) engine designed for grocery order processing.

The system converts voice-based grocery orders into structured SKU-level outputs using:

🎙 Speech-to-Text transcription

🧹 Text normalization

🔎 Fuzzy matching

📦 SKU mapping

📊 Evaluation metrics

This project focuses on improving transcription accuracy for grocery-specific vocabulary such as brands, product names, units, and quantities.

🎯 Problem Statement

General STT engines struggle with:

Brand name recognition (e.g., Vittania → Britannia)

Unit detection (gram, kg, liter)

SKU-level mapping

Accent & pronunciation variations

Noise in audio

Our goal was to build a domain-optimized post-processing pipeline to improve grocery order accuracy.

🏗️ System Architecture
Audio Input
     ↓
Speech-to-Text Engine
     ↓
Raw Transcript
     ↓
Text Normalization
     ↓
Fuzzy Matching
     ↓
SKU Mapping
     ↓
Structured JSON Output
     ↓
Evaluation Metrics

⚙️ Tech Stack

Python

Pandas

Regular Expressions (re)

Jellyfish (String similarity)

RapidFuzz / Fuzzy Matching

JSON