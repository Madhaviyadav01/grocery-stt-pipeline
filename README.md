## 🛒 Grocery Voice AI – SmartGrocery Voice STT Engine

## 📌 Project Overview

SmartGrocery Voice is an AI-based Speech-to-Text system designed for grocery sellers. It converts voice orders, usually received through WhatsApp or phone calls, into structured digital records.
The system matches spoken product names and quantities with a predefined SKU list.
It helps reduce manual work and improves order accuracy, especially in noisy environments like warehouses and wholesale markets.

## Key Features

- Multi-Phase Pipeline: A decoupled architecture covering ingestion, transformation, intelligence, and extraction.

- Noise Resilient: Specialized audio preprocessing using FFmpeg to handle warehouse and market background noise.

- Advanced Transcription: Leverages OpenAI’s Whisper model for robust speech recognition across various accents.

- Fuzzy SKU Mapping: Utilizes a hybrid retrieval mechanism (lexical and semantic) to match transcribed text with a master dataset of over 200 products.

- Structured Output: Generates final order data in JSON format for seamless integration with retail or billing systems.

## System Architecture

The system is organized into four primary layers:

- Ingestion Layer: Handles raw audio formats including MP3, WAV, and M4A.

- Transformation Layer: Standardizes audio to 16kHz Mono WAV format for optimal model feature extraction.

- Intelligence Layer: The core STT engine (Whisper) used for transcribing grocery-specific vocabulary.

- Extraction & Analytics Layer: Post-processing logic that performs entity extraction (product, quantity, unit) and fuzzy matching.

  <img width="1011" height="124" alt="image" src="https://github.com/user-attachments/assets/9ab5d2ae-4703-4a35-b6fa-aa8b9fa29b5d" />

```bash
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
```

## ⚙️ Tech Stack

- Python

- Pandas, Whisper model

- Regular Expressions (re)

- Jellyfish (String similarity)

- RapidFuzz / Fuzzy Matching

- JSON

## Performance Evaluation

The system was evaluated using 448 validated grocery voice order samples.

<img width="512" height="384" alt="image" src="https://github.com/user-attachments/assets/bdf49ebb-2903-4403-9282-ee8e7402104b" />

## Future Scope

- Domain Fine-Tuning: Training the STT model on grocery-specific and accent-heavy datasets.

- RAG/LLM Integration: Implementing Retrieval-Augmented Generation to improve contextual SKU selection.

- Phonetic Matching: Adding alias variations to handle brand name pronunciation errors.

```bash
Madhavi Yadav  
MCA – Amrita Vishwa Vidyapeetham  
February 2026
```
