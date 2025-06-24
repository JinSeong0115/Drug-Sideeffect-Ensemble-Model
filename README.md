# Drug-Sideeffect-Ensemble-Model

## 📖 Project Overview
This project aims to detect **adverse drug events (ADEs) from social media (SNS) data** by leveraging a stacking ensemble of pre-trained language models: BioBERT, RoBERTa, and ELECTRA.  
The meta-model is pre-trained on a medical dictionary derived from PubMed to improve its understanding of drug-related adverse effects.  
The system first learns from structured **medical data (ADE-Corpus and CADEC)** and then applies this knowledge to identify ADEs in unstructured SNS data.

#### Key Objectives:
- Pre-train a meta-model on a medical dictionary for better ADE detection.
- Use a stacking ensemble of BioBERT, RoBERTa, and ELECTRA to improve prediction accuracy.
- Detect ADEs in SNS data by identifying drug-effect relationships.
  
## 🧠 Architecture
![Architecture](https://github.com/JinSeong0115/Drug-Sideeffect-Ensemble-Model/architecture.svg)
