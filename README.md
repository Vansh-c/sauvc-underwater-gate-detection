# Autonomous Underwater Vehicle (AUV) Gate Detection & Localization
Computer Vision Pipeline for SAUVC 2026 – Ganpat University

## Overview
This repository contains the vision pipeline designed for autonomous underwater gate detection and localization in low-visibility aquatic environments.

## Technical Architecture
- **Model:** YOLOv8 OBB (Oriented Bounding Boxes) trained to infer orientation angles under severe pitch and roll.
- **Dataset:** 1,000+ custom annotated underwater frames (70/20/10 split) with negative sampling to filter background turbidity and light refraction.
- **Deployment:** Low-latency inference targeted for downstream vehicle control.

## Structure
- `train.py` / Training notebooks: Pipeline configuration and fine-tuning.
- `inference.py`: Real-time bounding box and orientation coordinate extraction.