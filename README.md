# Skip Song Prediction ML Model

This project implements an end-to-end workflow to predict whether a user will skip a song within a music-streaming session. It combines SQL-based feature engineering with baseline machine-learning models to demonstrate a practical approach to music-related behavioral prediction on session data. The aim is clarity, transparency, and a realistic small-scale workflow—nothing over-engineered, but grounded in applied ML practice.

## Overview
Skip prediction is an essential behavioral signal in music-streaming recommendation systems. Predicting which songs users are likely to skip enables more accurate ranking models, better session sequencing, and more responsive personalization strategies.

This repo provides a reproducible & lightweight pipeline including:
- Loading and reviewing streaming event data relevant to skip prediction  
- Exploratory analysis of skip behavior and listening patterns  
- SQL-based feature engineering across user, track, and session-context dimensions  
- Building a consolidated training dataset for supervised learning  
- Training baseline classification models to estimate skip likelihood  
- Evaluating model performance and outlining next steps  

The dataset is derived from the  
**[Song Skipping Challenge 2025 (Kaggle)](https://www.kaggle.com/competitions/song-skipping-challenge-2025/data)**.

Developed by **Nadine Daum** as part of my applied ml portfolio (focus on SQL + ML).
