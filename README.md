# Skip Song Prediction ML Model

This project implements an end-to-end workflow to predict whether a user will skip a song within a music-streaming session. It combines SQL-based feature engineering with baseline machine-learning models. The aim is clarity, transparency, and a realistic small-scale workflow.

The dataset used here is a **synthetic, Spotify-style session dataset** generated specifically for this project. Its structure and behavior are calibrated to follow patterns reported in published work on Spotify user behavior and skip prediction (see References).

## Overview

Skip prediction is an essential behavioral signal in music-streaming recommendation systems. Predicting which songs users are likely to skip enables more accurate ranking models, better session sequencing, and more responsive personalization strategies.

This repository provides a reproducible and lightweight pipeline including:

- Loading and reviewing (synthetic) streaming event data relevant to skip prediction  
- Exploratory analysis of skip behavior and listening patterns  
- SQL-based feature engineering across user, track, and session-context dimensions  
- Building a consolidated training dataset for supervised learning  
- Training baseline classification models to estimate skip likelihood  
- Evaluating model performance and outlining next steps  

The simulation design is inspired by the **Spotify Songs Analysis and Skip Prediction** notebook on Kaggle:  
https://www.kaggle.com/code/devraai/spotify-songs-analysis-and-skip-prediction

Developed by **Nadine Daum** as part of my data science portfolio (focus on SQL + ML).

## References
Zhang, B., Kreitz, G., Isaksson, M., Ubillos, J., Urdaneta, G., Pouwelse, J. A., & Epema, D. (2013). *Understanding user behavior in Spotify*.  
Brost, B., Mehrotra, R., & Jehan, T. (2019). *The Music Streaming Sessions Dataset*. In Proceedings of the Web Conference (WWW 2019).  
Hansen, C. H., et al. (2020). *Predicting song skipping behaviour in digital music services*. PLOS ONE.  
Author(s). (2019). *Sequential skip prediction in music streaming* (arXiv preprint arXiv:1901.09851).
