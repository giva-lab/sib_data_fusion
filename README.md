# Exploring Urban Factors with Autoencoders: Relationship Between Static and Dynamic Features

This project investigates how static and dynamic urban data can be fused and analyzed through deep learning methods, particularly autoencoders, within a visualization-assisted framework. The goal is to assess whether **fused latent representations** provide better insights than analyzing data modalities separately.

## Motivation

Urban data often comes in different forms—some static (e.g., population, infrastructure), and some dynamic (e.g., crime reports, traffic, pollution). Fusing these sources effectively can improve predictive performance and reveal deeper patterns in the data.

## Data

| Type     | Description                                   | Format       |
|----------|-----------------------------------------------|--------------|
| Static   | Invariant node-level features (infrastructure and demographics) | `.csv` files |
| Dynamic  | Monthly crime counts per node (144 months)    | `.csv` / DataFrames |

All data is **discretized at the street level** (nodes in a spatial graph).


## Fusion Strategy

The core methodological challenge lies in the fusion of heterogeneous node attributes—specifically, integrating static features with temporally dynamic crime series. This fusion process combines representation learning via Graph Autoencoders (GAEs).

We explore four GAE-based architectures that vary in how and when fusion is applied:

<img width="1690" height="857" alt="image" src="https://github.com/user-attachments/assets/e6e9caf0-914f-463c-ad72-81016a7b5499" />

🟦 M1 Dynamic and M1 Static — Independent Embedding of Static and Dynamic Features

🟥 M2 — Early Fusion via Feature Concatenation

🟨 M3 — Late Fusion of Embeddings

🟩 M4 — Hierarchical Fusion via Stacked GAEs

## Visualization Tool

Based on [CityHub](http://sibgrapi.sid.inpe.br/col/sid.inpe.br/sibgrapi/2022/09.14.21.46/doc/sibgrapi2022_cityhub-preprint.pdf)

<img width="1344" height="976" alt="image" src="https://github.com/user-attachments/assets/3123f5e4-50a3-40d7-a78c-6e3c4e8a0616" />


## Installation 


## References

