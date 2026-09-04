# Brain Tumor Detection & Segmentation System
## Overview

This project presents an AI-powered pipeline for detecting and segmenting brain tumors from MRI scans. It combines object detection (YOLO) with advanced segmentation models (ResUNet, Attention UNet, UNet++) to produce accurate tumor analysis.

## Dataset
[Brain Tumor Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

## Web App
The Web App can be used to give an MRI scan of the brain as input to detect and segment tumor
[Brain Tumor Detection and Segmentation](https://huggingface.co/spaces/Nadeef/BrainTumor)

<img width="1899" height="935" alt="Screenshot 2026-09-04 163209" src="https://github.com/user-attachments/assets/3e076c79-fb77-4756-bcb0-29a56178e0db" />

## Pipeline
-MRI Input  
-Preprocessing  
-Converting mask images to polygons identifiable by YOLO  
-Tumor Detection (YOLO)  
-Bounding boxes  
-Cropping Region of Interest  
-Tumor Segmentation (Ensemble UNets) and Tumor Area Calculation  

## Key Features
Hybrid detection + segmentation pipeline
Ensemble learning for robustness
Quantitative tumor analysis
Designed for real-world clinical usage

## Results
Detection(YOLO):
MAP50-90: 0.39, MAP50: 0.756
Segmentation(Ensemble UNet):
Dice Scores:
ResUNet: 0.9237
Attention U-Net: 0.9180
UNet++: 0.9232

## Future Work
Multi-modal data integration
Treatment recommendation system
3D tumor segmentation
Web/mobile deployment

## Tech Stack
Python
PyTorch 
OpenCV
YOLO

## Disclaimer
This project is for research purposes only and not intended for direct clinical use.
