# VQASynth Application Documentation

## 1. System Overview

**VQASynth** is a data synthesis pipeline designed to enhance the spatial reasoning capabilities of Vision Language Models (VLMs). It addresses the scarcity of spatial reasoning data in common pretraining datasets by generating high-quality, synthetic Visual Question Answering (VQA) samples from any image dataset on the Huggingface Hub.

The system fuses semantic and metric data into templated VQA conversations. It integrates several models for 3D scene reconstruction and understanding, including:

- Metric depth estimation with VGGT
- Object localization and refinement with SAM2
- Object-grounded captioning

### 1.1 Intended Use

The primary intended use of VQASynth is to generate instruction-tuning datasets for VLMs. Models fine-tuned on this data are expected to improve their ability to:

*   Estimate 3D distances between objects in a 2D image.
*   Describe spatial relationships and orientations colloquially (e.g., "left of," "behind").
*   Convert between common units of measurement.
*   Base responses on consistent references like floors and surfaces.
*   Apply Chain-of-Thought (CoT) "thinking" for more robust reasoning.

This enhanced capability is crucial for embodied AI applications such as robotics, augmented reality, and autonomous navigation.