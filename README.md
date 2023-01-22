# 2022-anomaly-detection-cirp

This repo contains files from a publication for the 56th CIRP International Conference on Manufacturing Systems. 

## Title
Unsupervised anomaly detection in unbalanced time series data from screw driving processes using k-means clustering 

## Abstract 
Since bolted joints are ubiquitous in manufacturing, their effective and reliable quality assurance is particularly important. Most tightening processes rely on statistical methods to detect faulty screw connections already during assembly. In this paper, we address the detection of faulty tightening processes using a clustering based approach from the field of Unsupervised Machine Learning. In particular, we deploy the k-Means algorithm on a real-world dataset from the automotive industry. The model uses Dynamic Time Warping to determine the similarity between the normal and abnormal tightening processes, treating each one as an independent temporal sequence. This approach offers three advantages compared to existing supervised methods: 1.) time series with different lengths can be utilized without extensive preprocessing steps, 2.) errors never seen before can be detected using the unsupervised approach, and 3.) extensive manual efforts to generate labels are no longer necessary. To evaluate our approach, we apply it in a scenario where actual class labels are available. This allows evaluating the clustering results using traditional classification scores. The approach manages to achieve an accuracy of up to 88.89% and a macro-average F1-score of up to 63.65%.

## Authors 
* Nikolai West
* Thomas Schlegl
* Jochen Deuse

## Key words 
* nomaly detection 
* time series clustering 
* unsupervised learning 
* screw driving 
* machine learning 
* tightening data
* manufacturing 
* open source

## Code structure
The anaylsis is mostly self-explainatory and includes many comments. 
- In principle, the analysis is divided into three steps for data preparation, clustering and evaluation: 
  - **automotive_data_preparation.py:** This file contains all required preprocessing steps after the automotive data was normalized and anonymized. Since we cannot provide the original dataset, the code cannot be executed. 
  - **automotive_data_clustering.py:** This file contains all required steps for creating the multitude of clustering analysis. The results (model + predictions) are then stored using pickle. 
  - **automotive_data_evaluation.py:** This file contains all final steps for evaluating the clustering results.
- Aside from this, two supportings skripts are available as well:
  - **paper_plots.py:** This file contains the code to create all visualzations used in the paper. 
  - **auxiliaries.py:** This file contains a number of smaller helper function used throughout the analysis. 
