# Visual-Question-Answering-System-using-Multimodal-Deep-Learning
## Project Description

This project focuses on building a Visual Question Answering (VQA) system capable of understanding abstract images and answering natural language questions about them. VQA is a challenging task in multimodal AI, requiring the integration of computer vision and natural language processing. This project explores various deep learning architectures to effectively fuse visual and textual information to provide accurate answers. The project uses the VQA v1 Abstract Scenes dataset, which allows for a focus on reasoning, language grounding, and object relationships.

## Dataset

The dataset used is the VQA v1 Abstract Scenes dataset. It comprises synthetic, clipart-style images with corresponding natural language questions and multiple-choice answers.

* **Type:** Multiple-Choice
* **Images:** Abstract scenes in color (700x400, RGB) 
* **Questions:** Natural language questions
* **Answers:** Multiple-choice options
* **Dataset Link:** [https://visualqa.org/vqa\_v1\_download.html](https://visualqa.org/vqa_v1_download.html)

## Exploratory Data Analysis (EDA)
A thorough Exploratory Data Analysis was conducted to understand the VQA v1 Abstract Scenes dataset. The key aspects explored include:

1. **Dataset Structure & Quality:** No missing values; 18 candidate answers per question.  
2. **Question & Answer Distribution:**  
   - Avg. question length: ~6 words; dominated by “What” questions.  
   - Answer types: 44.9% Other, 40.7% Yes/No, 14.5% Number.  
   - Imbalance: “yes”, “no”, “1”, “2”, “red” are most frequent.  
3. **Image Characteristics:**  
   - 50,000 synthetic scenes; 3 questions per image; consistent dimensions.

![image](https://github.com/user-attachments/assets/b4ba0d52-8193-4ff6-a5fc-147d8fe587ef)


## Models/Architecture

We implemented four VQA fusion architectures:

| # | Model                          | Fusion Strategy                            | Notes                            |
|---|--------------------------------|-------------------------------------------|----------------------------------|
| 1 | Baseline Concatenation         | Concatenate ResNet + BERT → FC layers      | Lightweight                      |
| 2 | **Gated Fusion Network**       | Sigmoid gates → element‑wise multiply/add  | **Best model**                   |
| 3 | Bilinear Fusion                | Bilinear pooling of gated features         | Richer interactions              |
| 4 | Transformer Fusion             | Modality dropout → multi‑head attention → Transformer block | State‑of‑the‑art reasoning |

The **Gated Fusion Network** was selected as the best-performing model. The architecture details are:

- **Projection:** Image and text features are projected to a 512-D space.
- **Gating Mechanism:** Sigmoid-activated gating layers control modality contribution.
- **Fusion:** Gated vectors are fused via element-wise multiplication and addition.
- **Residual FC Blocks:** Two dense layers with skip connection and dropout.
- **Output:** Softmax classification layer (181 classes).

## Technologies Used

* Python
* TensorFlow
* Hugging Face Transformers (BERT) 
* NumPy, Pandas, Matplotlib, Seaborn 
* Scikit-learn 
* ResNet50 (pre-trained model)
* Google Colab (with A100 GPU) 

## Usage

Follow these notebooks in order for a complete workflow:

1. **Exploratory Data Analysis** 
[`VQA Project EDA.ipynb`](https://github.com/rabiadanish/Visual-Question-Answering-System-using-Multimodal-Deep-Learning/tree/main/Exploratory%20Data%20Analysis/VQA_Project_EDA.ipynb)
* Understand data distributions, question lengths, and answer imbalance.

2. **Preprocessing & Feature Extraction**
[`Data_preprocessing.ipynb`](https://github.com/rabiadanish/Visual-Question-Answering-System-using-Multimodal-Deep-Learning/tree/main/Data%20preprocessing/Data_preprocessing.ipynb)
* Clean and tokenize text (BERT).
* Resize and normalize images, extract ResNet50 features.
* Label‑encode and one‑hot answers.

3. **Model Training**

* Baseline: [`Baseline Concatenation Model.ipynb`](https://github.com/rabiadanish/Visual-Question-Answering-System-using-Multimodal-Deep-Learning/tree/main/Model%20architecture/Baseline_Concatenation_Model.ipynb)
* Gated Fusion: [`Gated Fusion Network.ipynb`](https://github.com/rabiadanish/Visual-Question-Answering-System-using-Multimodal-Deep-Learning/tree/main/Model%20architecture/Gated_Fusion_Network.ipynb)
* Bilinear Fusion: [`Bilinear Fusion Network.ipynb`](https://github.com/rabiadanish/Visual-Question-Answering-System-using-Multimodal-Deep-Learning/tree/main/Model%20architecture/Bilinear_Fusion_Network.ipynb)
* Transformer Fusion: [`Transformer Fusion Network.ipynb`](https://github.com/rabiadanish/Visual-Question-Answering-System-using-Multimodal-Deep-Learning/tree/main/Model%20architecture/Transformer_Fusion_Network.ipynb)

Run the notebook of your chosen approach to train & validate.

4. **Evaluation & Visualization**
[`Evaluation_visualization.ipynb`](https://github.com/rabiadanish/Visual-Question-Answering-System-using-Multimodal-Deep-Learning/tree/main/Evaluation_visualization/Evaluation_visualization.ipynb)
* Generate learning curves.
* Display sample image–question–predicted answer.

*(Tip: If you use Google Colab, mount your Google Drive and point the notebooks at /content/drive/MyDrive/... so you don’t have to re-download data each session.)*

![image](https://github.com/user-attachments/assets/233f8b5e-5b47-4acf-90f5-030340529736)

## Evaluation Metrics

The models were evaluated using the following metrics:

* Validation Loss 
* Validation Accuracy 
* Top-3 Accuracy 

## Results

The Gated Fusion Network achieved the best performance among the four models, with a validation accuracy of 54.97%. After hyperparameter tuning and architectural refinements, the validation accuracy was further improved to 56.68%, with a validation loss of 1.2278 and a Top-3 Accuracy of 86.15%.

![image](https://github.com/user-attachments/assets/b830a41f-591d-4ac7-a585-e87667341563)

## Future Work

Potential improvements and future work include:

* Full fine-tuning of ResNet50/BERT 
* Exploring alternative architectures (e.g., VGG19, Glove+LSTM) 
* Integrating advanced fusion methods (e.g., SEAD, co-attention) 
* Multi-step reasoning 
* Expanding the dataset (e.g., VQA v2.0) 
* Robust data augmentation 
* Upgrading compute resources (e.g., high-end GPUs)
 
## Contributors

* Rabia Danish 
* Eyasu Deresa 
