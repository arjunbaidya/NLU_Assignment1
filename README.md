# Text Classification Analysis: Sports or Politics

**Author:** Arjun Baidya (M25CSA006)  
**Course:** Natural Language Understanding (NLU)  

---

## 1. Introduction
Text classification is a fundamental task in Natural Language Processing (NLP) with applications ranging from spam detection to automated content tagging. As digital news content grows, categorizing articles into domains like Sports or Politics becomes increasingly valuable for recommendation systems.

The objective of this project is to build a robust binary classifier capable of predicting whether a short news description belongs to the **"SPORTS"** or **"POLITICS"** category. We explored various feature representation techniques and machine learning models to identify the most effective pipeline for this task.

## 2. Data Collection and Description

### 2.1 Data Source
The dataset utilized is the **News Category Dataset** obtained from Kaggle (Rishabh Misra, 2022). It contains news headlines and short descriptions scraped from the Huffington Post between 2012 and 2022.

### 2.2 Data Selection
We filtered the original dataset to include only two specific categories:
* **POLITICS**
* **SPORTS**

* **Final Dataset Size:** 8,828 samples.
* **Features Used:** `short_description` (input text) and `category` (target label).

## 3. Exploratory Data Analysis (EDA)

### 3.1 Class Distribution
The dataset is well-balanced, mitigating the risk of bias.
* **Total Samples:** 8,828
* **Politics Samples:** ~4,427
* **Sports Samples:** ~4,401

### 3.2 Text Length Analysis
* **Politics:** Average length of 17.39 words.
* **Sports:** Average length of 15.48 words.
Both distributions are right-skewed, but the similarity in average length suggests document length is not a discriminative feature.

### 3.3 Data Quality
* **Missing Values:** Filled with empty strings.
* **Duplicates:** 76 duplicate rows (<1%) were identified and retained.

## 4. Methodology

### 4.1 Text Preprocessing
We implemented a pipeline to clean the raw text:
1.  **Lowercasing:** To ensure case insensitivity.
2.  **URL Removal:** Removing hyperlinks via Regex.
3.  **Punctuation/Number Removal:** Stripping non-alphabetic characters.
4.  **Whitespace Handling:** Trimming and collapsing spaces.

### 4.2 Feature Extraction Techniques
We employed three vectorization strategies:
* **Bag of Words (BoW):** Represents text as a multiset of words, disregarding grammar. We used the top 10,000 most frequent words.
* **TF-IDF:** Reflects word importance by diminishing the weight of frequent terms and increasing the weight of rare ones.
* **N-Grams (1-3):** Captures local context (unigrams, bigrams, trigrams) to distinguish phrases like "white house" vs. "home run".

### 4.3 Machine Learning Models
We compared three distinct algorithms:
1.  **Multinomial Naive Bayes (MNB):** Based on Bayes' theorem; effective for high-dimensional text data.
2.  **Logistic Regression:** A linear discriminative model that learns the decision boundary between classes.
3.  **Neural Network (MLP):** A Multi-Layer Perceptron with three hidden layers (256, 128, 64 units) and ReLU activation to capture non-linear relationships.

## 5. Experimental Results

The dataset was split into 80% training and 20% testing sets using stratified sampling.

### 5.1 Quantitative Comparison
The **Bag of Words + Naive Bayes** combination achieved the highest accuracy.

| Feature Extraction | Model | Accuracy |
| :--- | :--- | :--- |
| **Bag of Words** | **Naive Bayes** | **0.8652** |
| Bag of Words | Logistic Regression | 0.8607 |
| Bag of Words | Neural Network | 0.8596 |
| TF-IDF | Naive Bayes | 0.8607 |
| TF-IDF | Logistic Regression | 0.8596 |
| TF-IDF | Neural Network | 0.8573 |
| N-Grams (1-3) | Naive Bayes | 0.8641 |
| N-Grams (1-3) | Logistic Regression | 0.8590 |
| N-Grams (1-3) | Neural Network | 0.8567 |

### 5.2 Performance Analysis
* **Best Performer:** BoW + Naive Bayes (86.52%). This supports the heuristic that simpler models often outperform complex ones on smaller datasets.
* **Consistency:** All models performed within a tight range (85.6% - 86.5%), suggesting limitations in the data's discriminative power rather than model capacity.
* **Neural Networks:** The MLP did not outperform linear models, likely requiring more data to generalize better.

### 5.3 Confusion Matrix (BoW + Naive Bayes)

| | Predicted POLITICS | Predicted SPORTS |
| :--- | :--- | :--- |
| **True POLITICS** | **777** | 106 |
| **True SPORTS** | 132 | **751** |

* **Precision (Politics):** ~85.4%
* **Recall (Politics):** ~88.0%
The model is slightly biased towards predicting Politics, resulting in higher recall but more false positives.

### 5.4 Custom Testing
The system correctly classified 10/10 unseen real-world examples, such as "The quarterback threw a touchdown..." (SPORTS) and "The senator filibustered the bill..." (POLITICS).

## 6. Limitations
1.  **Contextual Ambiguity:** Frequency-based embeddings struggle with context shifts (e.g., "race" in politics vs. sports).
2.  **Short Text Data:** Short descriptions often lack sufficient keywords for meaningful vectors.
3.  **Out-of-Vocabulary (OOV):** Rare names or words outside the top 10,000 vocabulary are ignored.
4.  **Lack of Semantic Understanding:** The model treats related terms like "Touchdown" and "Goal" as orthogonal, unlike transformer models.

## 7. Conclusion
This project demonstrated the efficacy of classical machine learning for text classification. With an accuracy of **86.52%**, the **Bag of Words + Naive Bayes** model proved to be the most efficient solution.

**Future Work:**
* Implementing **Word2Vec/GloVe** embeddings for semantic relationships.
* Fine-tuning a **BERT** model to potentially exceed 90% accuracy.
* Hyperparameter tuning (Grid Search) for the MLP classifier.

## References
[1] Rishabh Misra. (2022). News Category Dataset. Kaggle.  
[2] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.  
[3] Jurafsky, D., & Martin, J. H. (2024). Speech and Language Processing. Pearson.
