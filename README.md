# Sentiment Analysis of E-commerce Reviews

This project aims to perform sentiment analysis on customer reviews collected from two major Indonesian e-commerce platforms, Tokopedia and Shopee. It explores two distinct approaches: a traditional machine learning pipeline using TF-IDF with various classifiers (Naive Bayes, Logistic Regression) and a modern deep learning approach involving the fine-tuning of an Indonesian Bidirectional Encoder Representations from Transformers (IndoBERT) model. The project includes comprehensive data preprocessing, sentiment labeling, feature engineering, model training, evaluation, and interactive prediction functionalities.

## Features

*   **Data Acquisition and Exploration**: Loads and provides initial insights into review datasets from Tokopedia and Shopee, including descriptive statistics and handling of missing/duplicate values.
*   **Indonesian Text Preprocessing**: Implements a robust text cleaning pipeline for Indonesian language reviews, including:
    *   Lowercasing and removal of non-alphabetic characters.
    *   Stop-word removal using the `Sastrawi` library.
    *   Word stemming using the `Sastrawi` library to reduce words to their root forms.
*   **Sentiment Labeling**:
    *   **Rating-based**: Converts numerical star ratings (1-5) into categorical sentiment labels (positif, netral, negatif).
    *   **Lexicon-based**: Utilizes external positive and negative Indonesian lexicons to assign sentiment scores and labels to reviews.
*   **Feature Engineering**:
    *   **TF-IDF Vectorization**: Transforms textual data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency), incorporating unigrams and bigrams.
    *   **Sentiment as Feature**: Incorporates the numerical representation of either rating-based or lexicon-based sentiment as an additional feature for machine learning models.
*   **Traditional Machine Learning Models**:
    *   **Multinomial Naive Bayes (MNB)**: A probabilistic classifier evaluated for its performance in text classification.
    *   **Logistic Regression**: A linear model used for classification, with `class_weight='balanced'` to handle potential class imbalances.
*   **Imbalanced Data Handling**: Employs oversampling techniques (`RandomOverSampler`, `SMOTE`) to mitigate the effects of imbalanced sentiment classes on model training.
*   **Transformer-based Model (IndoBERT)**:
    *   Fine-tunes the `indolem/indobert-base-uncased` pre-trained model from Hugging Face for sentiment classification on the collected review data.
    *   Utilizes the `transformers` library for efficient model training and evaluation.
*   **Model Evaluation**: Generates detailed classification reports (precision, recall, F1-score) to assess the performance of trained models.
*   **Word Cloud Visualization**: Creates a visual representation of the most frequent words in the processed reviews, excluding common Indonesian stopwords.
*   **Interactive Prediction**: Provides functions to predict the sentiment of new, user-provided review texts using both the traditional ML model and the fine-tuned IndoBERT model.
*   **Model Deployment**: The fine-tuned IndoBERT model is saved and can be uploaded to the Hugging Face Hub for public sharing and easy integration into other applications.

## Prerequisites

Before running this project, ensure you have the following installed:

*   **Python**: Version 3.8 or higher.
*   **Jupyter Notebook / JupyterLab**: For executing the `.ipynb` files.

*   **Python Libraries**:
    You can install all necessary libraries using `pip`:

    ```bash
    pip install pandas scikit-learn Sastrawi nltk matplotlib seaborn imblearn scipy transformers datasets huggingface_hub wordcloud
    ```

*   **NLTK Data**: Download the `stopwords` corpus for NLTK:

    ```python
    import nltk
    nltk.download('stopwords')
    ```

*   **Data Files**:
    *   `tokopedia_reviews.csv`: Raw review data for Tokopedia.
    *   `shopee_reviews.csv`: Raw review data for Shopee.
    *   `positive.tsv`: Tab-separated values file containing positive lexicon words and their weights.
    *   `negative.tsv`: Tab-separated values file containing negative lexicon words and their weights.

## Installation

1.  **Clone the repository**:
    If this project is hosted on GitHub, you can clone it using:
    ```bash
    git clone https://github.com/galang006/sentimen_analysis_tokopedia_dan_shopee
    cd sentimen_analysis_tokopedia_dan_shopee
    ```
    Alternatively, download the project files and navigate to the project root directory.

2.  **Create a virtual environment (recommended)**:
    It's good practice to use a virtual environment to manage project dependencies.

    ```bash
    python -m venv venv
    ```

    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies**:
    With your virtual environment activated, install the required Python libraries:

    ```bash
    pip install pandas scikit-learn Sastrawi nltk matplotlib seaborn imblearn scipy transformers datasets huggingface_hub wordcloud
    ```

4.  **Download NLTK stopwords**:
    Run the following Python code once to download necessary NLTK data:

    ```python
    import nltk
    nltk.download('stopwords')
    ```

5.  **Place Data and Lexicon Files**:
    Ensure your project structure includes the `dataset/` and `InSet/` folders with the necessary `.csv` and `.tsv` files:

    ```
    <project_root>/
    ├── sentimen_analysis.ipynb
    ├── sentiment_analysis_tokopedia_and_shopee_using_indobert.ipynb
    ├── .gitignore
    ├── dataset/
    │   ├── tokopedia_reviews.csv
    │   └── shopee_reviews.csv
    │   └── olshop_reviews_lexicon.csv (will be generated by first notebook)
    └── InSet/
        ├── positive.tsv
        └── negative.tsv
    ```
    If these data files are not included in the repository clone, you will need to obtain them and place them in the correct directory structure.

## Usage

This project consists of two main Jupyter Notebooks, each demonstrating a different approach to sentiment analysis.

### 1. Traditional Machine Learning Approach (`sentimen_analysis.ipynb`)

This notebook covers the full pipeline from raw data loading to model training and basic prediction using classical ML models.

1.  **Start Jupyter Notebook/Lab**:
    Navigate to your project directory in the terminal and run:
    ```bash
    jupyter notebook
    ```
    or
    ```bash
    jupyter lab
    ```
2.  **Open the Notebook**:
    In your browser, open `sentimen_analysis.ipynb`.
3.  **Run All Cells**:
    Execute all cells sequentially (`Cell > Run All`).
    *   The notebook will perform:
        *   Data loading and initial exploration of Tokopedia and Shopee reviews.
        *   Text preprocessing specific to Indonesian language (cleaning, stop-word removal, stemming).
        *   Sentiment labeling based on star ratings (`Sentimen_Star`) and external lexicons (`sentimen_lexicon`).
        *   Feature extraction using TF-IDF.
        *   Training and evaluation of Multinomial Naive Bayes and Logistic Regression models, with consideration for class imbalance.
        *   Generation of a word cloud.
4.  **Interactive Prediction**:
    At the end of the notebook, there's a cell that prompts for user input to predict sentiment. The `predict_sentiment_lexicon` function is designed to take a new review text, preprocess it, extract TF-IDF features, calculate its lexicon-based sentiment score (used as an additional feature), and then use the trained Logistic Regression model (which was trained on TF-IDF + rating-derived sentiment as features) to predict the sentiment.

    *Note: There appears to be a minor inconsistency in the `sentimen_analysis.ipynb` notebook where the final `lr_smote_model` is trained with TF-IDF features combined with `rating_sentiment_encoded`, but the `predict_sentiment_lexicon` function attempts to use lexicon-derived sentiment as a feature for new input. For robust predictions, ensure the features used during training and inference are consistent.*

### 2. IndoBERT Fine-tuning and Prediction (`sentiment_analysis_tokopedia_and_shopee_using_indobert.ipynb`)

This notebook demonstrates how to fine-tune a pre-trained IndoBERT model for sentiment analysis and use it for predictions. This approach leverages transfer learning for potentially higher accuracy.

1.  **Ensure Preprocessed Data**: This notebook relies on the `olshop_reviews_lexicon.csv` file, which is generated by running the `sentimen_analysis.ipynb` notebook. Make sure this file exists in your `dataset/` directory. If running in a cloud environment like Google Colab, you might need to upload this file or mount your Google Drive.
2.  **Start Jupyter Notebook/Lab**:
    Similar to the first notebook, open Jupyter Notebook/Lab.
3.  **Open the Notebook**:
    In your browser, open `sentiment_analysis_tokopedia_and_shopee_using_indobert.ipynb`.
4.  **Run All Cells**:
    Execute all cells sequentially (`Cell > Run All`).
    *   The notebook will:
        *   Load the preprocessed data and map sentiment labels to numerical IDs.
        *   Load the `indolem/indobert-base-uncased` tokenizer and model from Hugging Face.
        *   Tokenize the review texts.
        *   Split the dataset into training and evaluation sets.
        *   Define training arguments and metrics for the Hugging Face `Trainer`.
        *   Fine-tune the IndoBERT model.
        *   Evaluate the fine-tuned model's performance.
        *   Save the fine-tuned model and tokenizer locally in the `indobert-sentiment/` directory.
        *   **Optional Model Upload**: You will be prompted to log in to Hugging Face Hub (requires a Hugging Face token with write access) to upload the fine-tuned model.
5.  **Interactive Prediction**:
    The final cell allows you to input a review text, and the fine-tuned IndoBERT model will predict its sentiment (negatif, netral, or positif) along with probability scores.

## Code Structure

The project directory is organized as follows:

```
<project_root>/
├── sentimen_analysis.ipynb
├── sentiment_analysis_tokopedia_and_shopee_using_indobert.ipynb
├── .gitignore
├── dataset/
│   ├── tokopedia_reviews.csv
│   ├── shopee_reviews.csv
│   └── olshop_reviews_lexicon.csv (Generated by sentimen_analysis.ipynb)
├── InSet/
│   ├── positive.tsv
│   └── negative.tsv
├── indobert-sentiment/ (Generated during IndoBERT fine-tuning)
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── ... (other model/tokenizer files)
├── results/ (Generated during IndoBERT training)
└── logs/ (Generated during IndoBERT training)
```

*   **`sentimen_analysis.ipynb`**:
    This Jupyter Notebook implements the traditional machine learning pipeline for sentiment analysis. It handles data loading, extensive Indonesian text preprocessing (stop-word removal, stemming), lexicon-based sentiment scoring, TF-IDF feature extraction, and trains classification models such as Naive Bayes and Logistic Regression. It also includes data visualization components like word clouds and an interactive function for sentiment prediction using the best-performing traditional model.

*   **`sentiment_analysis_tokopedia_and_shopee_using_indobert.ipynb`**:
    This Jupyter Notebook focuses on the deep learning approach. It loads the preprocessed data (ideally generated by the first notebook), utilizes the Hugging Face Transformers library to fine-tune a pre-trained `IndoBERT` model for sentiment classification, evaluates its performance, and includes functionality to save and upload the fine-tuned model to the Hugging Face Hub. It also provides an interactive cell to test the IndoBERT model's predictions.

*   **`.gitignore`**:
    Specifies files and directories that should be excluded from version control. This includes the `dataset/` folder (for large data files), `InSet/` (for lexicon files), and generated model output directories like `indobert-sentiment/`, `results/`, and `logs/`.

*   **`dataset/`**:
    This directory is intended to store the raw and intermediate processed data files.
    *   `tokopedia_reviews.csv`: Contains raw review data specific to Tokopedia.
    *   `shopee_reviews.csv`: Contains raw review data specific to Shopee.
    *   `olshop_reviews_lexicon.csv`: A processed CSV file generated by `sentimen_analysis.ipynb`, containing combined reviews from both platforms with sentiment labels derived from lexicons.

*   **`InSet/`**:
    This directory holds the lexicon files crucial for the lexicon-based sentiment analysis and feature engineering.
    *   `positive.tsv`: A tab-separated file listing Indonesian words associated with positive sentiment, along with their assigned weights.
    *   `negative.tsv`: A tab-separated file listing Indonesian words associated with negative sentiment, along with their assigned weights.

*   **`indobert-sentiment/`**:
    This directory is automatically created by the `sentiment_analysis_tokopedia_and_shopee_using_indobert.ipynb` notebook after the IndoBERT model has been fine-tuned. It contains the model's weights (`pytorch_model.bin` or `model.safetensors`), configuration files (`config.json`), and tokenizer files (`vocab.txt`, `tokenizer_config.json`, etc.) necessary to load and use the fine-tuned model.

*   **`results/` and `logs/`**:
    These directories are generated by the Hugging Face `Trainer` during the fine-tuning process of the IndoBERT model. They store checkpoints, training statistics, and logs related to the model's performance and training progress.