# Relevance Classification

This project focuses on classifying the relevance of text data using machine learning techniques.

## Features

- Data preprocessing
- Model training and evaluation
- Relevance prediction

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/relevance-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd relevance-classification
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset and place it in the `data` directory.
2. Run the script:
    ```bash
    python train_user_model.py 10.10 1 --ratios 10 90 20 80 50 50
    ```
    Where 10.10 is the folder "./data/10.10" with 10 relevant and 10 irrelevant ("./data/10.10/relevant" and "./data/10.10/irrelevant")
    and after 0 or 1 denotes whether you want synthetic data to be used during training
    --ratios: Space seperate relevant and irrelevant ratios


## License

This project is licensed under the MIT License.