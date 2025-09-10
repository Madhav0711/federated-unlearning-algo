# Federated Unlearning for Medical Imaging using FedEraser

A B.Tech Project exploring the implementation of machine unlearning in a federated learning environment for COVID-19 detection from chest X-ray images.

## Project Overview

This project demonstrates a federated learning (FL) system for classifying chest X-ray images as either **COVID-19 positive** or **Normal**. The core focus is on implementing **"machine unlearning"** using the **FedEraser** algorithm. This allows for the selective removal of a specific client's data contribution from the final trained global model, without needing to retrain the entire system from scratch. This is crucial for data privacy regulations like GDPR's "right to be forgotten" and for mitigating the impact of malicious or corrupted data from a client.

---

## Team and Guide

This project was developed as part of our B.Tech curriculum under the valuable guidance of **Dr. Lal Upendra**.

**Team Members:**
* Madhav Nagpal
* Dudhat Hemil
* Janmej Rana

---

## Key Concepts 🧠

* **Federated Learning (FL):** A decentralized machine learning approach where multiple clients collaboratively train a model without sharing their local data. Each client trains a model on its own data, and the updates are aggregated on a central server to create a global model. This preserves data privacy.

* **Machine Unlearning:** The process of removing the influence of a specific subset of training data from a trained machine learning model. This is essential for user privacy and model maintenance.

* **FedEraser:** An efficient unlearning algorithm for FL. It works by approximating the removal of a client's contribution by applying a "reverse" update to the global model. This is achieved by using stored model checkpoints from the training history to perform a gradient ascent step, effectively canceling out the client's original gradient descent update.

---

## Dataset 🏥

We use the **COVID-19 Radiography Database** available on Kaggle.

* **Source:** [COVID-19 Radiography Database on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
* **Content:** The dataset contains chest X-ray images for two classes: COVID-19 and Normal.
* **Preprocessing:** The images are converted to grayscale, resized to 128x128 pixels, and normalized before being fed into the model.



---

## Methodology and Implementation

### 1. Federated Learning Setup

* **Model Architecture:** A **ResNet-50** model, pre-trained on ImageNet. We modified its first convolutional layer to accept single-channel (grayscale) images and adjusted the final fully connected layer for binary classification (COVID vs. Normal).
* **Clients:** The training data is distributed among **5 simulated clients**.
* **Training Rounds:** The federated training process runs for **15 rounds**.
* **Aggregation:** We use the standard **Federated Averaging (FedAvg)** algorithm to aggregate the client model updates and create the new global model in each round.

### 2. Checkpointing for Unlearning

A crucial part of the process is saving model states. In each round, we save:
1.  The state of the **global model** *before* it is sent to the clients.
2.  The state of each **client's local model** *after* it has been trained on local data.

These checkpoints are essential for the FedEraser algorithm to reconstruct and reverse a client's update.

### 3. FedEraser Unlearning Process

The `FedEraser` class implements the unlearning logic. To unlearn the contribution of a specific client (e.g., Client 2) up to a certain round `r`:
1.  The algorithm iterates backward from round `r` to round 0.
2.  In each round, it calculates the client's update by finding the difference between the client's saved model and the global model from that round:
    $$ \Delta w_{client} = w_{client} - w_{global}^{previous} $$
3.  It then "unlearns" this update from the current global model by performing a gradient ascent step:
    $$ w_{global}^{new} = w_{global}^{current} - \eta \cdot \Delta w_{client} $$
    where $ \eta $ is the unlearning rate.

This process effectively rewinds the contributions of the forgotten client, resulting in a model that approximates one never trained on that client's data.

---

## How to Run the Code

### Prerequisites

Make sure you have Python 3.8+ and the following libraries installed. You can install them using pip:

```bash
pip install torch torchvision pandas Pillow matplotlib kagglehub
```

### Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Kaggle API Credentials:**
    The script uses `kagglehub` to automatically download the dataset. Ensure you have your Kaggle API token (`kaggle.json`) set up. You can typically place it in `~/.kaggle/`.

### Execution

Simply run the Python script from your terminal:

```bash
python fed_avg.py
```

The script will perform the following steps:
1.  Download and preprocess the dataset.
2.  Simulate the federated learning process for 15 rounds, saving all necessary model checkpoints in `global_checkpoints/` and `client_checkpoints/`.
3.  At each round, perform the unlearning experiment to remove the contribution of **Client 2**.
4.  Generate and display two plots comparing the accuracy of the original global model and the unlearned model.

---

## Results and Discussion 📊

The script produces two plots to visualize the experiment's outcome.

**Plot 1: Global Accuracy Over Rounds**
This plot shows the test accuracy of the standard federated global model at the end of each round. We expect to see a general upward trend as the model learns from more client data over time.



**Plot 2: Global vs. Unlearned Accuracy**
This plot is the core result of our project. It compares the accuracy of the original global model with the accuracy of the model after unlearning Client 2's data.

* **Observation:** The accuracy of the "unlearned" model is consistently close to, but slightly different from, the original global model.
* **Interpretation:** This demonstrates that the FedEraser algorithm has successfully modified the model by removing the influence of the target client. The slight change in accuracy is expected, as removing one client's data (out of five) should have a noticeable but not drastic impact on the model's overall performance. This validates the effectiveness of the unlearning process.



## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
