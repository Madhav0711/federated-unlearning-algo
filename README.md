# Federated Unlearning for Medical Imaging using FedEraser

A B.Tech Project implementing federated learning and client-level machine unlearning with enhanced privacy for medical datasets.

## Project Overview

This project explores how to remove a specific client's contribution from a Federated Learning (FL) model without retraining from scratch.
We implement a full FL pipeline and integrate a custom influence-vector-based unlearning mechanism, enhanced with Differential Privacy–style noise masking to obscure which client was removed.

This approach aligns with privacy regulations like GDPR’s right to be forgotten and with practical needs such as mitigating malicious or corrupted client data.

---

## Team and Guide

This project was developed as part of our B.Tech curriculum under the valuable guidance of **Dr. Lal Upendra**.

**Team Members:**
* Madhav Nagpal
* Dudhat Hemil
* Janmej Rana

---

## Key Concepts 

* **Federated Learning (FL):** A decentralized machine learning approach where multiple clients collaboratively train a model without sharing their local data. Each client trains a model on its own data, and the updates are aggregated on a central server to create a global model. This preserves data privacy.

* **Machine Unlearning:** The process of removing the influence of a specific subset of training data from a trained machine learning model. This is essential for user privacy and model maintenance.

* **FedEraser:** An efficient unlearning algorithm for FL. It works by approximating the removal of a client's contribution by applying a "reverse" update to the global model. This is achieved by using stored model checkpoints from the training history to perform a gradient ascent step, effectively canceling out the client's original gradient descent update.

* **Differential Privacy (DP):** A technique that injects controlled noise into model updates or removal operations to mask sensitive information. In this project, DP-inspired Gaussian noise is added to the client’s influence vector during unlearning so that an external observer cannot infer which client was removed, enhancing privacy and preventing reverse engineering of the forgotten client’s contribution.
---

## Dataset 

1. **COVID-19 Radiography Database**.

* **Source:** [COVID-19 Radiography Database on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
* **Content:** The dataset contains chest X-ray images for two classes: COVID-19 and Normal.

2. **ISIC-2019 Skin Lesion Classification Dataset**
* **Content:** containing four lesion classes: 
MEL, BCC → labelled cancerous
NV, BKL → labelled non-cancerous


### Preprocessing: The images are converted to grayscale, resized to 128x128 pixels, and normalized before being fed into the model.
---

## Methodology and Implementation

### **1. Federated Learning Setup**

**Architecture:** Modified **ResNet50**
- First convolution modified for **1-channel grayscale** input  
- Final linear layer changed to **2-class output**

**Federated Configuration:**
- **5 simulated clients**
- **FedAvg** for global aggregation
- **7 local epochs** per client per round  

**Checkpointing (per round):**
- `global_checkpoints/` — global model state before sending to clients  
- `client_checkpoints/` — each client's local model after training  

This enables full reconstruction of the entire FL training history.

---

### **2. Client Influence Extraction**
For each client:
1. Load all round-wise client model snapshots  
2. Convert each snapshot to a **flat parameter vector**  
3. Compute the client’s **influence vector** by summing changes across rounds  
4. **Clip** the influence vector to control its magnitude  

This vector approximates how much each client contributed to the final global model.

---

### **3. Differential Privacy Noise Masking**

To hide which client was removed and enhance privacy:

- Sample Gaussian noise with **σ ∈ [0, 0.03]**
- Add noise to the client’s influence vector  
- Subtract this noised vector from the global weights  

This prevents an external observer from deducing **which specific client** was unlearned.

---

### **4. Model Reconstruction & Evaluation**

For each noise level σ:
1. Rebuild a global model from the modified parameter vector  
2. Evaluate its accuracy on the test set  
3. Generate **accuracy-vs-σ plots** for each client  

This reveals the stability and the **privacy–utility tradeoff** of the unlearning process.

---

### **5. Post-Unlearning Accuracy Recovery**

After subtracting a client’s influence:
- Run **additional FL rounds** with the remaining clients  
- Allow the model to regain accuracy  
- Maintain the forgotten client’s removal throughout  

This mirrors the “repair rounds” used in advanced unlearning algorithms.

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


## Results

The system produces several key outputs that highlight the effectiveness of the unlearning pipeline:

### **🔹 Per-Client Accuracy vs σ Plots**
For each client, the model is reconstructed after subtracting a noised influence vector.  
Plots visualize how **test accuracy changes** as the Gaussian noise level (σ) increases, illustrating the privacy–utility tradeoff.

### **🔹 Recovered Global Accuracy After Unlearning**
After a client is removed, the model may lose some accuracy.  
Running a few additional FL rounds with the remaining clients helps the model **recover performance** while keeping the removed client’s influence erased.

---

### **Key Observations**
- **Influence removal changes global accuracy slightly**, as expected when reversing one client's updates.  
- **Differential Privacy noise successfully masks which client was removed**, preventing external observers from identifying the forgotten client.  
- **Partial retraining restores most of the model’s predictive performance**, demonstrating practical unlearning without full retraining.

---

These results confirm that the pipeline achieves **effective, privacy-preserving client-level unlearning** while maintaining overall model utility.




## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
