
---

# 🚀 **Image Caption Generator using Deep Learning**

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-ResNet50-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NLP-LSTM-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=28&duration=4000&color=00F7FF&center=true&vCenter=true&width=900&lines=📸+AI+that+Understands+Images+and+Describes+Them!;🤖+Combining+Computer+Vision+%2B+NLP+to+Generate+Captions;🚀+Deep+Learning+based+End-to-End+Image+Captioning+System" />
</p>

---

## 🌟 **Project Overview**

This project automatically **generates English captions** for input images using:

* **ResNet50** for feature extraction
* **LSTM-based Encoder–Decoder** for caption generation
* **Flickr30k Dataset** for training
* **Greedy Search** for inference

The system combines **Computer Vision** + **Natural Language Processing** to make machines *describe* what they see.

---

## 🧠 **Architecture**

<p align="center">
  <img src="https://github.com/yourusername/yourrepo/raw/main/images/systemdiagram.png" width="80%" />
</p>

### 🔧 Workflow

```
Input Image → ResNet50 → Feature Vector (2048-d)
                         ↓
                  LSTM Decoder
                         ↓
                Generated Caption
```

---

## 📂 **Dataset Info**

✔ Flickr30k Dataset (30,000 images)
✔ Each image contains **5 human-written captions**
✔ Captions cleaned + tokenized
✔ Special tokens added: `<start>` and `<end>`

---

# ✨ **Features**

| Feature                         | Description                                   |
| ------------------------------- | --------------------------------------------- |
| 🔍 **Image Feature Extraction** | ResNet50 pretrained on ImageNet               |
| ✏️ **Caption Preprocessing**    | Cleaning, lowercasing, removing special chars |
| 🧠 **Sequence Modeling**        | LSTM model trained to predict next word       |
| 🚀 **Inference**                | Greedy Search for final caption               |
| 🧪 **Evaluation**               | BLEU Score                                    |
| 🖥️ **Desktop UI**              | Full Tkinter-based testing interface          |

---

## 🛠 **Tech Stack**

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,anaconda,git" />
</p>

---

# 📦 **Installation**

```bash
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator
pip install -r requirements.txt
```

---

# 🧹 **Data Cleaning Example**

```python
def clean(text):
    text = text.lower()
    text = re.sub("[^a-z]+", " ", text)
    return text
```

---

# 🏗 **Model Training**

### **Step 1 — Preprocess Text**

```
run text_data_processing.ipynb
```

### **Step 2 — Train the Model**

```
run model_build.ipynb
```

### **Step 3 — Test with UI**

```
python ui.py
```

---

# 🔥 **Results**

### 🖼 Example Output

<p align="center">
  <img src="https://github.com/yourusername/yourrepo/raw/main/images/caption3.JPG" width="45%" />
  <img src="https://github.com/yourusername/yourrepo/raw/main/images/caption4.JPG" width="45%" />
</p>

---

# 🖼 **Live Captioning UI**

<p align="center">
  <img src="https://github.com/yourusername/yourrepo/raw/main/images/ui.JPG" width="70%"/>
</p>

---

# 📁 **Project Structure**

```
📦 Image Caption Generator
 ┣ 📂 data
 ┃ ┣ 📂 Images
 ┃ ┗ 📂 textFiles
 ┣ 📂 model_checkpoints
 ┣ 📂 images
 ┣ 📜 text_data_processing.ipynb
 ┣ 📜 model_build.ipynb
 ┣ 📜 ui.py
 ┣ 📜 README.md
 ┗ 📜 requirements.txt
```

---

# 🏁 **How the System Works (Summary)**

1. Image sent through **ResNet50 CNN**
2. Last layer removed → produces **2048-dimension vector**
3. Vector + caption tokens passed to LSTM
4. LSTM predicts next word probabilities
5. Highest probability word selected (Greedy Search)
6. Final caption generated

---

# 🧪 **BLEU Evaluation**

BLEU score is used to measure similarity between generated and real captions.

---

# 🧑‍💻 **Author**

### 👤 *Sumit Kolgire (Shadow)*

🚀 AI/ML Engineer | Deep Learning | NLP | Computer Vision
🔗 [LinkedIn](https://www.linkedin.com/in/sumit-kolgire)

---

# ⭐ **Support**

If you like this project, give it a **star ⭐ on GitHub** — it motivates further development!

---


