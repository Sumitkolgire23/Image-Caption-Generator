
---

#      🚀 **Image Caption Generator using Deep Learning**

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-ResNet50-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NLP-LSTM-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-Type-Encoder--Decoder-purple?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=28&duration=3800&color=00E8FF&center=true&vCenter=true&width=900&lines=📸+Deep+Learning+Model+that+Understands+Images!;🤖+Generates+Human-like+Captions+from+Images;🚀+Computer+Vision+%2B+NLP+Hybrid+AI+System;🔥+End-to-End+Image+Caption+Generator+Model"/>
</p>

---

# 🌌 **About the Project**

This project is a **complete pipeline** that allows AI to *see an image and describe it in English*.
It combines **Convolutional Neural Networks (CNN)** for vision and **LSTM networks** for language modeling.

✨ The model understands scenes, objects, and their relationships — and transforms them into meaningful sentences.

---

# 🧠 **System Architecture**

<p align="center">
  <img src="images/systemdiagram.PNG" width="80%" />
</p>

---

# 🔧 **Processing Pipeline**

```
🖼 Image → 🔍 ResNet50 Feature Extractor → 📏 2048-d Vector
        ↓
📝 Caption Preprocessing (tokenization, cleaning, start/end tokens)
        ↓
🧠 LSTM Decoder learns to predict next words
        ↓
🎯 Greedy Search generates final caption
```

---

# 🗄 **Dataset Details**

* **Dataset Used:** Flickr30K
* **Images:** 31,783
* **Captions per Image:** 5
* **Training Process Includes:**

  * Lowercasing
  * Removing non-alphabetic characters
  * Sequence padding
  * Mapping words to indices
  * Vocabulary creation

---

# ✨ **Key Features**

| Feature                         | Description                                                   |
| ------------------------------- | ------------------------------------------------------------- |
| 🔍 **Image Feature Extraction** | ResNet50 pretrained on ImageNet extracts deep visual features |
| ✨ **Text Preprocessing**        | Cleans captions & prepares vocabulary dictionaries            |
| 🧠 **Encoder-Decoder Model**    | Vision encoder + LSTM decoder                                 |
| 🎯 **Greedy Search**            | Selects highest probability words                             |
| 🧪 **BLEU Score**               | Measures caption quality                                      |
| 🖥 **Tkinter GUI**              | Upload an image → get instant caption                         |

---

# 💡 **Advanced Details Added**

### 🧩 Vocabulary Construction

* Creates `word_to_index` and `index_to_word` mappings
* Filters rare words
* Handles unknown tokens

### 🏋️ Training Behavior

* Trains in batches using a **generator function**
* Uses parallel sequences of image features + partial captions
* Uses **categorical cross-entropy** loss

### 📊 Evaluation

* BLEU-1, BLEU-2 scores
* Testing on unseen images
* Visualization of captions

---

# 🛠 **Tech Stack**

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,keras,git,anaconda" />
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

# 🏗 **Training the Model**

### Step 1 — Text Preprocessing

```
run text_data_processing.ipynb
```

### Step 2 — Train CNN+LSTM Model

```
run model_build.ipynb
```

### Step 3 — Live Caption Testing

```
python ui.py
```

---

# 🎆 **Model Results**

<p align="center">
  <img src="images/caption3.JPG" width="45%" />
  <img src="images/caption4.JPG" width="45%" />
</p>

---

# 🖼 **Interactive Desktop UI**

<p align="center">
  <img src="images/ui.JPG" width="70%"/>
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

# 🏁 **How it Works — Summary**

* Image → ResNet50 → feature vector
* Caption → integer tokens
* LSTM predicts next words
* Decoder + Greedy Search → final output sentence

---

# 🔥 **Animated Hero Banner**

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=25&duration=3500&color=FF6AE6&center=true&vCenter=true&width=700&lines=AI+that+Describes+the+World.;From+Pixels+to+Words.;Image+Captioning+Made+Simple.;Powered+by+Deep+Learning."/>
</p>

---

# 👤 **Author**

### **Sumit Kolgire**

AI/ML Engineer | Deep Learning | NLP | Computer Vision
🔗 LinkedIn: [https://www.linkedin.com/in/sumit-kolgire](https://www.linkedin.com/in/sumit-kolgire)

---

# ⭐ **Support the Project**

If this project helped you, consider giving it a **⭐ on GitHub** to support future work!

---
