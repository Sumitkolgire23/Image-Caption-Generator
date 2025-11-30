# Image Caption Generator — Flickr8k

Instructions:

1. Install dependencies: `pip install -r requirements.txt`
2. Download Flickr8k images and caption file and put them in `data/raw/` as explained in `download_flickr8k.py`.
3. Build vocab: `python -m src.preprocessing`
4. Train: `python -m src.train`
5. Run demo: `streamlit run app.py` or start API: `python api/main.py`

Notes:
- Training requires GPU for reasonable speed.
- This project is a baseline: swap encoder or decoder for transformer-based models (BLIP, ViT+Transformer) to improve quality.
