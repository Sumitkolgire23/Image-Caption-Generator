from Image_Caption_Generator.src.config import CAPTIONS_FILE, IMAGES_DIR
import nltk
from nltk.translate.bleu_score import corpus_bleu
from .preprocessing import load_captions
from .inference import generate_caption


def evaluate_on_sample(n_refs=5, sample_images=None):
    # sample_images: list of full image paths
    # For each image, we have multiple ground-truth captions in Flickr8k
    refs = []
    hyps = []
    caps = load_captions(CAPTIONS_FILE)
    if sample_images is None:
        sample_images = list(caps.keys())[:100]
    for img in sample_images:
        gt = caps[img]
        gt_tokens = [nltk.word_tokenize(s.lower()) for s in gt]
        preds = generate_caption(str(IMAGES_DIR / img))
        pred_tokens = nltk.word_tokenize(preds.lower())
        refs.append(gt_tokens)
        hyps.append(pred_tokens)
    score = corpus_bleu(refs, hyps)
    print('BLEU score:', score)
    return score

if __name__ == '__main__':
    evaluate_on_sample()
