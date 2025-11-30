import torch
from .feature_extractor import EncoderCNN
from .model import DecoderRNN
from .utils import load_pickle
from .config import *
from PIL import Image
import torchvision.transforms as transforms


def load_models(device=DEVICE):
    vocab = load_pickle(VOCAB_PATH)
    encoder = EncoderCNN(EMBED_SIZE)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab))
    encoder.load_state_dict(torch.load(MODEL_ENCODER, map_location=device))
    decoder.load_state_dict(torch.load(MODEL_DECODER, map_location=device))
    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder, vocab


def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def generate_caption(image_path, max_len=20):
    encoder, decoder, vocab = load_models()
    image_tensor = transform_image(image_path)
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample(features, max_len=max_len, vocab=vocab)
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.itos.get(word_id, '<unk>')
        if word == '<end>':
            break
        sampled_caption.append(word)
    caption = ' '.join(sampled_caption)
    return caption


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    print(generate_caption(args.image))
