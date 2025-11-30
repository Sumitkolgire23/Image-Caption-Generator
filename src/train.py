import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import *
from .utils import set_seed, save_pickle
from .preprocessing import load_captions
from .dataset import FlickrDataset, collate_fn
from .feature_extractor import EncoderCNN
from .model import DecoderRNN
from .preprocessing import Vocabulary
from pathlib import Path

set_seed(SEED)


def train():
    # load vocab
    from .utils import load_pickle
    vocab = load_pickle(VOCAB_PATH)

    dataset = FlickrDataset(IMAGES_DIR, CAPTIONS_FILE, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    encoder = EncoderCNN(EMBED_SIZE).to(DEVICE)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), num_layers=NUM_LAYERS, drop_prob=DROP_OUT).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        encoder.train()
        decoder.train()
        epoch_loss = 0
        for images, captions, lengths in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            features = encoder(images)
            targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

            outputs = decoder(features, captions, lengths)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader):.4f}")
        Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), MODEL_ENCODER)
        torch.save(decoder.state_dict(), MODEL_DECODER)


if __name__ == '__main__':
    train()
