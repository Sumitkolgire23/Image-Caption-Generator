import re
from collections import Counter
from pathlib import Path
import nltk
nltk.download('punkt')

from .utils import save_pickle

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v:k for k,v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_caption(text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = self.tokenizer_caption(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized = self.tokenizer_caption(text)
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokenized]


def load_captions(captions_file):
    captions = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            image_id_caption = line.split('\t')
            if len(image_id_caption) != 2:
                # some files use space separation
                image_id_caption = [p.strip() for p in re.split(r'\s+', line, maxsplit=1)]
            image_id = image_id_caption[0].split('#')[0]
            caption = image_id_caption[1]
            captions.setdefault(image_id, []).append(caption)
    return captions


def build_and_save_vocab(captions_file, save_path, freq_threshold=5):
    caps = load_captions(captions_file)
    sentences = []
    for k,v in caps.items():
        for s in v:
            sentences.append(s)
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(sentences)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(vocab, save_path)
    print(f"Vocab saved to {save_path} | size={len(vocab)}")

if __name__ == '__main__':
    import argparse
    from .config import CAPTIONS_FILE, VOCAB_PATH, MIN_WORD_FREQ

    parser = argparse.ArgumentParser()
    parser.add_argument('--captions', default=str(CAPTIONS_FILE))
    parser.add_argument('--out', default=str(VOCAB_PATH))
    parser.add_argument('--min_freq', type=int, default=MIN_WORD_FREQ)
    args = parser.parse_args()

    build_and_save_vocab(args.captions, Path(args.out), freq_threshold=args.min_freq)
