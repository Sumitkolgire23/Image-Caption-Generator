import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super().__init__()
        self.feature_att = nn.Linear(feature_dim, attention_dim)
        self.hidden_att = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden):
        # features: (batch, feat_dim)
        # hidden: (batch, hidden_dim)
        att1 = self.feature_att(features)
        att2 = self.hidden_att(hidden)
        att = self.full_att(self.relu(att1 + att2))
        alpha = self.softmax(att)
        context = alpha * features
        return context, alpha


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        # prepend features as first 'word'
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None, max_len=20, vocab=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (1,1,hidden)
            outputs = self.linear(hiddens.squeeze(1))    # (1, vocab)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return sampled_ids
        return padded, lengths              