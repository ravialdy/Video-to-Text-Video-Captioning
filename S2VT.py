from unicodedata import bidirectional
import torch
from torch import nn
import torch.nn.functional as F

class S2VT_Baseline(nn.Module):

    """S2VT Baseline Model.

    Parameters
    ----------
    vocab_size : int
        The size of MSVD vocabulary using word tokenization. 
    feature_dim : int
        The dimension of extracted features.
    length : int
        The length of the time steps.
    hidden_dim : int
        The number of hidden dimensions used.
    embedding_dim : int
        The number of embedding dimensions used.
    dropout_lstm : float
        The value of probability of an element to be zeroed implemented in the LSTM layer.
    p_dropout : float
        The value of probability of an element to be zeroed used in the dropout layer.

    Attributes
    ----------
    lstm_encoder : nn.LSTM
        The first LSTM layer to encode the input sequence.
    lstm_decoder : nn.LSTM
        The second LSTM layer to decode and generate a sequence of output.
    dropout : nn.Dropout
        Dropout layer.
    fc1 : nn.Linear
        Linear layer that maps the feature dimentions to hidden dimentions.
    fc2 : nn.Linear
        Linear layer that maps the hidden dimentions to size of the vocabulary.
    embedding : nn.Embedding
        Embedding layer.         
    """

    def __init__(self, vocab_size, feature_dim, length, hidden_dim=500, embedding_dim=500, dropout_lstm=0, p_dropout=0.3, num_layers=1):
        super(S2VT_Baseline, self).__init__()

        self.lstm_encoder = nn.LSTM(input_size = hidden_dim, hidden_size = hidden_dim, batch_first=True, num_layers=num_layers,
                                 dropout=dropout_lstm).cuda()

        self.lstm_decoder = nn.LSTM(input_size = hidden_dim + embedding_dim, hidden_size = hidden_dim, batch_first=True, num_layers=num_layers,
                                 dropout=dropout_lstm).cuda()

        self.dropout = nn.Dropout(p=p_dropout)
        self.fc1 = nn.Linear(in_features=feature_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.feature_dim = feature_dim
        self.length = length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

    def forward(self, features, mask_labels=None, mode='train'):
        """
        :features (batch_size, length, feature_dim)
        :mask_labels (batch_size, length - 1, 1)
        :return:
        """

        batch_size = features.shape[0]

        features = self.fc1(features.cuda()) # (batch_size, length, hidden_dim)

        padding = torch.zeros([batch_size, self.length - 1, self.hidden_dim]).cuda() # (batch_size, length - 1, hidden_dim)
        pad_features = torch.cat([features, padding], dim=1) # (batch_size, 2*length - 1, hidden_dim)

        # LSTM Encoder
        output_enc, state_enc = self.lstm_encoder(pad_features) # (batch_size, 2*length - 1, hidden_dim)

        if mode == 'train':

            label_embeddings = self.embedding(mask_labels.cuda()) # (batch_size, length - 1, embedding_dim)
            padding = torch.zeros([batch_size, self.length, self.embedding_dim]).cuda() # (batch_size, length, embedding_dim)
            pad_embedding = torch.cat([padding, label_embeddings], dim=1) # (batch_size, 2*length - 1, embedding_dim)

            input_dec = torch.cat([pad_embedding, output_enc], dim=2) # (batch_size, 2*length - 1, hidden_dim + embedding_dim)

            # LSTM Decoder
            output_dec, state_dec = self.lstm_decoder(input_dec) # (batch_size, 2*length - 1, hidden_dim)
            result = output_dec[:, self.length:, :] # (batch_size, length - 1, hidden_dim)
            result = self.dropout(result) # (batch_size, length - 1, hidden_dim)
            result = self.fc2(result) # (batch_size, length - 1, vocab_size)
            return result

        elif mode == 'test':
            
            # Encoding stage in Testing phase
            padding = torch.zeros([batch_size, self.length, self.embedding_dim]).cuda() # (batch_size, length, embedding_dim)
            input_dec = torch.cat([padding, output_enc[:, :self.length, :]], dim=2) # (batch_size, length, 2*hidden_dim + embedding_dim)
            _, state_dec = self.lstm_decoder(input_dec) # (batch_size, length, hidden_dim)

            # Decoding stage in Testing phase

            #tag"<BOS>" to start decoding its current hidden representation into a sequence of words
            bos_idx = 3
            start_embed = (bos_idx * torch.ones([batch_size], dtype=torch.long)).cuda() # (batch_size)
            start_embed = self.embedding(start_embed).unsqueeze(dim=1) # (batch_size, 1, embedding_dim)
            input_start = torch.cat([start_embed, output_enc[:, self.length, :].unsqueeze(dim=1)], dim=2) # (batch_size, 1, 2*hidden_dim + embedding_dim)

            output_dec, state_dec = self.lstm_decoder(input_start, state_dec) # (batch_size, 1, hidden_dim)
            word_pred = self.fc2(output_dec.squeeze(dim=1)) # (batch_size, vocab_size)
            word_pred = torch.argmax(word_pred, dim=1) # (batch_size)
            predicted_words = [word_pred] 
            for i in range(self.length - 2):
                input_dec = self.embedding(word_pred.unsqueeze(1)) # (batch_size, 1, embedding_dim)
                input_dec = torch.cat([input_dec, output_enc[:, self.length + i + 1, :].unsqueeze(dim=1)], dim=2) # (batch_size, 1, 2*hidden_dim + embedding_dim)
                output_dec, state_dec = self.lstm_decoder(input_dec, state_dec) # (batch_size, 1, hidden_dim)
                word_pred = self.fc2(output_dec.squeeze(dim=1)) # (batch_size, vocab_size)
                word_pred = torch.argmax(word_pred, dim=1) # (batch_size)
                predicted_words.append(word_pred) 
            predicted_words = torch.cat(predicted_words, dim=0).view(self.length - 1, batch_size) # (length - 1, batch_size)
            return predicted_words.transpose(dim0=0, dim1=1) # (batch_size, length - 1)

# # Debungging
# if __name__ == '__main__':   
#     model_s2vt = S2VT_Baseline(vocab_size=100000, feature_dim=4096, length=80, hidden_dim=1000, embedding_dim=500,
#                             dropout_lstm=0, p_dropout=0.3).cuda()
#     feats = torch.randn(10, 80, 4096)
#     mask_labels = torch.randint(0, 10200, (10, 79))
#     print(model_s2vt(feats, mask_labels=mask_labels[:, :-1], mode='train').shape)