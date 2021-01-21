import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = num_layers, batch_first = True)
        self.fc1 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        batch_size = captions.size(0)
        seq_length = captions.size(1)
        
        embedded_captions = self.embedding(captions)
        print(embedded_captions.size())
        print(features.size())
        out, hidden = self.lstm(embedded_captions, features)
        out = out.view(-1, self.hidden_size)
        
        prob_word = self.fc1(out)
        prob_word = prob_word.view(batch_size, seq_length, -1)
        
        return prob_word, hidden

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass