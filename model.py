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
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = num_layers, batch_first = True)
        self.fc1 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        #Put all the captions except the last word in embedding shape: (10, 11, 256) 
        embedded_captions = self.embedding(captions[:, :-1])
        batch_size = captions.size(0)
        
        #Add the features a another part of the sequence length so caption_features.shape == (10, 12, 256)
        features = features.unsqueeze(1)
        caption_features = torch.cat((features, embedded_captions), 1)
        
        out, _ = self.lstm(caption_features)
        
        prob_word = self.fc1(out)
        prob_word = prob_word.view(batch_size, -1, self.vocab_size)
        
        return prob_word

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pred_caption = list()
        states = (torch.randn(1, 1, self.hidden_size).to(inputs.device), 
                  torch.randn(1, 1, self.hidden_size).to(inputs.device))
        for index in range(max_len):
            vocab_hidden_size, states = self.lstm(inputs, states)
            vocab_prob = self.fc1(vocab_hidden_size)
            _, pred_word = torch.topk(vocab_prob, 1)
            pred_caption.append(int(pred_word))
            
            inputs = self.embedding(pred_word).squeeze(0)
            
         
        return pred_caption
            
            
         
        
        
        
        
        
        
        
        
        