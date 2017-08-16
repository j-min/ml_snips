import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from skip_connections import Highway

class Embedding(nn.Module):
    def __init__(self, word_vocab_size, word_embed_size=100, char_embedding=False, char_vocab_size=26,
                 char_embed_size=8, n_channel=100, kernel_sizes=[5], positional_encoding=False):
        
        super(Embedding, self).__init__()
        
        self.word_embed_size = word_embed_size
        
        self.char_embedding = char_embedding
        
        self.positional_encoding = positional_encoding
        
        self.word_embedding = nn.Embedding(word_vocab_size, self.word_embed_size)
        
        if self.char_embedding:                        
            self.char_embed_size = char_embed_size
            
            self.char_vec_dim = n_channel * len(kernel_sizes)
            
            self.embed_size = self.word_embed_size + self.char_vec_dim
            
            self.char_embedding = nn.Embedding(char_vocab_size, self.char_embed_size)

            self.char_cnn = nn.ModuleList([nn.Conv2d(1, n_channel, [width, self.char_embed_size]) for width in kernel_sizes])

            # 2-layer Highway Networks
            self.highway = Highway(self.embed_size, num_layers=2)
        
        else:
            self.embed_size = self.word_embed_size

        if self.positional_encoding:
            self.positional_embed = self.make_positional_encoding(self.embed_size, max_seq_len=1000)
            if torch.cuda.is_available():
                self.positional_embed.cuda()
                
    def load_pretrained(self, pretrained_word_embedding=None, pretrained_char_embeding=None):
        if pretrained_word_embed:
            self.word_embedding.weight = pretrained_word_embedding
            
        if pretrained_char_embed:
            self.char_embedding.weight = pretrained_char_embedding
            
    def word_embed(self, word_indices):
        word_vector = self.word_embedding(word_indices)
        return word_vector
    
    def char_embed(self, char_indices):
        """word vector of dimension `n_channel * len(kernel_sizes)` from character embedding"""
        batch_size, max_seq_len, max_word_len = char_indices.size()
        
        # [batch_size * max_seq_len,    max_word_len]
        char_indices = char_indices.view(-1, max_word_len)
        
        # [batch_size * max_seq_len,    max_word_len, char_embed_size]
        char_vectors = self.char_embedding(char_indices)
        
        # [batch_size * max_seq_len, 1, max_word_len, char_embed_size]
        char_vectors = char_vectors.unsqueeze(1)
        
        # Extract most important feature (the maximum value) for a given filter
        word_vectors = []
        for char_cnn in self.char_cnn:
            
            # [batch_size * max_seq_len, n_channel, len_after_conv, 1]
            char_vectors = char_cnn(char_vectors)
            
            # [batch_size * max_seq_len, n_channel, len_after_conv]
            char_vectors = char_vectors.squeeze(-1)
            
            # [batch_size * max_seq_len, n_channel]
            word_vector = F.max_pool1d(char_vectors, char_vectors.size(-1)).squeeze(-1)
            
            word_vectors.append(word_vector)
            
        # [batch_size * max_seq_len, n_channel * len(kernel_sizes)]
        word_vector = torch.cat(word_vectors, dim=-1)
        
        # [batch_size, max_seq_len, n_channel * len(kernel_sizes)]
        # char_vec_dim = n_channel * len(kernel_sizes)
        word_vector = word_vector.view(batch_size, max_seq_len, -1)

        return word_vector
        
    def make_positional_encoding(self, embed_size, max_seq_len):
        """ 'Attention-is-all-you-need' style positional encoding. """
        # [max_seq_len, embed_size]
        pe = torch.arange(0, max_seq_len).unsqueeze(1).expand(max_seq_len, embed_size)
        
        # [embed_size]
        div_term = torch.pow(10000, torch.arange(0, embed_size * 2, 2) / embed_size)
        
        # [max_seq_len, embed_size]
        pe /= div_term
        
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        
        return pe

    def forward(self, word_indices, char_indices=None):
        """
        Apply Word-level (+ Character-level / postional) Embedding
        
        1) Apply Word embedding => word_embed_size
        2) Apply Character embedding (+prefix/suffix & padding up to max_word_len) => char_embed_size
        3) Apply Char-CNN on Character embedding => char_vec_size (= number of convolutional filters due to max pooling)
        4) Concatenate [Word embedding; CNN outputs] => word_embed_size + char_embed_size
        5) Apply 2-layer Highway networks => word_embed_size + char_embed_size = embed_size
        6) Apply Positional Encoding (not concatenation)
        
        Final embedding size = word_embed_size + char_embed_size
        
        Args:
            word_indices: [batch_size, max_seq_len]
            char_indices: [batch_size, max_seq_len, max_word_len]
        Return:
            word vectors: [batch_size, max_seq_len, embed_size]
        """ 
        batch_size, max_seq_len = word_indices.size()
        
        if self.char_embedding:
            assert word_indices.size() == char_indices.size()[:2], \
            'Word indices and character indices must have the same sizes, but {} and {} found'.format(
                word_indices.size(), char_indices.size())
            
            # [batch_size * max_seq_len, word_embed_size]
            word_vector = self.word_embed(word_indices)
            
            # [batch_size, max_seq_len, char_vec_dim]
            word_vector_from_char_embedding = self.char_embed(char_indices)
            
            # [batch_size, max_seq_len, embed_size]
            word_vector = self.highway(torch.cat([word_vector, word_vector_from_char_embedding], dim=-1))
            
        else:
            # [batch_size * max_seq_len, embed_size]
            word_vector = self.word_embed(word_indices)
            
        if self.positional_encoding:
            # [max_seq_len, embed_size]
            positional_vector = Variable(self.positional_embed[:max_seq_len])
            
            word_vector += positional_vector

        return word_vector
