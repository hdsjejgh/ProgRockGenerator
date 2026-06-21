import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Decoder Only Transformer Stuff
# gotten from Cameron R Wolfe's blog (thank you Cameron R. Wolfe)

class CausalSelfAttention(nn.Module):
    def __init__(self,
                 EMBEDDING_DIM,
                 HEADS,
                 MAX_TOKENS,
                 LAYERS,
                 BIAS=False,
                 DROPOUT=0.2):
        super().__init__()
        assert EMBEDDING_DIM % HEADS == 0

        self.c_attn = nn.Linear(EMBEDDING_DIM, 3 * EMBEDDING_DIM, bias = BIAS) #Concatenated matrices for key, query, and value
        self.c_proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=BIAS)

        self.HEADS = HEADS
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.dropout = nn.Dropout(DROPOUT)

        #Causal buffer
        self.register_buffer("mask", torch.tril(torch.ones(MAX_TOKENS, MAX_TOKENS))
                             .view(1, 1, MAX_TOKENS, MAX_TOKENS))

    def forward(self, x):
        BATCH_SIZE, SEQ_LENGTH, _ = x.size()

        #gets query, key, and value matrices
        q,k,v = self.c_attn(x).split(self.EMBEDDING_DIM, dim=2)

        #splits each matrix into the different heads
        k = k.view(BATCH_SIZE, SEQ_LENGTH, self.HEADS, self.EMBEDDING_DIM // self.HEADS).transpose(1,2)
        q = q.view(BATCH_SIZE, SEQ_LENGTH, self.HEADS, self.EMBEDDING_DIM // self.HEADS).transpose(1, 2)
        v = v.view(BATCH_SIZE, SEQ_LENGTH, self.HEADS, self.EMBEDDING_DIM // self.HEADS).transpose(1, 2)

        #gets attention matrix and performs dropour
        att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:SEQ_LENGTH,:SEQ_LENGTH] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        #gets the final attentioned vectors
        y = att @ v

        y = y.transpose(1,2).contiguous().view(BATCH_SIZE,SEQ_LENGTH,self.EMBEDDING_DIM)
        y = self.dropout(self.c_proj(y))
        return y


class FFNN(nn.Module): #feed forward neural network
    def __init__(self,
                 EMBEDDING_DIM,
                 BIAS=False,
                 DROPOUT=0.2
                 ):
        super().__init__()
        self.c_fc = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM*4, BIAS)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(EMBEDDING_DIM * 4, EMBEDDING_DIM, BIAS)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self,
                 EMBEDDING_DIM,
                 HEADS,
                 MAX_TOKENS,
                 VOCAB_SIZE,
                 LAYERS,
                 BIAS=False,
                 DROPOUT=0.2):
        super().__init__()
        self.ln_1 = nn.LayerNorm(EMBEDDING_DIM)
        self.attn = CausalSelfAttention(EMBEDDING_DIM,HEADS,MAX_TOKENS,LAYERS,BIAS,DROPOUT)
        self.ln_2 = nn.LayerNorm(EMBEDDING_DIM)
        self.ffnn = FFNN(EMBEDDING_DIM,BIAS,DROPOUT)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffnn(self.ln_2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self,
                 EMBEDDING_DIM,
                 HEADS,
                 MAX_TOKENS,
                 VOCAB_SIZE,
                 LAYERS,
                 BIAS=False,
                 DROPOUT=0.2):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(VOCAB_SIZE,EMBEDDING_DIM), #token embeddings
            pos_emb = nn.Embedding(MAX_TOKENS,EMBEDDING_DIM), #positional embeddings
            drop = nn.Dropout(DROPOUT),
            blocks = nn.ModuleList([Block(EMBEDDING_DIM, HEADS, MAX_TOKENS, BIAS, DROPOUT) for _ in range(LAYERS)]),
            ln_f = nn.LayerNorm(EMBEDDING_DIM),
            head = nn.Linear(EMBEDDING_DIM,VOCAB_SIZE, bias=BIAS), #token classification head
        ))

    def forward(self, idx, targets=None):
        device = idx.device
        _, SEQ_LENGTH = idx.size()
        pos = torch.arange(0,SEQ_LENGTH, dtype=torch.long, device=device)

        tok_emb = self.transformer.tok_emb(idx)
        pos_emb = self.transformer.pos_emb(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            #gets loss if target values are given
            logits = self.transformer.head(x)
            # print(logits.size())
            # loss = F.cross_entropy(
            #     logits.view(-1, logits.size(-1)),
            #     targets.view(-1),
            #     ignore_index=-1,
            # )
        else:
            #only look at last token if making inference
            logits = self.transformer.head(x[:, [-1], :])
            loss = None

        return logits#, loss



