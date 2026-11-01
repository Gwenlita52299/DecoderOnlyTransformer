# %% Cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position =torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)

        pe[:,0::2] = torch.sin(position * div_term) #偶数维sin
        pe[:,1::2] = torch.cos(position * div_term) #奇数维cos

        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0),:]
    
class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        
        self.row_dim = 0
        self.col_dim = 1
    
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim,dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores   
    
class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        super().__init__()

        self.we = nn.Embedding(num_embeddings=num_tokens,
                               embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model=d_model,
                                     max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0)), device=token_ids.device))
        mask = mask == 0

        self_attention_values = self.self_attention(position_encoded,
                                                    position_encoded,
                                                    position_encoded,
                                                    mask=mask)
        
        residual_connection_values = position_encoded + self_attention_values

        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_tokens, labels =batch
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss


# %% Cell 2
token_to_id = {'what': 0,
               'is': 1,
               'statquest': 2,
               'awesome': 3,
               '<EOS>': 4,
               }
id_to_token = dict(map(reversed, token_to_id.items()))
inputs = torch.tensor([[
    token_to_id['what'],
    token_to_id['is'],
    token_to_id['statquest'],
    token_to_id['<EOS>'],
    token_to_id['awesome']],

    [token_to_id['statquest'],
    token_to_id['is'],
    token_to_id['what'],
    token_to_id['<EOS>'],
    token_to_id['awesome']]])

labels = torch.tensor([[
    token_to_id['is'],
    token_to_id['statquest'],
    token_to_id['<EOS>'],
    token_to_id['awesome'],
    token_to_id['<EOS>']],

    [token_to_id['is'],
    token_to_id['what'],
    token_to_id['<EOS>'],
    token_to_id['awesome'],
    token_to_id['<EOS>']]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)


# %% Cell 3
print(id_to_token)


# %% Cell 4
L.seed_everything(seed=42)
model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6)


# %% Cell 5
def predict(model, prompt_tokens, max_length=6):
    """Greedy decode: feed the prompt, keep appending argmax tokens until <EOS> or max_length."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model_input = torch.tensor([token_to_id[t] for t in prompt_tokens], device=device)
    predicted_ids = []
    with torch.no_grad():
        for _ in range(model_input.size(dim=0), max_length + 1):
            predictions = model(model_input)
            predicted_id = torch.argmax(predictions[-1, :]).view(1)
            predicted_ids.append(predicted_id.item())
            if predicted_id.item() == token_to_id['<EOS>']:
                break
            model_input = torch.cat((model_input, predicted_id))
    model.train(was_training)
    return [id_to_token[i] for i in predicted_ids]


prompts = [
    (['what', 'is', 'statquest', '<EOS>'], ['awesome', '<EOS>']),
    (['statquest', 'is', 'what', '<EOS>'], ['awesome', '<EOS>']),
]


def evaluate(model, title):
    print(f"\n===== {title} =====")
    all_ok = True
    for prompt, expected in prompts:
        got = predict(model, prompt)
        ok = got == expected
        all_ok &= ok
        print(f"{' '.join(prompt):<30} -> {' '.join(got):<20} expected: {' '.join(expected):<15} {'PASS' if ok else 'FAIL'}")
    return all_ok


evaluate(model, "Before training")


# %% Cell 6
trainer = L.Trainer(max_epochs=30, enable_progress_bar=False)
trainer.fit(model, train_dataloaders=dataloader)
print(f"\nFinal train_loss: {trainer.callback_metrics['train_loss'].item():.4f}")

passed = evaluate(model, "After training")
print("\nRESULT:", "ALL PASS" if passed else "SOME FAILED")
