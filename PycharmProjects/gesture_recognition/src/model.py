import torch
import torch.nn as nn
from typing import Tuple
class GestureLSTM(nn.Module):
    def __init__(
            self,
            input_size: int = 126,
            hidden_size: int = 128,
            num_layers: int = 3,
            num_classes: int = 8,
            dropout: float = 0.3,
            bidirectional: bool = False,
            use_attention: bool = False
    ):
        super(GestureLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LayerNorm(input_size),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_out_size = hidden_size * self.num_directions

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_out_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )

        self.fc1 = nn.Linear(lstm_out_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        if self.use_attention:
            for module in self.attention:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def _fc_layers(self, h: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.ln1(self.fc1(h))))
        out = self.dropout2(self.relu2(self.ln2(self.fc2(out))))
        return self.fc3(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            attention_scores = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_scores, dim=1)
            h_last = torch.sum(lstm_out * attention_weights, dim=1)
        else:
            h_last = lstm_out[:, -1, :]

        return self._fc_layers(h_last)

    def forward_stateful(self, h_last: torch.Tensor) -> torch.Tensor:
        return self._fc_layers(h_last)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model: nn.Module) -> None:
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(model)
    print("=" * 70)
    total, trainable = count_parameters(model)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size:           ~{total * 4 / (1024 ** 2):.2f} MB (float32)")
    print("=" * 70)


if __name__ == "__main__":
    print("Тестирование GestureLSTM модели\n")

    model = GestureLSTM(
        input_size=126,
        hidden_size=128,
        num_layers=3,
        num_classes=8,
        dropout=0.3,
        bidirectional=False,
        use_attention=False
    )

    print_model_summary(model)

    batch_size = 16
    sequence_length = 45
    dummy_input = torch.randn(batch_size, sequence_length, 126)

    model.train()
    output = model(dummy_input)
    print(f"\n[Training mode — с Attention]")
    print(f"Вход:  {dummy_input.shape}")
    print(f"Выход: {output.shape}")

    model.eval()
    h_last = torch.randn(1, 128)
    output_stateful = model.forward_stateful(h_last)
    print(f"\n[Stateful mode — без Attention]")
    print(f"h_last вход:  {h_last.shape}")
    print(f"Выход:        {output_stateful.shape}")

    probs = model.predict_proba(dummy_input)
    print(f"\nВероятности: {probs.shape}")
    print(f"Сумма[0]:    {probs[0].sum().item():.6f}")