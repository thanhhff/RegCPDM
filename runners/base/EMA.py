import torch.nn as nn

class EMA:
    def __init__(self, ema_decay):
        super().__init__()
        self.ema_decay = ema_decay
        self.backup = {}
        self.shadow = {}

    def register(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def reset_device(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.shadow[name].to(param.data.device)

    def update(self, current_model: nn.Module, with_decay=True):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if with_decay:
                    new_average = (1.0 - self.ema_decay) * param.data + self.ema_decay * self.shadow[name]
                else:
                    new_average = param.data
                self.shadow[name].copy_(new_average)  # Use copy_ here

    def apply_shadow(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])  # Use copy_ here

    def restore(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])  # Use copy_ here
        self.backup = {}