import torch

class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        self.buffers = {}
        self.collected = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

        for name, b in model.named_buffers():
            self.buffers[name] = b.detach().clone()

    @staticmethod
    def _normalize_name(name):
        return name.replace("module.", "").replace("_orig_mod.", "")

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

        for name, b in model.named_buffers():
            if name in self.buffers:
                self.buffers[name].copy_(b.detach())

    @torch.no_grad()
    def store(self, model):
        self.collected = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.collected[name] = p.detach().clone()

        for name, b in model.named_buffers():
            self.collected[f"__buffer__{name}"] = b.detach().clone()

    @torch.no_grad()
    def copy_to(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name])

        for name, b in model.named_buffers():
            if name in self.buffers:
                b.copy_(self.buffers[name])

    @torch.no_grad()
    def restore(self, model):
        if not self.collected:
            return

        for name, p in model.named_parameters():
            if name in self.collected:
                p.copy_(self.collected[name])

        for name, b in model.named_buffers():
            key = f"__buffer__{name}"
            if key in self.collected:
                b.copy_(self.collected[key])

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
            "buffers": {k: v.detach().cpu() for k, v in self.buffers.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, state, model):
        if not state:
            return

        self.decay = state.get("decay", self.decay)
        device = next(model.parameters()).device

        shadow_by_norm = {self._normalize_name(k): k for k in self.shadow.keys()}
        buffers_by_norm = {self._normalize_name(k): k for k in self.buffers.keys()}

        for name, value in state.get("shadow", {}).items():
            target_name = name if name in self.shadow else shadow_by_norm.get(self._normalize_name(name))
            if target_name is not None:
                self.shadow[target_name].copy_(value.to(device))

        for name, value in state.get("buffers", {}).items():
            target_name = name if name in self.buffers else buffers_by_norm.get(self._normalize_name(name))
            if target_name is not None:
                self.buffers[target_name].copy_(value.to(device))
