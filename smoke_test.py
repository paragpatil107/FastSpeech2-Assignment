import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import Dataset
from model import FastSpeech2, ScheduledOptim, FastSpeech2Loss
from utils.model import get_vocoder
from utils.tools import to_device, synth_one_sample

language = sys.argv[1] if len(sys.argv) > 1 else "English"
tag = "en" if language == "English" else "te"

preprocess_config = yaml.load(
    open(f"config/{tag}/preprocess.yaml", "r"), Loader=yaml.FullLoader
)
model_config = yaml.load(open(f"config/{tag}/model.yaml", "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(f"config/{tag}/train.yaml", "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastSpeech2(preprocess_config, model_config).to(device)
model.train()
optimizer = ScheduledOptim(model, train_config, model_config, 0)
amp = train_config["optimizer"].get("amp", False)
scaler = torch.amp.GradScaler("cuda", enabled=amp)
loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)

dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
loader = DataLoader(
    dataset,
    batch_size=train_config["optimizer"]["batch_size"] * 4,
    shuffle=False,
    collate_fn=dataset.collate_fn,
)

batches = []
for group in loader:
    for batch in group:
        batches.append(batch)
        if len(batches) >= 3:
            break
    if len(batches) >= 3:
        break

print(f"{language}: collected {len(batches)} test batches, vocab={model_config.get('vocab_size')}")

vocoder = get_vocoder(model_config, device)

for i, batch in enumerate(batches):
    batch = to_device(batch, device)
    ids = batch[0]

    start = time.time()
    with torch.autocast(device_type="cuda", enabled=amp):
        output = model(*(batch[2:]))
        losses = loss_fn(batch, output)
        scaled_loss = losses[0]
    optimizer.zero_grad()
    scaler.scale(scaled_loss).backward()
    scaler.unscale_(optimizer._optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.update_learning_rate()
    scaler.step(optimizer._optimizer)
    scaler.update()
    torch.cuda.synchronize()

    elapsed = time.time() - start
    mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
    mem_res = torch.cuda.max_memory_reserved() / 1024**3
    print(
        f"batch {i} ({ids[0]}): total={losses[0].item():.4f} mel={losses[1].item():.4f} "
        f"postnet={losses[2].item():.4f} pitch={losses[3].item():.4f} "
        f"energy={losses[4].item():.4f} dur={losses[5].item():.4f} "
        f"step={elapsed:.3f}s vram_alloc={mem_alloc:.2f}GB vram_res={mem_res:.2f}GB"
    )

with torch.no_grad():
    batch = to_device(batches[0], device)
    output = model(*(batch[2:]))
    fig, wav_recon, wav_pred, tag_s = synth_one_sample(
        batch, output, vocoder, model_config, preprocess_config
    )
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    print(f"synth check: recon={wav_recon[0].shape} pred={wav_pred[0].shape} sr={sr}")

print("SMOKE TEST PASSED")
