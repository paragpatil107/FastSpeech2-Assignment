import argparse
import os
import time

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    amp = train_config["optimizer"].get("amp", False)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    vocoder = get_vocoder(model_config, device)

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)
    step_log_path = os.path.join(
        train_config["path"]["log_path"], "step_losses.csv"
    )
    with open(step_log_path, "a") as f:
        f.write(
            "step,total_loss,mel_loss,mel_postnet_loss,pitch_loss,energy_loss,duration_loss\n"
        )

    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                with torch.autocast(device_type="cuda", enabled=amp):
                    output = model(*(batch[2:]))
                    losses = loss_fn(batch, output)
                    scaled_loss = losses[0] / grad_acc_step
                scaler.scale(scaled_loss).backward()
                if step % grad_acc_step == 0:
                    scaler.unscale_(optimizer._optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.update_learning_rate()
                    scaler.step(optimizer._optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                    with open(step_log_path, "a") as f:
                        f.write(
                            "{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                                step, *losses
                            )
                        )

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                            "scheduled_optim": {
                                "current_step": optimizer.current_step
                            },
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
