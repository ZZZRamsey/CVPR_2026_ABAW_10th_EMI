import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchmetrics.regression import PearsonCorrCoef

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    model.train()
    losses = AverageMeter()
    blackimgs = AverageMeter()
    corr = PearsonCorrCoef(num_outputs=6).to(train_config.device)

    time.sleep(0.1)
    optimizer.zero_grad(set_to_none=True)
    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    for batch in bar:
        audio, vision, length, text, label, avg, vision_missing, text_missing = batch[:8]
        # Optional: sample_weights at position 8
        sample_weights = batch[8] if len(batch) > 8 else None

        if scaler:
            with autocast():
                audio = {key: val.to(train_config.device) for key, val in audio.items()}
                vision = vision.to(train_config.device)
                label = label.to(train_config.device)
                text = {key: val.to(train_config.device) for key, val in text.items()}
                vision_missing = vision_missing.to(train_config.device)
                text_missing = text_missing.to(train_config.device)
                if sample_weights is not None:
                    sample_weights = sample_weights.to(train_config.device)

                features = model(audio, vision, text, length,
                                 vision_missing=vision_missing, text_missing=text_missing)

                if train_config.use_gate_intensity:
                    loss = loss_function(features, label, sample_weights=sample_weights)
                    pred_for_corr = features[0]
                elif train_config.mtl:
                    loss_mtl = loss_function(features, label)
                    loss = loss_mtl['loss']
                    pred_for_corr = features
                else:
                    if sample_weights is not None:
                        loss = loss_function(features, label, sample_weights=sample_weights)
                    else:
                        loss = loss_function(features, label)
                    pred_for_corr = features

                # Scale loss for gradient accumulation
                loss_scaled = loss / train_config.gradient_accumulation
                losses.update(loss.item())

            scaler.scale(loss_scaled).backward()

            if step % train_config.gradient_accumulation == 0:
                if train_config.clip_grad:
                    scaler.unscale_(optimizer)
                    if train_config.mtl:
                        torch.nn.utils.clip_grad_value_(list(model.parameters()) + list(loss_function.parameters()), train_config.clip_grad)
                    else:
                        torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if train_config.scheduler in ("polynomial", "cosine", "constant"):
                    scheduler.step()

        else:
            audio = {key: val.to(train_config.device) for key, val in audio.items()}
            text = {key: val.to(train_config.device) for key, val in text.items()}
            vision = vision.to(train_config.device)
            label = label.to(train_config.device)
            vision_missing = vision_missing.to(train_config.device)
            text_missing = text_missing.to(train_config.device)
            if sample_weights is not None:
                sample_weights = sample_weights.to(train_config.device)

            features = model(audio, vision, text, length,
                             vision_missing=vision_missing, text_missing=text_missing)

            if train_config.use_gate_intensity:
                loss = loss_function(features, label, sample_weights=sample_weights)
                pred_for_corr = features[0]
            elif train_config.mtl:
                loss_mtl = loss_function(features, label)
                loss = loss_mtl['loss']
                pred_for_corr = features
            else:
                if sample_weights is not None:
                    loss = loss_function(features, label, sample_weights=sample_weights)
                else:
                    loss = loss_function(features, label)
                pred_for_corr = features

            loss_scaled = loss / train_config.gradient_accumulation
            losses.update(loss.item())
            loss_scaled.backward()

            if step % train_config.gradient_accumulation == 0:
                if train_config.clip_grad:
                    if train_config.mtl:
                        torch.nn.utils.clip_grad_value_(list(model.parameters()) + list(loss_function.parameters()), train_config.clip_grad)
                    else:
                        torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if train_config.scheduler in ("polynomial", "cosine", "constant"):
                    scheduler.step()

        if train_config.verbose:
            blackimgs.update(avg)
            if train_config.mtl:
                corr.update((loss_mtl['task_weight']/loss_mtl['task_weight'].sum())[0] * features[0] + (loss_mtl['task_weight']/loss_mtl['task_weight'].sum())[1] * features[1], label)
            else:
                corr.update(pred_for_corr, label)
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr']),
                       "cutoff": "{:.4f}".format(blackimgs.avg),
                       "corr": "{:.4f}".format(corr.compute().cpu().numpy().mean())}
            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    if train_config.mtl:
        print('task_loss:        {:.3f}, {:.3f}'.format(*list(loss_mtl['task_loss'])))
        print('task_weight:      {:.3f}, {:.3f}'.format(*list(loss_mtl['task_weight'])))
        print('task_uncertainty: {:.3f}, {:.3f}'.format(*list(loss_mtl['task_uncertainty'])))

    return losses.avg
