import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
from torchmetrics.regression import PearsonCorrCoef

def evaluate(config, model, eval_dataloader, weight=None):
    with torch.no_grad():
        preds, labels, filenames = predict(config, model, eval_dataloader, weight=weight)
        r = PearsonCorrCoef(num_outputs=6)
        r = r(1000*preds, 1000*labels)
        r = r.mean()
    return r.cpu().numpy(), preds, filenames


def predict(train_config, model, dataloader, weight=None):
    model.eval()
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    r = PearsonCorrCoef(num_outputs=6).cuda()
    preds = []
    labels = []
    filenames = []
    with torch.no_grad():
        for batch in bar:
            audio, vit, length, text, label, filename, vision_missing, text_missing = batch[:8]

            with autocast():
                audio = {key: val.to(train_config.device) for key, val in audio.items()}
                vit = vit.to(train_config.device)
                label = label.to(train_config.device)
                text = {key: val.to(train_config.device) for key, val in text.items()}
                vision_missing = vision_missing.to(train_config.device)
                text_missing = text_missing.to(train_config.device)
                output = model(audio, vit, text, length,
                               vision_missing=vision_missing, text_missing=text_missing)

                # Handle gate-intensity output
                if isinstance(output, tuple):
                    pred = output[0]  # pred = gate * intensity
                else:
                    pred = output

                if train_config.mtl:
                    pred = weight[0]/weight.sum() * pred[0] + weight[1]/weight.sum() * pred[1]

            labels.append(label.detach().cpu())
            preds.append(pred.to(torch.float32).detach().cpu())
            filenames.extend(filename)
            r.update(1000*pred, 1000*label)
            bar.set_postfix(ordered_dict={'corr': f'{r.compute().mean().cpu().numpy():.4f}'})

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

    if train_config.verbose:
        bar.close()

    return preds, labels, filenames
