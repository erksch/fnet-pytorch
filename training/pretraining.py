import logging
import os
import json
from argparse import ArgumentParser
from datetime import datetime
from time import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from fnet import FNetForPreTraining
from tabulate import tabulate
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from .data_preparation import pretraining_data_gen
from .losses import MLMWeightedCELoss
from .tokenization import get_tokenizer
from .load_config import load_config, print_config

report_frequency = 100
analyize_prediction_frequency = 200
num_warmup_samples = 2_560_000

logging.basicConfig(level=logging.INFO)

def pretraining(config: Dict):
    device = torch.device(f"cuda:{config['gpu_id']}")

    logging.info(f"Loading FNet config {config['fnet_config']}")
    with open(config['fnet_config']) as f:
        fnet_config = json.load(f)

    max_seq_len = fnet_config['max_position_embeddings']

    model = FNetForPreTraining(fnet_config)

    if config['fnet_checkpoint']:
        logging.info(f"Loading FNet pre-training checkpoint {config['fnet_checkpoint']}")
        state_dict = torch.load(config['fnet_checkpoint'], map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    tokenizer = get_tokenizer(config['tokenizer'], max_seq_len)

    optimizer = Adam(model.parameters(), lr=config['learning_rate'], eps=1e-6, weight_decay=0.01)
    warmup_steps = int(num_warmup_samples / config['train_batch_size'])
    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_steps)

    logging.info(f'Scheduled {warmup_steps} warm-up steps ({num_warmup_samples} samples)')

    mlm_criterion = MLMWeightedCELoss()
    nsp_criterion = CrossEntropyLoss()

    log_dir = 'experiments'
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    experiment_name = f"{date_str}_{config['experiment_name']}"
    train_writer = SummaryWriter(os.path.join(log_dir, experiment_name, 'train'))
    eval_writer = SummaryWriter(os.path.join(log_dir, experiment_name, 'eval'))

    train_gen = pretraining_data_gen(tokenizer, config['train_batch_size'], max_seq_len, device)
    eval_gen = pretraining_data_gen(tokenizer, config['eval_batch_size'], max_seq_len, device)

    model.to(device)
    model.train()

    step = 0

    logging.info('Starting training')

    for batch in train_gen:
        step_start = time()
        optimizer.zero_grad()

        pred = model(
            input_ids=batch['input_ids'],
            type_ids=batch['type_ids'],
            mlm_positions=batch['mlm_positions']
        )

        losses = get_loss(batch, pred, mlm_criterion, nsp_criterion)
        loss = losses['loss']
        loss.backward()
        optimizer.step()

        train_writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], step)
        warmup_scheduler.step()

        for name, value in losses.items():
            train_writer.add_scalar(f'Loss/{name}', value.item(), step)

        step_end = time()
        duration = step_end - step_start

        train_writer.add_scalar('StepDuration', duration, step)
        train_writer.add_scalar('StepsPerSecond', 1 / duration, step)

        if step % report_frequency == 0 and step > 0:
            logging.info(f'Step {step}, Loss {loss.item()}')

        if step % analyize_prediction_frequency == 0 and step > 0:
            analyze_prediction(batch, pred, tokenizer)

        if step % config['eval_frequency'] == 0 and step > 0:
            evaluate(model, eval_writer, eval_gen, config, mlm_criterion, nsp_criterion, step, tokenizer)
            export_checkpoint(model, step, experiment_name)

        step += 1


def get_loss(batch, pred, mlm_criterion, nsp_criterion):
    mlm_loss = mlm_criterion(
        pred['mlm_logits'].flatten(0, 1),
        F.one_hot(batch['mlm_ids'].ravel(), pred['mlm_logits'].shape[-1]),
        batch['mlm_weights'].ravel()
    )

    nsp_loss = nsp_criterion(pred['nsp_logits'], batch['nsp_labels'])
    loss = mlm_loss + nsp_loss

    return {
        'mlm_loss': mlm_loss,
        'nsp_loss': nsp_loss,
        'loss': loss
    }


def export_checkpoint(model, step, experiment_name):
    name = f'pretraining_model_step_{step}.statedict.pt'
    checkpoints_dir = os.path.join('exports', experiment_name)
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, name))


def evaluate(model, eval_writer, eval_gen, config, mlm_criterion, nsp_criterion, train_step, tokenizer):
    print('Running evaluation')
    model.eval()
    with torch.no_grad():
        eval_step = 0
        losses, mlm_losses, nsp_losses = 0, 0, 0
        mlm_hits, nsp_hits = 0, 0
        mlm_total, nsp_total = 0, 0

        for batch in eval_gen:
            pred = model(
                input_ids=batch['input_ids'],
                type_ids=batch['type_ids'],
                mlm_positions=batch['mlm_positions']
            )
            metrics = get_loss(batch, pred, mlm_criterion, nsp_criterion)
            hits = get_hits(batch, pred)

            mlm_hits += hits['mlm_hits']
            nsp_hits += hits['nsp_hits']
            mlm_total += hits['mlm_total']
            nsp_total += hits['nsp_total']

            losses += metrics['loss']
            mlm_losses += metrics['mlm_loss']
            nsp_losses += metrics['nsp_loss']

            eval_step += 1

            if eval_step % report_frequency == 0:
                logging.info('Eval Step', eval_step)

            if eval_step % analyize_prediction_frequency:
                analyze_prediction(batch, pred, tokenizer)

            if eval_step >= config['eval_steps']:
                break

        losses = (('MLM', mlm_losses / eval_step), ('NSP', nsp_losses / eval_step), ('Total', losses / eval_step))
        averages = (('MLM', mlm_hits / mlm_total), ('NSP', nsp_hits / nsp_total))

        logging.info(f'Losses: {losses}')
        logging.info(f'Averages: {averages}')

        if eval_writer:
            for name, value in losses:
                eval_writer.add_scalar(f'Loss/{name}', value, train_step)
            for name, value in averages:
                eval_writer.add_scalar(f'Accuracy/{name}', value, train_step)

    model.train()


def get_hits(batch, pred):
    predicted_mlm_ids = pred['mlm_logits'].flatten(0, 1).argmax(-1)
    mlm_hits = torch.sum((predicted_mlm_ids == batch['mlm_ids'].ravel()) * batch['mlm_weights'].ravel()).item()
    nsp_hits = torch.sum(pred['nsp_logits'].argmax(-1) == batch['nsp_labels']).item()

    return {
        'mlm_hits': mlm_hits,
        'mlm_total': batch['mlm_weights'].ravel().sum().item(),
        'nsp_hits': nsp_hits,
        'nsp_total': batch['nsp_labels'].shape[0]
    }


def analyze_prediction(batch, pred, tokenizer):
    logging.info('Printing qualitative result')

    input_ids = batch['input_ids'][0]
    mlm_positions = batch['mlm_positions'][0]
    non_null_positions = mlm_positions[mlm_positions != 0]
    correct_ids = input_ids.detach().clone()
    correct_ids[non_null_positions] = batch['mlm_ids'][0][:len(non_null_positions)]
    predicted_ids = input_ids.detach().clone()
    predicted_ids[non_null_positions] = pred['mlm_logits'][0].argmax(-1)[:len(non_null_positions)]

    print('Input text', '\n', tokenizer.decode(input_ids.cpu().numpy().tolist()), '\n')
    print('Correct text', '\n', tokenizer.decode(correct_ids.cpu().numpy().tolist()), '\n')
    print('Predicted text', '\n', tokenizer.decode(predicted_ids.cpu().numpy().tolist()), '\n')
    print('Type Ids', '\n', batch['type_ids'][0], '\n')

    print(tabulate([(
        tokenizer.decode([input_ids[idx].item()]),
        tokenizer.decode([correct_ids[idx].item()]),
        tokenizer.decode([predicted_ids[idx].item()]),
        'X' if correct_ids[idx] == predicted_ids[idx] else '',
    ) for idx in non_null_positions], tablefmt='grid', headers=('input', 'correct', 'predicted', 'hit')))

    hits = torch.sum(correct_ids[non_null_positions] == predicted_ids[non_null_positions]).item()
    accuracy = hits / len(non_null_positions)

    print(f'{hits} / {len(non_null_positions)} Hits ({accuracy:.2f} accuracy)')

    print('NSP Truth:', batch['nsp_labels'][0].item())
    print('NSP Pred:', pred['nsp_logits'][0].argmax(-1).item())

    print()


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--config', required=True, help='Path to config file')
    args, _ = argparser.parse_known_args()

    config = load_config(args.config)
    print_config(config)

    pretraining(config)
