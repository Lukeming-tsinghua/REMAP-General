import os
import sys
import time
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from data import graph_reader, BertEntityPairDataset, bert_collate_func
from model import *
from args import get_args
from args import print_args
from utils import (AlphaTest, AverageMeter, ModelCheckpoint, Summary, accuracy,
                   set_all_seed)


def train(train_loader,
          model,
          criterion,
          optimizer,
          scheduler,
          accumulate_step,
          epoch,
          figure_writer,
          phase,
          graph):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()
    model.zero_grad()

    end = time.time()
    for i, ((cui1, cui2, sentences, split, entity_1_begin_idxs,
             entity_2_begin_idxs), labels) in enumerate(train_loader):

        data_time.update(time.time() - end)

        sentences = {key: value.to(device) for key, value in sentences.items()}
        labels = labels.to(device)

        score, _ = model(sentences, split, entity_1_begin_idxs,
                         entity_2_begin_idxs)

        one_hot_labels = F.one_hot(labels, graph.relation_num).float()
        loss = criterion(score, one_hot_labels) / accumulate_step

        losses.update(loss.item(), labels.size(0))
        acc.update(accuracy(labels.detach().cpu(), score.detach().cpu()), labels.size(0))

        figure_writer.add_scalar('%s/loss' % phase,
                                 loss.item(),
                                 global_step=epoch * len(train_loader) + i)
        figure_writer.add_scalar('%s/accuracy' % phase,
                                 accuracy(labels.detach().cpu(), score.detach().cpu()),
                                 global_step=epoch * len(train_loader) + i)

        loss.backward()
        if (i % accumulate_step) == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        labels = labels.cpu()
        sentences = {key: value.cpu() for key, value in sentences.items()}
        del labels, sentences

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=acc))


def validate(val_loader, model, criterion, epoch, summary, figure_writer,
             phase, graph):
    losses = AverageMeter()
    pred = []
    label = []
    scores = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(val_loader), ncols=50) as pbar:
            pbar.set_description("Validation iter:")
            for i, ((cui1, cui2, sentences, split, entity_1_begin_idxs,
                     entity_2_begin_idxs), labels) in enumerate(val_loader):

                sentences = {
                    key: value.to(device)
                    for key, value in sentences.items()
                }
                labels = labels.to(device)

                score, _ = model(sentences, split, entity_1_begin_idxs,
                                 entity_2_begin_idxs)
                one_hot_labels = F.one_hot(labels, graph.relation_num).float()
                loss = criterion(score, one_hot_labels)
                preds = score.argmax(dim=1)
                scores.append(score.detach().cpu().numpy())

                pred += list(preds.detach().cpu().numpy())
                label += list(labels.detach().cpu().numpy())

                losses.update(loss.item(), labels.size(0))

                figure_writer.add_scalar('%s/cls_loss' % phase,
                                         loss.item(),
                                         global_step=epoch * len(val_loader) +
                                         i)
                figure_writer.add_scalar('%s/accuracy' % phase,
                                         accuracy(labels.detach(),
                                                  score.detach()),
                                         global_step=epoch * len(val_loader) +
                                         i)

                labels = labels.cpu()
                sentences = {
                    key: value.cpu()
                    for key, value in sentences.items()
                }
                del labels, sentences
                pbar.update(1)
    summary.update(epoch, np.vstack(scores), pred, label, losses.avg)


if __name__ == "__main__":
    args = get_args()
    print_args(args)

    device = torch.device("cuda", args.cuda)
    set_all_seed(args.seed)

    print("loading attribution info...")
    graph = graph_reader(args.path)

    print("loading dataset...")
    train_dataset = BertEntityPairDataset(
        args.path,
        "train",
        graph,
        max_length=args.maxLength,
        tokenizer=args.tokenizer)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.trainBatchSize,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=bert_collate_func,
                                  num_workers=args.nworkers,
                                  pin_memory=args.pinMemory)

    if args.do_eval:
        test_dataset = BertEntityPairDataset(
            args.path,
            "valid",
            graph,
            max_length=args.maxLength,
            tokenizer=args.tokenizer)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.testBatchSize,
                                     shuffle=False,
                                     collate_fn=bert_collate_func,
                                     num_workers=args.nworkers,
                                     pin_memory=args.pinMemory)

    if args.pred:
        pred_dataset = BertEntityPairDataset(
            args.path,
            "test",
            graph,
            max_length=args.maxLength,
            tokenizer=args.tokenizer)

        pred_dataloader = DataLoader(pred_dataset,
                                     batch_size=args.testBatchSize,
                                     shuffle=False,
                                     collate_fn=bert_collate_func,
                                     num_workers=args.nworkers,
                                     pin_memory=args.pinMemory)

    print("building model...")
    model = BertEncoderScore.from_pretrained(args.model, label_num=graph.relation_num).to(device)
    model.resize_token_embeddings(len(train_dataset.tokenizer))

    t_total = len(train_dataloader) * args.epoch
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_rate * t_total,
        num_training_steps=t_total)

    criterion = nn.BCELoss().to(device)

    figure_writer = SummaryWriter(comment="proposal")
    testWriter = Summary(args.result, args.name, "test")
    predWriter = Summary(args.result, args.name, "pred")
    checkpoint = ModelCheckpoint(args.result, args.name)
    

    for e in range(args.epoch):
        if args.do_train:
            train(train_dataloader, model, criterion, optimizer, scheduler,
                  args.accumulate_step, e, figure_writer, "train", graph)
            torch.cuda.empty_cache()
        if args.do_eval:
            validate(test_dataloader,
                     model,
                     criterion,
                     e,
                     testWriter,
                     figure_writer,
                     "test", graph)
        if args.do_predict:
            validate(pred_dataloader,
                     model,
                     criterion,
                     e,
                     predWriter,
                     figure_writer,
                     "annotated", graph)
        modelname = "../pretrained-models/text/scibert-epoch-%d/" % e
        if not os.path.exists(modelname):
            os.makedirs(modelname)
        model.save_pretrained(modelname)

    testWriter.save()
    predWriter.save()
