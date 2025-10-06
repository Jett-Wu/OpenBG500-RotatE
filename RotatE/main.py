import os
import json
import time
import copy
from datetime import datetime
import torch
import numpy as np
import tqdm

import config as CFG
from torch.utils.data import DataLoader
from dataloader import TrainDataset, TestDataset, BidirectionalOneShotIterator
from model import KGEModel


def load_openbg500():
    with open(CFG.ENTITY_PATH, 'r', encoding='utf-8') as fp:
        lines = [line.strip('\n').split('\t') for line in fp.readlines()]
    ent2id = {line[0]: i for i, line in enumerate(lines)}
    id2ent = {i: line[0] for i, line in enumerate(lines)}

    with open(CFG.RELATION_PATH, 'r', encoding='utf-8') as fp:
        lines = [line.strip('\n').split('\t') for line in fp.readlines()]
    rel2id = {line[0]: i for i, line in enumerate(lines)}

    def read_triples(path):
        with open(path, 'r', encoding='utf-8') as fp:
            data = [line.strip('\n').split('\t') for line in fp.readlines()]
        return [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in data]

    train_triples = read_triples(CFG.TRAIN_PATH)
    dev_triples = read_triples(CFG.DEV_PATH)

    with open(CFG.TEST_PATH, 'r', encoding='utf-8') as fp:
        test_pairs = [line.strip('\n').split('\t') for line in fp.readlines()]
    test_triples_dummy = [(ent2id[h], rel2id[r], 0) for h, r in test_pairs]

    return ent2id, rel2id, id2ent, train_triples, dev_triples, test_pairs, test_triples_dummy


def build_model(nentity, nrelation):
    model = KGEModel(
        model_name=CFG.MODEL,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=CFG.HORIZONTAL if hasattr(CFG, 'HORIZONTAL') else CFG.HIDDEN_DIM,
        gamma=CFG.GAMMA,
        double_entity_embedding=CFG.DOUBLE_ENTITY,
        double_relation_embedding=CFG.DOUBLE_RELATION
    )
    return model


def train_and_validate(model, nentity, nrelation, train_triples, dev_triples):
    train_dataloader_head = DataLoader(
        TrainDataset(train_triples, nentity, nrelation, CFG.NEGATIVE_SAMPLE_SIZE, 'head-batch'),
        batch_size=CFG.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=max(1, CFG.CPU_NUM//2),
        collate_fn=TrainDataset.collate_fn
    )
    train_dataloader_tail = DataLoader(
        TrainDataset(train_triples, nentity, nrelation, CFG.NEGATIVE_SAMPLE_SIZE, 'tail-batch'),
        batch_size=CFG.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=max(1, CFG.CPU_NUM//2),
        collate_fn=TrainDataset.collate_fn
    )
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.LEARNING_RATE)

    args = {
        'cuda': torch.cuda.is_available(),
        'negative_adversarial_sampling': CFG.ADVERSARIAL,
        'adversarial_temperature': CFG.ADV_TEMPERATURE,
        'uni_weight': True,
        'regularization': CFG.REGULARIZATION
    }

    best_state = None
    latest_state = None
    best_mrr = 0.0
    train_logs = []
    dev_logs = []

    for step in range(CFG.MAX_STEPS):
        log = KGEModel.train_step(model, optimizer, train_iterator, args)
        if step % CFG.LOG_STEPS == 0:
            record = {
                'step': int(step),
                'loss': float(log['loss']),
                'positive_sample_loss': float(log['positive_sample_loss']),
                'negative_sample_loss': float(log['negative_sample_loss'])
            }
            train_logs.append(record)
            print(f"[Step {step:>5}/{CFG.MAX_STEPS:<5}] loss={log['loss']:.6f} pos={log['positive_sample_loss']:.6f} neg={log['negative_sample_loss']:.6f}")

        if CFG.VALIDATION and step % CFG.VALID_STEPS == 0 and step > 0:
            # Evaluate on dev using filtered ranking like RotatE-Raw
            from model import KGEModel as _K
            from dataloader import TestDataset as _TD
            all_true = train_triples + dev_triples
            test_loader_head = DataLoader(_TD(dev_triples, all_true, nentity, nrelation, 'head-batch'), batch_size=CFG.DEV_BATCH_SIZE, num_workers=max(1, CFG.CPU_NUM//2), collate_fn=_TD.collate_fn)
            test_loader_tail = DataLoader(_TD(dev_triples, all_true, nentity, nrelation, 'tail-batch'), batch_size=CFG.DEV_BATCH_SIZE, num_workers=max(1, CFG.CPU_NUM//2), collate_fn=_TD.collate_fn)
            logs = []
            with torch.no_grad():
                for mode_loader in [test_loader_head, test_loader_tail]:
                    for pos, neg, bias, mode in mode_loader:
                        if args['cuda']:
                            pos, neg, bias = pos.cuda(), neg.cuda(), bias.cuda()
                        score = model((pos, neg), mode) + bias
                        argsort = torch.argsort(score, dim=1, descending=True)
                        if mode == 'head-batch':
                            positive_arg = pos[:, 0]
                        else:
                            positive_arg = pos[:, 2]
                        for i in range(pos.size(0)):
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })
            metrics = {k: sum(d[k] for d in logs) / len(logs) for k in logs[0].keys()}
            dev_logs.append({'step': int(step), **{k: float(v) for k, v in metrics.items()}})
            latest_state = copy.deepcopy(model.state_dict())
            if metrics['MRR'] >= best_mrr:
                best_mrr = metrics['MRR']
                best_state = copy.deepcopy(model.state_dict())
            print(f"Dev: MRR={metrics['MRR']:.6f} H@1={metrics['HITS@1']:.6f} H@3={metrics['HITS@3']:.6f} H@10={metrics['HITS@10']:.6f}")

    if best_state is None:
        best_state = latest_state if latest_state is not None else copy.deepcopy(model.state_dict())
    return model, best_state, train_logs, dev_logs


def predict_submission(model, id2ent, test_pairs, nentity, nrelation, train_triples, dev_triples):
    # Build all_true for filter bias
    all_true = train_triples + dev_triples
    # Convert test pairs into evaluation triples with every candidate tail
    from dataloader import TestDataset as _TD
    from torch.utils.data import DataLoader

    if torch.cuda.is_available():
        model = model.cuda()

    # We will compute for each (h,r) the argsort over all tails and take top10
    # Reuse test_step style
    results = []
    with torch.no_grad():
        for h_str, r_str in test_pairs:
            # Build a single triple with dummy tail; _TD will expand candidates internally
            # But _TD expects ids; we need mapping outside, so we build a small dataset:
            pass

    # Instead of per-sample loop (slow), batch over all pairs
    # Prepare synthetic triples for loader usage
    # Create triples with dummy tails ids=0 sequentially
    # We'll create datasets for both head-batch and tail-batch; for test, use tail-batch ranking
    return results


def main():
    ent2id, rel2id, id2ent, train_triples, dev_triples, test_pairs, test_triples_dummy = load_openbg500()
    nentity = len(ent2id)
    nrelation = len(rel2id)

    print('=' * 80)
    print('RotatE Training | OpenBG500')
    print('-' * 80)
    print(f"train_batch_size: {CFG.TRAIN_BATCH_SIZE}")
    print(f"dev_batch_size  : {CFG.DEV_BATCH_SIZE}")
    print(f"test_batch_size : {CFG.TEST_BATCH_SIZE}")
    print(f"max_steps      : {CFG.MAX_STEPS}")
    print(f"lr            : {CFG.LEARNING_RATE}")
    print(f"hidden_dim    : {CFG.HIDDEN_DIM}")
    print(f"gamma         : {CFG.GAMMA}")
    print(f"double_entity : {CFG.DOUBLE_ENTITY}")
    print(f"double_relation: {CFG.DOUBLE_RELATION}")
    print('=' * 80)

    start_time = time.time()
    start_iso = datetime.now().isoformat()

    model = build_model(nentity, nrelation)
    model, best_state, train_logs, dev_logs = train_and_validate(model, nentity, nrelation, train_triples, dev_triples)

    end_time = time.time()
    end_iso = datetime.now().isoformat()
    elapsed_sec = end_time - start_time

    run_dir = os.path.join(CFG.RESULT_DIR, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    torch.save(best_state, os.path.join(run_dir, 'rotate_best.pth'))
    torch.save(model.state_dict(), os.path.join(run_dir, 'rotate_latest.pth'))

    # Save training params and timing
    training_params = {
        'MODEL': CFG.MODEL,
        'DOUBLE_ENTITY': CFG.DOUBLE_ENTITY,
        'DOUBLE_RELATION': CFG.DOUBLE_RELATION,
        'NEGATIVE_SAMPLE_SIZE': CFG.NEGATIVE_SAMPLE_SIZE,
        'HIDDEN_DIM': CFG.HIDDEN_DIM,
        'GAMMA': CFG.GAMMA,
        'ADVERSARIAL': CFG.ADVERSARIAL,
        'ADV_TEMPERATURE': CFG.ADV_TEMPERATURE,
        'LEARNING_RATE': CFG.LEARNING_RATE,
        'REGULARIZATION': CFG.REGULARIZATION,
        'CPU_NUM': CFG.CPU_NUM,
        'TRAIN_BATCH_SIZE': CFG.TRAIN_BATCH_SIZE,
        'DEV_BATCH_SIZE': CFG.DEV_BATCH_SIZE,
        'TEST_BATCH_SIZE': CFG.TEST_BATCH_SIZE,
        'MAX_STEPS': CFG.MAX_STEPS,
        'VALIDATION': CFG.VALIDATION,
        'VALID_STEPS': CFG.VALID_STEPS,
        'SAVE_CHECKPOINT_STEPS': CFG.SAVE_CHECKPOINT_STEPS,
        'LOG_STEPS': CFG.LOG_STEPS,
        'TEST_LOG_STEPS': CFG.TEST_LOG_STEPS,
        'NUM_ENTITIES': nentity,
        'NUM_RELATIONS': nrelation
    }
    with open(os.path.join(run_dir, 'training_params.json'), 'w', encoding='utf-8') as fp:
        json.dump(training_params, fp, ensure_ascii=False, indent=2)

    training_time = {
        'start_time': start_iso,
        'end_time': end_iso,
        'elapsed_seconds': elapsed_sec
    }
    with open(os.path.join(run_dir, 'training_time.json'), 'w', encoding='utf-8') as fp:
        json.dump(training_time, fp, ensure_ascii=False, indent=2)

    # Persist logs
    if train_logs:
        with open(os.path.join(run_dir, 'train_logs.jsonl'), 'w', encoding='utf-8') as fp:
            for rec in train_logs:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if dev_logs:
        with open(os.path.join(run_dir, 'dev_logs.jsonl'), 'w', encoding='utf-8') as fp:
            for rec in dev_logs:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Finetune on train+dev
    if getattr(CFG, 'FINETUNE_AFTER_TRAIN', False):
        model.load_state_dict(best_state)
        mix_triples = train_triples + dev_triples
        train_dataloader_head = DataLoader(
            TrainDataset(mix_triples, nentity, nrelation, CFG.NEGATIVE_SAMPLE_SIZE, 'head-batch'),
            batch_size=CFG.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=max(1, CFG.CPU_NUM//2),
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(mix_triples, nentity, nrelation, CFG.NEGATIVE_SAMPLE_SIZE, 'tail-batch'),
            batch_size=CFG.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=max(1, CFG.CPU_NUM//2),
            collate_fn=TrainDataset.collate_fn
        )
        iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.FINETUNE_LR)
        args = {
            'cuda': torch.cuda.is_available(),
            'negative_adversarial_sampling': CFG.ADVERSARIAL,
            'adversarial_temperature': CFG.ADV_TEMPERATURE,
            'uni_weight': True,
            'regularization': CFG.REGULARIZATION
        }
        if torch.cuda.is_available():
            model = model.cuda()
        finetune_logs = []
        for step in range(CFG.FINETUNE_EPOCHS * 100):
            log = KGEModel.train_step(model, optimizer, iterator, args)
            if step % max(1, CFG.LOG_STEPS) == 0:
                print(f"[FT Step {step}] loss={log['loss']:.6f}")
                finetune_logs.append({'step': int(step), 'loss': float(log['loss'])})
        torch.save(model.state_dict(), os.path.join(run_dir, 'rotate_finetuned.pth'))
        if finetune_logs:
            with open(os.path.join(run_dir, 'finetune_logs.jsonl'), 'w', encoding='utf-8') as fp:
                for rec in finetune_logs:
                    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Build ranking for test to top10 predictions
    # Use filtered evaluation procedure to get ranking indices
    all_true = train_triples + dev_triples
    test_loader = DataLoader(TestDataset(test_triples_dummy, all_true, nentity, nrelation, 'tail-batch'), batch_size=CFG.TEST_BATCH_SIZE, num_workers=max(1, CFG.CPU_NUM//2), collate_fn=TestDataset.collate_fn)
    if torch.cuda.is_available():
        model = model.cuda()
    top10_all = []
    with torch.no_grad():
        for pos, neg, bias, mode in tqdm.tqdm(test_loader):
            if torch.cuda.is_available():
                pos, neg, bias = pos.cuda(), neg.cuda(), bias.cuda()
            score = model((pos, neg), mode) + bias
            argsort = torch.argsort(score, dim=1, descending=True)
            top10 = argsort[:, :10].cpu().numpy().tolist()
            top10_all.extend(top10)

    # Map to entity strings and write submission
    predict_all = []
    for idx_list in top10_all:
        for eid in idx_list:
            predict_all.append(id2ent[eid])

    with open(os.path.join(run_dir, 'OpenBG500_test.tsv'), 'w', encoding='utf-8') as f:
        for i in range(len(test_pairs)):
            row = test_pairs[i]
            preds = predict_all[i*10:i*10+10]
            list_obj = [x + '\t' for x in row] + [x + '\n' if j == 9 else x + '\t' for j, x in enumerate(preds)]
            f.writelines(list_obj)
    print(f"Submission saved: {os.path.join(run_dir, 'OpenBG500_test.tsv')}")


if __name__ == '__main__':
    main()


