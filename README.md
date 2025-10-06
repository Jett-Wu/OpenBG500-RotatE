## 运行环境
- 环境：单卡 RTX 4090 24GB
- 框架：Python + PyTorch
- 数据：OpenBG500

## 赛题与数据
- 任务：给定 (h, r) 预测尾实体 t，仅评估 Top-10（主指标 MRR）。
- 获取：下载 `OpenBG500.zip`（约 13MB）或使用 ossutil
- 数据结构：
  - 训练/验证/测试：`OpenBG500_train.tsv`、`OpenBG500_dev.tsv`、`OpenBG500_test.tsv`
  - 文本映射：`OpenBG500_entity2text.tsv`、`OpenBG500_relation2text.tsv`
  - 提交示例：`OpenBG500_example_pred.tsv`
- Baseline：`https://github.com/OpenBGBenchmark/OpenBG500`

## 1. 我的开源实现
- 链接：[`Jett-Wu/OpenBG500-RotatE`](https://github.com/Jett-Wu/OpenBG500-RotatE)
https://github.com/Jett-Wu/OpenBG500-RotatE
 - 运行：`python RotatE/main.py`

— 思考：本数据是结构主导、语义弱化的电商 KG 子集。任务聚焦尾实体预测（多对一属性普遍），避免头实体预测的歧义性；Top-10 评测契合业务召回链路，MRR 能稳定反映全局排序质量。

---

## 2. 入口与工作目录
约束：按项目结构走相对路径，保证跨环境可复现；不要在代码里“写死”绝对路径。
```python
import os
project_root = os.path.dirname(os.path.abspath(".."))
os.chdir(project_root)
print("CWD:", os.getcwd())
```

## 3. 配置
— 设计取舍：
- 显存预算 24GB → `HIDDEN_DIM=1000`、`NEGATIVE_SAMPLE_SIZE=500` 在吞吐与表达力间较稳。
- RotatE 要求 `DOUBLE_ENTITY=True`；`GAMMA=9.0` 与 `embedding_range` 一起稳定数值尺度。
- 测试阶段按全实体枚举（`TEST_BATCH_SIZE=1`）最直接、最鲁棒，便于严格 filtered 排名。
```python
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in globals() else os.path.dirname(os.path.abspath('..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'OpenBG500')
ENTITY_PATH = os.path.join(DATA_DIR, 'OpenBG500_entity2text.tsv')
RELATION_PATH = os.path.join(DATA_DIR, 'OpenBG500_relation2text.tsv')
TRAIN_PATH = os.path.join(DATA_DIR, 'OpenBG500_train.tsv')
DEV_PATH = os.path.join(DATA_DIR, 'OpenBG500_dev.tsv')
TEST_PATH = os.path.join(DATA_DIR, 'OpenBG500_test.tsv')

TRAIN_BATCH_SIZE = 1024
DEV_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1

MODEL = 'RotatE'
DOUBLE_ENTITY = True
DOUBLE_RELATION = False
NEGATIVE_SAMPLE_SIZE = 500
HIDDEN_DIM = 1000
GAMMA = 9.0
ADVERSARIAL = True
ADV_TEMPERATURE = 1.0
LEARNING_RATE = 0.00005
REGULARIZATION = 0.0
CPU_NUM = 10

MAX_STEPS = 100000
VALID_STEPS = 1000
SAVE_CHECKPOINT_STEPS = 2000
LOG_STEPS = 100
TEST_LOG_STEPS = 1000

RESULT_DIR = 'RotatE/result'
```
— 敏感性建议：若内存更充裕，优先增大 `HIDDEN_DIM` 或负采样数；若收敛抖动，适当降低 `ADV_TEMPERATURE` 或增大 `LOG_STEPS` 平滑观测。

## 3. 数据封装（负采样 + filtered）
— 关键点：
- 负采样“先采后滤”提升吞吐，但必须严格从真三元组集合中过滤，避免泄露。
- `BidirectionalOneShotIterator` 头/尾交替，平衡两类梯度信号，训练更稳。
- filtered 排名通过 `filter_bias` 排除已知真三元组，避免“错误惩罚正确”的伪负例。
```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(negative_sample, self.true_head[(relation, tail)], assume_unique=True, invert=True)
            elif self.mode == 'tail-batch':
                mask = np.in1d(negative_sample, self.true_tail[(head, relation)], assume_unique=True, invert=True)
            else:
                raise ValueError('mode not supported')
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        return positive_sample, negative_sample, subsampling_weight, self.mode
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsampling_weight, mode
    @staticmethod
    def count_frequency(triples, start=4):
        count = {}
        for head, relation, tail in triples:
            count[(head, relation)] = count.get((head, relation), start) + 1 if (head, relation) in count else start
            count[(tail, -relation-1)] = count.get((tail, -relation-1), start) + 1 if (tail, -relation-1) in count else start
        return count
    @staticmethod
    def get_true_head_and_tail(triples):
        import numpy as _np
        true_head = {}
        true_tail = {}
        for head, relation, tail in triples:
            true_tail.setdefault((head, relation), []).append(tail)
            true_head.setdefault((relation, tail), []).append(head)
        for relation, tail in true_head:
            true_head[(relation, tail)] = _np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = _np.array(list(set(true_tail[(head, relation)])))
        return true_head, true_tail

class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('mode not supported')
        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))
        return positive_sample, negative_sample, filter_bias, self.mode
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
    def __next__(self):
        self.step += 1
        data = next(self.iterator_head) if self.step % 2 == 0 else next(self.iterator_tail)
        return data
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
```
— 易错点：使用 filtered 评测时，必须基于“训练+验证”汇总构造 `all_true`，否则 bias 不完整，会放大伪负例影响。

## 4. 模型（RotatE）
— 建模直觉：关系在复平面上对应相位旋转，`gamma - distance` 将“旋转后头实体到尾实体的残差”转成得分；双通道（实/虚）增强方向性表达。
```python
import torch.nn as nn
import torch.nn.functional as F

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False)
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model name)
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')
        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
    def forward(self, sample, mode='single'):
        if mode == 'single':
            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)
        elif mode == 'head-batch':
            tail_part, head_part = sample
            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(head_part.size(0), head_part.size(1), -1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(tail_part.size(0), tail_part.size(1), -1)
        else:
            raise ValueError('mode %s not supported' % mode)
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        return score
    def TransE(self, head, relation, tail, mode):
        score = head + (relation - tail) if mode == 'head-batch' else (head + relation) - tail
        return self.gamma.item() - torch.norm(score, p=1, dim=2)
    def DistMult(self, head, relation, tail, mode):
        score = head * (relation * tail) if mode == 'head-batch' else (head * relation) * tail
        return score.sum(dim=2)
    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * rerelation
            score = re_score * re_tail + im_score * im_tail
        return score.sum(dim=2)
    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = rerelation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim=0).norm(dim=0)
        return self.gamma.item() - score.sum(dim=2)
    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)
        score = phase_head + (phase_relation - phase_tail) if mode == 'head-batch' else (phase_head + phase_relation) - phase_tail
        score = torch.sin(score).abs()
        return self.gamma.item() - score.sum(dim=2) * self.modulus
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        if args['cuda']:
            positive_sample = positive_sample.cuda(); negative_sample = negative_sample.cuda(); subsampling_weight = subsampling_weight.cuda()
        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args['negative_adversarial_sampling']:
            negative_score = (F.softmax(negative_score * args['adversarial_temperature'], dim=1).detach() * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
        if args['uni_weight']:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        loss = (positive_sample_loss + negative_sample_loss) / 2
        if args['regularization'] != 0.0:
            regularization = args['regularization'] * (model.entity_embedding.norm(p=3) ** 3 + model.relation_embedding.norm(p=3).norm(p=3) ** 3)
            loss = loss + regularization
        loss.backward(); optimizer.step()
        return {'positive_sample_loss': positive_sample_loss.item(), 'negative_sample_loss': negative_sample_loss.item(), 'loss': loss.item()}
```
— 调参提示：`ADV_TEMPERATURE` 越高越聚焦“难负例”，但训练方差也更大；若不稳定，降低温度或开启梯度裁剪。

## 5. 训练与提交（filtered ranking）
— 工程建议：
- I/O：测试阶段是全实体打分，注意 SSD 带宽与 DataLoader 线程配置（`CPU_NUM//2`）匹配。
- 可靠性：提交前用一小段测试样本抽查 Top-10 映射是否合理（实体 id → 文本是否一致）。
```python
import json, time
from datetime import datetime
import tqdm

def load_openbg500():
    with open(ENTITY_PATH, 'r', encoding='utf-8') as fp:
        lines = [line.strip('\n').split('\t') for line in fp.readlines()]
    ent2id = {line[0]: i for i, line in enumerate(lines)}
    id2ent = {i: line[0] for i, line in enumerate(lines)}
    with open(RELATION_PATH, 'r', encoding='utf-8') as fp:
        lines = [line.strip('\n').split('\t') for line in fp.readlines()]
    rel2id = {line[0]: i for i, line in enumerate(lines)}
    def read_triples(path):
        with open(path, 'r', encoding='utf-8') as fp:
            data = [line.strip('\n').split('\t') for line in fp.readlines()]
        return [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in data]
    train_triples = read_triples(TRAIN_PATH)
    dev_triples = read_triples(DEV_PATH)
    with open(TEST_PATH, 'r', encoding='utf-8') as fp:
        test_pairs = [line.strip('\n').split('\t') for line in fp.readlines()]
    test_triples_dummy = [(ent2id[h], rel2id[r], 0) for h, r in test_pairs]
    return ent2id, rel2id, id2ent, train_triples, dev_triples, test_pairs, test_triples_dummy

def build_model(nentity, nrelation):
    return KGEModel(
        model_name=MODEL,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=HIDDEN_DIM,
        gamma=GAMMA,
        double_entity_embedding=DOUBLE_ENTITY,
        double_relation_embedding=DOUBLE_RELATION
    )

def train_only(model, nentity, nrelation, train_triples):
    train_dataloader_head = DataLoader(TrainDataset(train_triples, nentity, nrelation, NEGATIVE_SAMPLE_SIZE, 'head-batch'), batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=max(1, CPU_NUM//2), collate_fn=TrainDataset.collate_fn)
    train_dataloader_tail = DataLoader(TrainDataset(train_triples, nentity, nrelation, NEGATIVE_SAMPLE_SIZE, 'tail-batch'), batch size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=max(1, CPU_NUM//2), collate_fn=TrainDataset.collate_fn)
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    args = {
        'cuda': torch.cuda.is_available(),
        'negative_adversarial_sampling': ADVERSARIAL,
        'adversarial_temperature': ADV_TEMPERATURE,
        'uni_weight': True,
        'regularization': REGULARIZATION
    }
    latest_state = None
    train_logs = []
    for step in range(MAX_STEPS):
        log = KGEModel.train_step(model, optimizer, train_iterator, args)
        if step % LOG_STEPS == 0:
            train_logs.append({'step': int(step), 'loss': float(log['loss']), 'positive_sample_loss': float(log['positive_sample_loss']), 'negative_sample_loss': float(log['negative_sample_loss'])})
            print(f"[Step {step:>5}/{MAX_STEPS:<5}] loss={log['loss']:.6f} pos={log['positive_sample_loss']:.6f} neg={log['negative_sample_loss']:.6f}")
        latest_state = model.state_dict()
    return model, latest_state, train_logs

def export_submission(model, id2ent, test_pairs, nentity, nrelation, train_triples, dev_triples, run_dir):
    all_true = train_triples + dev_triples
    test_loader = DataLoader(TestDataset(test_triples_dummy, all_true, nentity, nrelation, 'tail-batch'), batch_size=TEST_BATCH_SIZE, num_workers=max(1, CPU_NUM//2), collate_fn=TestDataset.collate_fn)
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
    predict_all = []
    for idx_list in top10_all:
        for eid in idx_list:
            predict_all.append(id2ent[eid])
    sub_path = os.path.join(run_dir, 'OpenBG500_test.tsv')
    with open(sub_path, 'w', encoding='utf-8') as f:
        for i in range(len(test_pairs)):
            row = test_pairs[i]
            preds = predict_all[i*10:i*10+10]
            list_obj = [x + '\t' for x in row] + [x + '\n' if j == 9 else x + '\t' for j, x in enumerate(preds)]
            f.writelines(list_obj)
    print('提交文件已保存:', sub_path)

start_time = time.time(); start_iso = datetime.now().isoformat()
ent2id, rel2id, id2ent, train_triples, dev_triples, test_pairs, test_triples_dummy = load_openbg500()
nentity, nrelation = len(ent2id), len(rel2id)
print('='*80); print('RotatE Training | OpenBG500'); print('-'*80)
print(f"train_batch_size: {TRAIN_BATCH_SIZE}")
print(f"test_batch_size : {TEST_BATCH_SIZE}")
print(f"max_steps      : {MAX_STEPS}")
print(f"lr            : {LEARNING_RATE}")
print(f"hidden_dim    : {HIDDEN_DIM}")
print(f"gamma         : {GAMMA}")
print(f"double_entity : {DOUBLE_ENTITY}")
print(f"double_relation: {DOUBLE_RELATION}")
print('='*80)

model = build_model(nentity, nrelation)
model, latest_state, train_logs = train_only(model, nentity, nrelation, train_triples)

end_time = time.time(); end_iso = datetime.now().isoformat(); elapsed_sec = end time
run_dir = os.path.join(RESULT_DIR, time.strftime('%Y%m%d_%H%M%S'))
os.makedirs(run_dir, exist_ok=True)

if latest_state is not None:
    torch.save(latest_state, os.path.join(run_dir, 'rotate_latest.pth'))
training_params = {
    'MODEL': MODEL,
    'DOUBLE_ENTITY': DOUBLE_ENTITY,
    'DOUBLE_RELATION': DOUBLE_RELATION,
    'NEGATIVE_SAMPLE_SIZE': NEGATIVE_SAMPLE_SIZE,
    'HIDDEN_DIM': HIDDEN_DIM,
    'GAMMA': GAMMA,
    'ADVERSARIAL': ADVERSARIAL,
    'ADV_TEMPERATURE': ADV_TEMPERATURE,
    'LEARNING_RATE': LEARNING_RATE,
    'REGULARIZATION': REGULARIZATION,
    'CPU_NUM': CPU_NUM,
    'TRAIN_BATCH_SIZE': TRAIN_BATCH_SIZE,
    'TEST_BATCH_SIZE': TEST_BATCH_SIZE,
    'MAX_STEPS': MAX_STEPS,
    'VALID_STEPS': VALID_STEPS,
    'SAVE_CHECKPOINT_STEPS': SAVE_CHECKPOINT_STEPS,
    'LOG_STEPS': LOG_STEPS,
    'TEST_LOG_STEPS': TEST_LOG_STEPS,
    'NUM_ENTITIES': nentity,
    'NUM_RELATIONS': nrelation
}
with open(os.path.join(run_dir, 'training_params.json'), 'w', encoding='utf-8') as fp:
    json.dump(training_params, fp, ensure_ascii=False, indent=2)
training_time = {'start_time': start_iso, 'end_time': end_iso, 'elapsed_seconds': elapsed_sec}
with open(os.path.join(run_dir, 'training_time.json'), 'w', encoding='utf-8') as fp:
    json.dump(training_time, fp, ensure_ascii=False, indent=2)
if train_logs:
    with open(os.path.join(run_dir, 'train_logs.jsonl'), 'w', encoding='utf-8') as fp:
        for rec in train_logs:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

export_submission(model, id2ent, test_pairs, nentity, nrelation, train_triples, dev_triples, run_dir)
```
— 指标解读：MRR 强调“首个正确答案”的排名质量；若 Hits@10 高、Hits@1 偏低，说明整体候选合理但“尖锐度”不足，可通过更强负采样或小幅增大维度改进。

## 6. 结果检查（可选）
```python
from pathlib import Path
runs = sorted(Path('RotatE/result').glob('*'))
if runs:
    latest = runs[-1]
    print('输出目录:', latest)
    for fn in ['OpenBG500_test.tsv', 'rotate_latest.pth']:
        p = latest / fn
        print(fn, '->', '存在' if p.exists() else '缺失')
    sub = latest / 'OpenBG500_test.tsv'
    if sub.exists():
        with open(sub, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 2:
                    print(line.strip())
                else:
                    break
```

## 7. 成绩与结语
- 线上 MRR：57.65。
- 实战建议：
  - 资源充足：先增维（1000→1200/1500），再增负采样；
  - 不稳定：降温度或加权平均日志步长；
  - 上线前：抽检 Top-10 映射、核对提交格式，确保可追责与复核。