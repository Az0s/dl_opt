# =====================================================
# 真实数据集实验代码
# 使用 WikiText-2, WikiText-103, LAMBADA 等标准数据集
# =====================================================

import torch
import torch.nn as nn
import torch. nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import os
from tqdm import tqdm
from typing import Optional, List, Dict
from dataclasses import dataclass
import json

# =====================================================
# Part 1: 数据集加载
# =====================================================

class WikiTextDataset(Dataset):
    """WikiText-2/103 数据集"""
    def __init__(self, split:  str = 'train', seq_len: int = 512, 
                 dataset_name: str = 'wikitext-2-raw-v1'):
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
        
        print(f"加载数据集: {dataset_name}, split={split}")
        
        # 加载数据
        dataset = load_dataset('wikitext', dataset_name, split=split)
        
        # 加载tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 将所有文本拼接并tokenize
        all_text = '\n'.join([item['text'] for item in dataset if item['text']. strip()])
        self.tokens = self.tokenizer. encode(all_text)
        
        self.seq_len = seq_len
        self.vocab_size = self.tokenizer.vocab_size
        
        # 计算样本数
        self.num_samples = (len(self.tokens) - 1) // seq_len
        print(f"总token数: {len(self.tokens)}, 样本数: {self. num_samples}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        tokens = self.tokens[start:end]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class LAMBADADataset(Dataset):
    """LAMBADA 数据集 - 测试长距离依赖理解能力"""
    def __init__(self, split: str = 'test', max_len: int = 512):
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
        
        print(f"加载 LAMBADA 数据集, split={split}")
        dataset = load_dataset('lambada', split=split)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.samples = []
        for item in dataset:
            text = item['text']
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_len:
                self. samples.append(tokens)
        
        self.max_len = max_len
        self.vocab_size = self.tokenizer.vocab_size
        print(f"有效样本数: {len(self. samples)}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # 补齐到固定长度
        if len(tokens) < self.max_len:
            padding = [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            tokens = padding + tokens  # 左padding
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class PG19Dataset(Dataset):
    """PG-19 数据集 - 长文本语言建模"""
    def __init__(self, split: str = 'train', seq_len: int = 2048, max_books: int = 100):
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
        
        print(f"加载 PG-19 数据集, split={split}")
        dataset = load_dataset('pg19', split=split)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer. eos_token
        
        # 只加载部分书籍以节省时间
        all_tokens = []
        for i, item in enumerate(tqdm(dataset, desc="Tokenizing")):
            if i >= max_books:
                break
            tokens = self.tokenizer.encode(item['text'])
            all_tokens.extend(tokens)
        
        self.tokens = all_tokens
        self.seq_len = seq_len
        self.vocab_size = self.tokenizer.vocab_size
        self.num_samples = (len(self.tokens) - 1) // seq_len
        print(f"总token���: {len(self. tokens)}, 样本数: {self.num_samples}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        tokens = self.tokens[start:end]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# =====================================================
# Part 2: 合成任务数据集 (更详细版本)
# =====================================================

class InductionHeadDataset(Dataset):
    """Induction Head 测试数据集
    任务: [A][B].. .[A] -> [B]
    测试模型是否能学会 in-context 的归纳模式
    """
    def __init__(self, vocab_size: int = 1000, seq_len: int = 256, 
                 num_samples: int = 10000, num_patterns: int = 3):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_patterns = num_patterns
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = torch.randint(10, self.vocab_size, (self. seq_len,))
        
        # 插入多个 [A][B]...[A] 模式
        for _ in range(self.num_patterns):
            a = torch.randint(10, self.vocab_size, (1,)).item()
            b = torch. randint(10, self.vocab_size, (1,)).item()
            
            # 第一次出现位置
            pos1 = torch.randint(5, self.seq_len // 3, (1,)).item()
            seq[pos1] = a
            seq[pos1 + 1] = b
            
            # 第二次出现位置 (需要预测)
            pos2 = torch.randint(self.seq_len // 2, self.seq_len - 2, (1,)).item()
            seq[pos2] = a
            seq[pos2 + 1] = b
        
        return seq[:-1], seq[1:]


class AssociativeRecallDataset(Dataset):
    """联想记忆测试数据集
    任务: 给定 key-value 对，之后查询 key 要能返回 value
    """
    def __init__(self, vocab_size: int = 1000, seq_len: int = 512,
                 num_samples: int = 10000, num_pairs: int = 8, query_at_end: bool = True):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        self.query_at_end = query_at_end
        
        # 特殊token
        self.sep_token = 1
        self.query_token = 2
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成 key-value 对
        keys = torch.randint(100, self.vocab_size // 2, (self.num_pairs,))
        values = torch.randint(self.vocab_size // 2, self.vocab_size, (self.num_pairs,))
        
        # 构建序列:  [k1, v1, SEP, k2, v2, SEP, .. ., QUERY, k_i] -> v_i
        seq_parts = []
        for k, v in zip(keys, values):
            seq_parts.extend([k. item(), v.item(), self.sep_token])
        
        # 随机选择一个 key 进行查询
        query_idx = torch.randint(0, self.num_pairs, (1,)).item()
        query_key = keys[query_idx]. item()
        query_value = values[query_idx].item()
        
        seq_parts.extend([self.query_token, query_key, query_value])
        
        # 补齐或截断
        if len(seq_parts) < self.seq_len:
            # 在前面加噪声
            noise_len = self.seq_len - len(seq_parts)
            noise = torch.randint(10, 100, (noise_len,)).tolist()
            seq_parts = noise + seq_parts
        else:
            seq_parts = seq_parts[-self.seq_len:]
        
        seq = torch.tensor(seq_parts, dtype=torch.long)
        return seq[:-1], seq[1:]


class PasskeyRetrievalDataset(Dataset):
    """Passkey 检索数据集 - 测试长距离信息检索
    在长文本中插入一个随机数字密钥，末尾要求模型回忆
    """
    def __init__(self, seq_len: int = 4096, num_samples: int = 1000, 
                 passkey_len: int = 5, noise_vocab:  int = 100):
        self.seq_len = seq_len
        self. num_samples = num_samples
        self.passkey_len = passkey_len
        self. noise_vocab = noise_vocab
        
        # 特殊token
        self.passkey_start = 1
        self.passkey_end = 2
        self.query_token = 3
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成 passkey
        passkey = torch.randint(100, 1000, (self.passkey_len,))
        
        # 生成噪声填充
        noise_len = self.seq_len - self.passkey_len - 10
        noise = torch.randint(10, self.noise_vocab, (noise_len,))
        
        # 随机选择插入位置 (在前半部分)
        insert_pos = torch.randint(10, noise_len // 2, (1,)).item()
        
        # 构建序列
        seq = torch.cat([
            noise[: insert_pos],
            torch.tensor([self.passkey_start]),
            passkey,
            torch.tensor([self.passkey_end]),
            noise[insert_pos: ],
            torch.tensor([self.query_token]),
            passkey  # 目标:  复现 passkey
        ])
        
        # 截断到指定长度
        seq = seq[:self.seq_len]
        
        return seq[:-1], seq[1:], insert_pos  # 返回插入位置用于分析


# =====================================================
# Part 3: 完整的真实数据实验
# =====================================================

# 复用之前定义的模型类
from gdn import (
    StandardAttention, FlashAttention, LinearAttention, 
    GatedDeltaNetAttention, SmallLM, ExperimentConfig
)


class RealDataExperiment:
    """真实数据集实验"""
    
    ATTENTION_CLASSES = {
        'Standard': StandardAttention,
        'Flash': FlashAttention,
        'Linear': LinearAttention,
        'GatedDeltaNet': GatedDeltaNetAttention,
    }
    
    def __init__(self, config: ExperimentConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        
    def train_and_evaluate(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        attention_name: str,
        num_epochs: int = 3,
        eval_every: int = 500
    ) -> Dict:
        """训练并评估模型"""
        
        attn_class = self.ATTENTION_CLASSES[attention_name]
        vocab_size = getattr(train_dataset, 'vocab_size', self.config.vocab_size)
        
        model = SmallLM(
            vocab_size=vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_seq_len=self.config.max_seq_len,
            attention_class=attn_class
        ).to(self.device).to(self.dtype)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, num_workers=2)
        
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        
        # 训练记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_ppl': [],
            'steps': []
        }
        
        global_step = 0
        best_val_ppl = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, y = batch[0], batch[1]
                    
                x, y = x.to(self. device), y.to(self.device)
                
                logits = model(x)
                loss = F.cross_entropy(logits. view(-1, vocab_size), y.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer. step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # 定期评估
                if global_step % eval_every == 0:
                    val_loss, val_ppl = self.evaluate(model, val_loader, vocab_size)
                    history['train_loss'].append(epoch_loss / (batch_idx + 1))
                    history['val_loss']. append(val_loss)
                    history['val_ppl']. append(val_ppl)
                    history['steps'].append(global_step)
                    
                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                    
                    print(f"\nStep {global_step}:  Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}")
                    model.train()
        
        # 最终评估
        final_val_loss, final_val_ppl = self.evaluate(model, val_loader, vocab_size)
        
        return {
            'attention':  attention_name,
            'final_val_loss': final_val_loss,
            'final_val_ppl': final_val_ppl,
            'best_val_ppl': best_val_ppl,
            'num_params': model.num_params,
            'history': history
        }
    
    def evaluate(self, model:  nn.Module, dataloader: DataLoader, vocab_size: int) -> tuple:
        """评估模型"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, y = batch[0], batch[1]
                    
                x, y = x.to(self.device), y.to(self.device)
                
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction='sum')
                
                # 不计算 padding token 的 loss
                mask = (y != 0).float()  # 假设 0 是 padding
                total_loss += loss.item()
                total_tokens += mask. sum().item()
        
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 100))  # 防止溢出
        
        return avg_loss, ppl
    
    def run_wikitext_experiment(self, dataset_name: str = 'wikitext-2-raw-v1', 
                                 num_epochs: int = 3) -> Dict:
        """WikiText 实验"""
        print("\n" + "="*80)
        print(f"WikiText 实验: {dataset_name}")
        print("="*80)
        
        seq_len = 512
        train_dataset = WikiTextDataset('train', seq_len, dataset_name)
        val_dataset = WikiTextDataset('validation', seq_len, dataset_name)
        
        # 更新 vocab_size
        self.config.vocab_size = train_dataset.vocab_size
        
        results = {}
        for attn_name in self.ATTENTION_CLASSES.keys():
            print(f"\n--- 训练 {attn_name} ---")
            try:
                result = self.train_and_evaluate(
                    train_dataset, val_dataset, attn_name, 
                    num_epochs=num_epochs, eval_every=500
                )
                results[attn_name] = result
                print(f"{attn_name}: Final PPL = {result['final_val_ppl']:.2f}")
            except Exception as e:
                print(f"{attn_name}:  Error - {str(e)}")
                results[attn_name] = {'error': str(e)}
        
        return results
    
    def run_synthetic_experiments(self) -> Dict:
        """合成任务实验"""
        print("\n" + "="*80)
        print("合成任务实验")
        print("="*80)
        
        all_results = {}
        
        # 1.  Induction Head 测试
        print("\n--- Induction Head 测试 ---")
        induction_train = InductionHeadDataset(vocab_size=1000, seq_len=256, num_samples=5000)
        induction_val = InductionHeadDataset(vocab_size=1000, seq_len=256, num_samples=500)
        
        for attn_name in self. ATTENTION_CLASSES.keys():
            print(f"测试 {attn_name}...")
            try:
                result = self.train_and_evaluate(
                    induction_train, induction_val, attn_name,
                    num_epochs=2, eval_every=200
                )
                all_results[f'induction_{attn_name}'] = result
            except Exception as e:
                print(f"  Error:  {e}")
        
        # 2. 联想记忆测试
        print("\n--- Associative Recall 测试 ---")
        assoc_train = AssociativeRecallDataset(vocab_size=1000, seq_len=256, num_samples=5000)
        assoc_val = AssociativeRecallDataset(vocab_size=1000, seq_len=256, num_samples=500)
        
        for attn_name in self.ATTENTION_CLASSES.keys():
            print(f"测试 {attn_name}...")
            try:
                result = self. train_and_evaluate(
                    assoc_train, assoc_val, attn_name,
                    num_epochs=2, eval_every=200
                )
                all_results[f'assoc_{attn_name}'] = result
            except Exception as e:
                print(f"  Error: {e}")
        
        return all_results


def main_real_data():
    """真实数据实验主函数"""
    
    device = 'cuda' if torch. cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    config = ExperimentConfig(
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        vocab_size=50257,  # GPT-2 vocab size
        max_seq_len=1024,
        batch_size=8,
        learning_rate=3e-4,
    )
    
    experiment = RealDataExperiment(config, device)
    
    # 运行 WikiText-2 实验
    wikitext_results = experiment.run_wikitext_experiment(
        dataset_name='wikitext-2-raw-v1',
        num_epochs=3
    )
    
    # 运行合成任务实验
    synthetic_results = experiment.run_synthetic_experiments()
    
    # 保存结果
    os.makedirs('./results', exist_ok=True)
    
    with open('./results/wikitext_results.json', 'w') as f:
        # 移除不可序列化的部分
        save_results = {}
        for k, v in wikitext_results.items():
            if isinstance(v, dict):
                save_results[k] = {kk: vv for kk, vv in v.items() if kk != 'history'}
        json. dump(save_results, f, indent=2)
    
    print("\n实验完成!")
    return wikitext_results, synthetic_results


if __name__ == "__main__":
    main_real_data()