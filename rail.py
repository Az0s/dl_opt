# =====================================================
# 线性注意力机制性能测试 (真实数据版)
# 使用 WikiText-2 数据集
# =====================================================

import torch
import torch.nn as nn
import torch. nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import gc
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# =====================================================
# Part 1: 注意力层实现
# =====================================================

class StandardAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self. num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch. matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.o_proj(out)


class FlashAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self. head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout_p = dropout
        
        try:
            from flash_attn import flash_attn_func
            self.flash_attn = flash_attn_func
            self.use_flash = True
        except: 
            self.use_flash = False
            
    def forward(self, x, attention_mask=None):
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self. head_dim)
        
        if self.use_flash:
            out = self.flash_attn(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, causal=True)
        else:
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2)
            
        out = out.contiguous().view(B, L, self.hidden_size)
        return self.o_proj(out)


class LinearAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self. head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            
    def forward(self, x, attention_mask=None):
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self. num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        
        q = F. elu(q) + 1
        k = F.elu(k) + 1
        
        kv = torch.einsum('blhd,blhe->blhde', k, v)
        kv_cumsum = torch.cumsum(kv, dim=1)
        k_cumsum = torch.cumsum(k, dim=1)
        
        out = torch.einsum('blhd,blhde->blhe', q, kv_cumsum)
        normalizer = torch.einsum('blhd,blhd->blh', q, k_cumsum).unsqueeze(-1).clamp(min=1e-6)
        out = out / normalizer
        
        out = out.contiguous().view(B, L, self.hidden_size)
        return self.o_proj(out)


class GatedDeltaNetAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        try:
            from fla.layers import GatedDeltaNet
            self.gated_deltanet = GatedDeltaNet(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mode='chunk'
            )
            self.use_fla = True
        except:
            print("FLA not available, using fallback")
            self.use_fla = False
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
            
    def forward(self, x, attention_mask=None):
        if self.use_fla:
            out, _, _ = self.gated_deltanet(x, attention_mask=attention_mask)
            return out
        else:
            B, L, _ = x.shape
            q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
            beta = torch.sigmoid(self.beta_proj(x)).unsqueeze(-1)
            
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            
            kv = torch.einsum('blhd,blhe->blhde', k, v)
            kv_cumsum = torch.cumsum(kv * beta. unsqueeze(-1), dim=1)
            out = torch.einsum('blhd,blhde->blhe', q, kv_cumsum)
            
            out = out.contiguous().view(B, L, self. hidden_size)
            return self.o_proj(out)


# =====================================================
# Part 2: Transformer 模型
# =====================================================

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attention_class=StandardAttention, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = attention_class(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self. norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SmallLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len=2048,
                 attention_class=StandardAttention, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn. Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, attention_class=attention_class, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn. LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.num_params = sum(p.numel() for p in self.parameters())
        
    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        
        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.drop(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return self.head(x)


# =====================================================
# Part 3: 数据集
# =====================================================

class WikiTextDataset(Dataset):
    """WikiText-2 数据集"""
    def __init__(self, split='train', seq_len=256):
        from datasets import load_dataset
        
        print("加载 WikiText-2 数据集, split=" + split + "...")
        
        # 加载数据
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, trust_remote_code=True)
        
        # 简单的字符级 tokenizer (避免依赖问题)
        # 收集所有文本
        all_text = '\n'.join([item['text'] for item in dataset if item['text']. strip()])
        
        # 构建词表 (字符级)
        chars = sorted(list(set(all_text)))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        
        # 编码
        self.tokens = [self.char_to_idx[c] for c in all_text]
        self.seq_len = seq_len
        self.num_samples = (len(self.tokens) - 1) // seq_len
        
        print("词表大小: " + str(self.vocab_size) + ", 样本数: " + str(self.num_samples))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        tokens = self.tokens[start:end]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class SimpleTextDataset(Dataset):
    """简单文本数据集 - 使用预定义文本，不需要下载"""
    def __init__(self, seq_len=256, num_samples=5000, vocab_size=256):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        
        # 生成有一定模式的数据（比纯随机更有意义）
        # 模拟自然语言的一些特性：局部相关性
        self.data = []
        for _ in range(num_samples):
            seq = []
            current = np.random.randint(0, vocab_size)
            for _ in range(seq_len + 1):
                seq.append(current)
                # 80% 概率选择相近的 token，20% 概率随机跳转
                if np.random.random() < 0.8:
                    current = (current + np.random.randint(-5, 6)) % vocab_size
                else:
                    current = np.random.randint(0, vocab_size)
            self.data.append(seq)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class SyntheticCopyDataset(Dataset):
    """复制任务数据集 - 测试记忆能力"""
    def __init__(self, vocab_size, seq_len, num_samples, copy_len=10):
        self.vocab_size = vocab_size
        self. seq_len = seq_len
        self.num_samples = num_samples
        self.copy_len = copy_len
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 格式:  [随机序列] [分隔符] [填充] [随机序列的复制]
        seq = torch.randint(4, self.vocab_size - 2, (self.copy_len,))
        sep = torch.tensor([2])
        padding_len = self.seq_len - 2 * self.copy_len - 1
        padding = torch. zeros(max(0, padding_len), dtype=torch.long)
        
        input_seq = torch.cat([seq, sep, padding, seq])[:self.seq_len]
        return input_seq[:-1], input_seq[1:]


class AssociativeRecallDataset(Dataset):
    """关联记忆数据集 - 测试 key-value 检索能力"""
    def __init__(self, vocab_size, seq_len, num_samples, num_pairs=5):
        self.vocab_size = vocab_size
        self. seq_len = seq_len
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 格式: [k1 v1] [k2 v2] ... [kn vn] [填充] [query_k] -> 预测 query_v
        keys = torch.randint(10, self.vocab_size // 2, (self.num_pairs,))
        values = torch.randint(self.vocab_size // 2, self.vocab_size - 10, (self.num_pairs,))
        
        # 构建 key-value 对序列
        pairs = torch.stack([keys, values], dim=1).flatten()
        
        # 选择一个 key 作为查询
        query_idx = torch.randint(0, self.num_pairs, (1,)).item()
        query_key = keys[query_idx]
        query_value = values[query_idx]
        
        # 填充
        padding_len = self.seq_len - len(pairs) - 2
        padding = torch.zeros(max(0, padding_len), dtype=torch.long)
        
        # 完整序列:  pairs + padding + query_key + query_value
        full_seq = torch.cat([pairs, padding, query_key. unsqueeze(0), query_value.unsqueeze(0)])[:self.seq_len]
        
        return full_seq[:-1], full_seq[1:]


# =====================================================
# Part 4: 测试类
# =====================================================

@dataclass
class Config:
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    vocab_size: int = 256
    max_seq_len: int = 512
    dropout: float = 0.1
    batch_size: int = 8
    learning_rate: float = 3e-4
    num_warmup:  int = 3
    num_iterations: int = 10


class Benchmark:
    
    ATTENTION_CLASSES = {
        'Standard': StandardAttention,
        'Flash': FlashAttention,
        'Linear': LinearAttention,
        'GatedDeltaNet': GatedDeltaNetAttention,
    }
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        
    def clear_cache(self):
        gc.collect()
        if self.device == 'cuda': 
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def get_memory(self):
        if self.device != 'cuda':
            return 0
        return torch.cuda.max_memory_allocated() / 1024**2
        
    def reset_memory(self):
        if self.device == 'cuda':
            torch. cuda.reset_peak_memory_stats()

    # ==================== 实验1: 训练效率 (合成数据) ====================
    def benchmark_training(self, seq_lengths=[256, 512, 1024]):
        print("\n" + "="*80)
        print("实验1: 训练效率测试")
        print("="*80)
        
        results = []
        
        for seq_len in seq_lengths: 
            print("\n序列长度:", seq_len)
            print("-" * 60)
            
            for attn_name, attn_class in self.ATTENTION_CLASSES.items():
                try:
                    self.clear_cache()
                    self.reset_memory()
                    
                    model = SmallLM(
                        vocab_size=self. config.vocab_size,
                        hidden_size=self.config.hidden_size,
                        num_layers=self.config. num_layers,
                        num_heads=self.config.num_heads,
                        max_seq_len=seq_len + 100,
                        attention_class=attn_class
                    ).to(self.device).to(self.dtype)
                    
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
                    
                    dataset = SimpleTextDataset(seq_len=seq_len, num_samples=1000, vocab_size=self.config. vocab_size)
                    dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
                    
                    # 预热
                    model.train()
                    for i, (x, y) in enumerate(dataloader):
                        if i >= self.config.num_warmup:
                            break
                        x, y = x.to(self.device), y.to(self.device)
                        logits = model(x)
                        loss = F.cross_entropy(logits. view(-1, self.config. vocab_size), y.view(-1))
                        loss.backward()
                        optimizer. step()
                        optimizer.zero_grad()
                    
                    if self.device == 'cuda': 
                        torch.cuda.synchronize()
                    
                    # 测试
                    self.reset_memory()
                    step_times = []
                    total_tokens = 0
                    
                    start_total = time.perf_counter()
                    for i, (x, y) in enumerate(dataloader):
                        if i >= self.config.num_iterations:
                            break
                            
                        x, y = x.to(self.device), y.to(self.device)
                        
                        if self.device == 'cuda': 
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, self. config.vocab_size), y.view(-1))
                        loss.backward()
                        optimizer. step()
                        optimizer.zero_grad()
                        
                        if self.device == 'cuda':
                            torch.cuda.synchronize()
                        step_time = time.perf_counter() - start
                        
                        step_times.append(step_time * 1000)
                        total_tokens += x.numel()
                    
                    total_time = time.perf_counter() - start_total
                    peak_mem = self.get_memory()
                    
                    avg_step = np.mean(step_times)
                    std_step = np.std(step_times)
                    throughput = total_tokens / total_time
                    
                    result = {
                        'attention':  attn_name,
                        'seq_len': seq_len,
                        'step_time_ms': avg_step,
                        'step_time_std': std_step,
                        'throughput': throughput,
                        'memory_peak_mb': peak_mem,
                        'num_params': model.num_params
                    }
                    results.append(result)
                    
                    print("  " + attn_name + " | Step: " + str(round(avg_step, 2)) + "ms | Throughput: " + str(int(throughput)) + " tok/s | Mem: " + str(round(peak_mem, 1)) + "MB")
                    
                except Exception as e: 
                    print("  " + attn_name + " | Error: " + str(e)[:80])
                    import traceback
                    traceback. print_exc()
                finally:
                    self.clear_cache()
                    
        return pd.DataFrame(results)

    # ==================== 实验2: 内存扩展性 ====================
    def benchmark_memory(self, seq_lengths=[256, 512, 1024, 2048]):
        print("\n" + "="*80)
        print("实验2: 内存扩展性测试")
        print("="*80)
        
        results = []
        
        for seq_len in seq_lengths:
            print("\n序列长度:", seq_len)
            print("-" * 60)
            
            for attn_name, attn_class in self.ATTENTION_CLASSES.items():
                try:
                    self.clear_cache()
                    self. reset_memory()
                    
                    attn = attn_class(
                        hidden_size=self.config.hidden_size,
                        num_heads=self.config.num_heads
                    ).to(self.device).to(self.dtype)
                    
                    x = torch.randn(self.config.batch_size, seq_len, self.config.hidden_size,
                                   device=self.device, dtype=self.dtype, requires_grad=True)
                    
                    self.reset_memory()
                    out = attn(x)
                    forward_mem = self.get_memory()
                    
                    self.reset_memory()
                    loss = out.sum()
                    loss.backward()
                    backward_mem = self.get_memory()
                    
                    result = {
                        'attention': attn_name,
                        'seq_len': seq_len,
                        'forward_memory_mb': forward_mem,
                        'backward_memory_mb': backward_mem,
                        'total_memory_mb': max(forward_mem, backward_mem)
                    }
                    results.append(result)
                    
                    print("  " + attn_name + " | Forward: " + str(round(forward_mem, 1)) + "MB | Backward: " + str(round(backward_mem, 1)) + "MB")
                    
                except torch.cuda.OutOfMemoryError:
                    print("  " + attn_name + " | OOM")
                    results.append({
                        'attention': attn_name,
                        'seq_len': seq_len,
                        'forward_memory_mb': float('inf'),
                        'backward_memory_mb': float('inf'),
                        'total_memory_mb': float('inf'),
                        'oom':  True
                    })
                except Exception as e:
                    print("  " + attn_name + " | Error: " + str(e)[:80])
                finally:
                    self.clear_cache()
                    
        return pd.DataFrame(results)

    # ==================== 实验3: Copy Task (记忆能力) ====================
    def benchmark_copy_task(self, copy_lengths=[5, 10, 20, 30], num_epochs=3):
        print("\n" + "="*80)
        print("实验3: 复制任务 (记忆能力测试)")
        print("="*80)
        
        results = []
        
        for copy_len in copy_lengths:
            print("\n复制长度:", copy_len)
            print("-" * 60)
            
            seq_len = copy_len * 3 + 20
            
            for attn_name, attn_class in self.ATTENTION_CLASSES.items():
                try:
                    self.clear_cache()
                    
                    model = SmallLM(
                        vocab_size=self.config.vocab_size,
                        hidden_size=self.config.hidden_size,
                        num_layers=self. config.num_layers,
                        num_heads=self.config.num_heads,
                        max_seq_len=seq_len + 100,
                        attention_class=attn_class
                    ).to(self.device).to(self.dtype)
                    
                    optimizer = torch. optim.AdamW(model.parameters(), lr=self.config.learning_rate)
                    
                    train_dataset = SyntheticCopyDataset(self.config.vocab_size, seq_len, 3000, copy_len)
                    test_dataset = SyntheticCopyDataset(self.config. vocab_size, seq_len, 500, copy_len)
                    train_loader = DataLoader(train_dataset, batch_size=self.config. batch_size, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
                    
                    # 训练
                    model.train()
                    train_losses = []
                    for epoch in range(num_epochs):
                        for x, y in train_loader: 
                            x, y = x. to(self.device), y.to(self.device)
                            logits = model(x)
                            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            train_losses.append(loss. item())
                    
                    # 评估
                    model.eval()
                    correct = 0
                    total = 0
                    with torch. no_grad():
                        for x, y in test_loader: 
                            x, y = x.to(self.device), y.to(self.device)
                            logits = model(x)
                            preds = logits.argmax(dim=-1)
                            # 只计算最后复制部分的准确率
                            copy_start = seq_len - copy_len - 1
                            correct += (preds[:, copy_start:] == y[: , copy_start:]).sum().item()
                            total += y[: , copy_start:].numel()
                    
                    accuracy = correct / total * 100 if total > 0 else 0
                    final_loss = np.mean(train_losses[-100:]) if train_losses else 0
                    
                    result = {
                        'attention': attn_name,
                        'copy_len': copy_len,
                        'accuracy': accuracy,
                        'final_loss': final_loss,
                    }
                    results.append(result)
                    
                    print("  " + attn_name + " | Accuracy: " + str(round(accuracy, 2)) + "% | Loss: " + str(round(final_loss, 4)))
                    
                except Exception as e:
                    print("  " + attn_name + " | Error: " + str(e)[:80])
                    import traceback
                    traceback.print_exc()
                finally:
                    self.clear_cache()
                    
        return pd.DataFrame(results)

    # ==================== 实验4: 关联记忆任务 ====================
    def benchmark_associative_recall(self, num_pairs_list=[3, 5, 8, 10], num_epochs=3):
        print("\n" + "="*80)
        print("实验4: 关联记忆任务 (Key-Value 检索)")
        print("="*80)
        
        results = []
        seq_len = 128
        
        for num_pairs in num_pairs_list: 
            print("\nKey-Value 对数:", num_pairs)
            print("-" * 60)
            
            for attn_name, attn_class in self.ATTENTION_CLASSES.items():
                try:
                    self.clear_cache()
                    
                    model = SmallLM(
                        vocab_size=self.config.vocab_size,
                        hidden_size=self. config.hidden_size,
                        num_layers=self.config.num_layers,
                        num_heads=self.config. num_heads,
                        max_seq_len=seq_len + 100,
                        attention_class=attn_class
                    ).to(self.device).to(self.dtype)
                    
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
                    
                    train_dataset = AssociativeRecallDataset(self.config.vocab_size, seq_len, 3000, num_pairs)
                    test_dataset = AssociativeRecallDataset(self. config.vocab_size, seq_len, 500, num_pairs)
                    train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
                    
                    # 训练
                    model. train()
                    train_losses = []
                    for epoch in range(num_epochs):
                        for x, y in train_loader:
                            x, y = x.to(self.device), y.to(self.device)
                            logits = model(x)
                            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                            loss. backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            train_losses.append(loss.item())
                    
                    # 评估 - 只看最后一个位置的预测
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for x, y in test_loader:
                            x, y = x.to(self.device), y.to(self.device)
                            logits = model(x)
                            preds = logits[: , -1, :].argmax(dim=-1)
                            targets = y[:, -1]
                            correct += (preds == targets).sum().item()
                            total += targets.numel()
                    
                    accuracy = correct / total * 100 if total > 0 else 0
                    final_loss = np.mean(train_losses[-100:]) if train_losses else 0
                    
                    result = {
                        'attention': attn_name,
                        'num_pairs': num_pairs,
                        'accuracy':  accuracy,
                        'final_loss': final_loss,
                    }
                    results.append(result)
                    
                    print("  " + attn_name + " | Accuracy:  " + str(round(accuracy, 2)) + "% | Loss: " + str(round(final_loss, 4)))
                    
                except Exception as e: 
                    print("  " + attn_name + " | Error: " + str(e)[:80])
                    import traceback
                    traceback.print_exc()
                finally:
                    self.clear_cache()
                    
        return pd.DataFrame(results)

    # ==================== 实验5: 语言建模困惑度 (真实数据) ====================
    def benchmark_perplexity(self, num_epochs=5, use_wikitext=True):
        print("\n" + "="*80)
        print("实验5: 语言建模困惑度测试")
        if use_wikitext:
            print("使用 WikiText-2 数据集")
        else:
            print("使用合成数据集")
        print("="*80)
        
        results = []
        seq_len = 256
        
        # 加载数据集
        if use_wikitext:
            try:
                train_dataset = WikiTextDataset(split='train', seq_len=seq_len)
                test_dataset = WikiTextDataset(split='test', seq_len=seq_len)
                vocab_size = train_dataset.vocab_size
            except Exception as e:
                print("WikiText 加载失败，使用合成数据:  " + str(e))
                use_wikitext = False
        
        if not use_wikitext:
            vocab_size = self.config.vocab_size
            train_dataset = SimpleTextDataset(seq_len=seq_len, num_samples=10000, vocab_size=vocab_size)
            test_dataset = SimpleTextDataset(seq_len=seq_len, num_samples=1000, vocab_size=vocab_size)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, num_workers=0)
        
        for attn_name, attn_class in self.ATTENTION_CLASSES.items():
            print("\n训练:", attn_name)
            print("-" * 40)
            
            try:
                self.clear_cache()
                
                model = SmallLM(
                    vocab_size=vocab_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    num_heads=self.config.num_heads,
                    max_seq_len=seq_len + 100,
                    attention_class=attn_class
                ).to(self.device).to(self.dtype)
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
                total_steps = len(train_loader) * num_epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
                
                # 训练
                model.train()
                train_losses = []
                step = 0
                for epoch in range(num_epochs):
                    pbar = tqdm(train_loader, desc="Epoch " + str(epoch+1))
                    for x, y in pbar:
                        x, y = x.to(self. device), y.to(self.device)
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        train_losses.append(loss. item())
                        step += 1
                        
                        if step % 100 == 0:
                            pbar.set_postfix({'loss': str(round(loss.item(), 4))})
                
                # 评估困惑度
                model.eval()
                total_loss = 0
                total_tokens = 0
                with torch. no_grad():
                    for x, y in test_loader: 
                        x, y = x.to(self.device), y.to(self.device)
                        logits = model(x)
                        loss = F. cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction='sum')
                        total_loss += loss.item()
                        total_tokens += y. numel()
                
                perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
                final_loss = np.mean(train_losses[-100:]) if train_losses else 0
                
                result = {
                    'attention': attn_name,
                    'perplexity': perplexity,
                    'final_train_loss': final_loss,
                    'vocab_size': vocab_size,
                    'dataset': 'WikiText-2' if use_wikitext else 'Synthetic'
                }
                results.append(result)
                
                print("  Perplexity: " + str(round(perplexity, 2)) + " | Final Loss: " + str(round(final_loss, 4)))
                
            except Exception as e: 
                print("  Error: " + str(e))
                import traceback
                traceback.print_exc()
            finally:
                self.clear_cache()
                
        return pd.DataFrame(results)

    # ==================== 实验6: 模型规模扩展性 ====================
    def benchmark_scaling(self):
        print("\n" + "="*80)
        print("实验6: 模型规模扩展性测试")
        print("="*80)
        
        model_configs = [
            {'hidden_size': 128, 'num_layers': 2, 'num_heads': 4, 'name': '~1M'},
            {'hidden_size': 256, 'num_layers':  4, 'num_heads': 8, 'name': '~5M'},
            {'hidden_size': 384, 'num_layers': 6, 'num_heads':  8, 'name': '~15M'},
            {'hidden_size': 512, 'num_layers': 8, 'num_heads': 8, 'name': '~30M'},
        ]
        
        results = []
        seq_len = 512
        
        for cfg in model_configs:
            print("\n模型规模:", cfg['name'])
            print("-" * 60)
            
            for attn_name, attn_class in self.ATTENTION_CLASSES.items():
                try:
                    self.clear_cache()
                    self.reset_memory()
                    
                    model = SmallLM(
                        vocab_size=self.config.vocab_size,
                        hidden_size=cfg['hidden_size'],
                        num_layers=cfg['num_layers'],
                        num_heads=cfg['num_heads'],
                        max_seq_len=seq_len + 100,
                        attention_class=attn_class
                    ).to(self.device).to(self.dtype)
                    
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
                    
                    x = torch.randint(0, self.config. vocab_size, (self.config.batch_size, seq_len), device=self.device)
                    y = torch.randint(0, self.config.vocab_size, (self.config.batch_size, seq_len), device=self.device)
                    
                    # 预热
                    for _ in range(3):
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, self. config.vocab_size), y.view(-1))
                        loss.backward()
                        optimizer. step()
                        optimizer.zero_grad()
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    # 测��
                    self.reset_memory()
                    step_times = []
                    for _ in range(10):
                        if self.device == 'cuda': 
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        
                        logits = model(x)
                        loss = F. cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        if self.device == 'cuda': 
                            torch.cuda.synchronize()
                        step_times.append((time.perf_counter() - start) * 1000)
                    
                    peak_mem = self.get_memory()
                    avg_step = np.mean(step_times)
                    throughput = (self.config.batch_size * seq_len) / (avg_step / 1000)
                    
                    result = {
                        'attention': attn_name,
                        'model_size': cfg['name'],
                        'num_params': model.num_params,
                        'hidden_size': cfg['hidden_size'],
                        'num_layers': cfg['num_layers'],
                        'step_time_ms': avg_step,
                        'memory_peak_mb': peak_mem,
                        'throughput':  throughput
                    }
                    results. append(result)
                    
                    print("  " + attn_name + " | Params: " + str(round(model.num_params/1e6, 2)) + "M | Step:  " + str(round(avg_step, 2)) + "ms | Mem: " + str(round(peak_mem, 1)) + "MB")
                    
                except torch.cuda.OutOfMemoryError:
                    print("  " + attn_name + " | OOM")
                except Exception as e:
                    print("  " + attn_name + " | Error: " + str(e)[:80])
                finally:
                    self.clear_cache()
                    
        return pd.DataFrame(results)


# =====================================================
# Part 5: 绘图
# =====================================================

def plot_results(results, save_dir='./figures'):
    os.makedirs(save_dir, exist_ok=True)
    
    colors = {'Standard': '#1f77b4', 'Flash': '#ff7f0e', 'Linear': '#2ca02c', 'GatedDeltaNet': '#d62728'}
    markers = {'Standard': 'o', 'Flash': 's', 'Linear': '^', 'GatedDeltaNet': 'D'}
    
    # 图1: 训练效率
    if 'training' in results and len(results['training']) > 0:
        df = results['training']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for attn in df['attention'].unique():
            data = df[df['attention'] == attn]
            c = colors. get(attn, 'gray')
            m = markers.get(attn, 'o')
            axes[0].plot(data['seq_len'], data['step_time_ms'], marker=m, color=c, label=attn, linewidth=2, markersize=8)
            axes[1].plot(data['seq_len'], data['throughput'], marker=m, color=c, label=attn, linewidth=2, markersize=8)
            valid = data[data['memory_peak_mb']. notna() & (data['memory_peak_mb'] < 1e10)]
            if len(valid) > 0:
                axes[2].plot(valid['seq_len'], valid['memory_peak_mb'], marker=m, color=c, label=attn, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('Step Time (ms)')
        axes[0].set_title('Training Step Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Sequence Length')
        axes[1].set_ylabel('Throughput (tokens/s)')
        axes[1].set_title('Training Throughput')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Sequence Length')
        axes[2].set_ylabel('Peak Memory (MB)')
        axes[2].set_title('GPU Memory Usage')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir + '/training_efficiency. png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved:  " + save_dir + "/training_efficiency.png")
    
    # 图2: 内存扩展性
    if 'memory' in results and len(results['memory']) > 0:
        df = results['memory']
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for attn in df['attention'].unique():
            data = df[df['attention'] == attn]
            data = data[data['total_memory_mb'] < 1e10]
            if len(data) > 0:
                c = colors.get(attn, 'gray')
                m = markers.get(attn, 'o')
                ax.plot(data['seq_len'], data['total_memory_mb'], marker=m, color=c, label=attn, linewidth=2, markersize=8)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Scaling with Sequence Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir + '/memory_scaling.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: " + save_dir + "/memory_scaling. png")
    
    # 图3: Copy Task
    if 'copy_task' in results and len(results['copy_task']) > 0:
        df = results['copy_task']
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for attn in df['attention'].unique():
            data = df[df['attention'] == attn]
            c = colors.get(attn, 'gray')
            m = markers.get(attn, 'o')
            ax.plot(data['copy_len'], data['accuracy'], marker=m, color=c, label=attn, linewidth=2, markersize=8)
        
        ax.set_xlabel('Copy Length')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Copy Task Accuracy (Memory Retrieval)')
        ax.legend()
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir + '/copy_task.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: " + save_dir + "/copy_task.png")
    
    # 图4: 关联记忆
    if 'associative' in results and len(results['associative']) > 0:
        df = results['associative']
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for attn in df['attention'].unique():
            data = df[df['attention'] == attn]
            c = colors.get(attn, 'gray')
            m = markers.get(attn, 'o')
            ax.plot(data['num_pairs'], data['accuracy'], marker=m, color=c, label=attn, linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Key-Value Pairs')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Associative Recall Accuracy')
        ax.legend()
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir + '/associative_recall.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: " + save_dir + "/associative_recall. png")
    
    # 图5: 困惑度
    if 'perplexity' in results and len(results['perplexity']) > 0:
        df = results['perplexity']
        fig, ax = plt. subplots(figsize=(8, 5))
        
        x = np.arange(len(df))
        bar_colors = [colors.get(a, 'gray') for a in df['attention']]
        bars = ax.bar(x, df['perplexity'], color=bar_colors)
        ax.set_xticks(x)
        ax.set_xticklabels(df['attention'], rotation=15)
        ax.set_ylabel('Perplexity')
        ax.set_title('Language Modeling Perplexity (Lower is Better)')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, df['perplexity']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   str(round(val, 1)), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_dir + '/perplexity.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: " + save_dir + "/perplexity.png")
    
    # 图6: 模型扩展性
    if 'scaling' in results and len(results['scaling']) > 0:
        df = results['scaling']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for attn in df['attention']. unique():
            data = df[df['attention'] == attn]. sort_values('num_params')
            c = colors.get(attn, 'gray')
            m = markers.get(attn, 'o')
            axes[0].plot(data['num_params']/1e6, data['step_time_ms'], marker=m, color=c, label=attn, linewidth=2, markersize=8)
            axes[1].plot(data['num_params']/1e6, data['throughput'], marker=m, color=c, label=attn, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Parameters (M)')
        axes[0].set_ylabel('Step Time (ms)')
        axes[0].set_title('Training Time vs Model Size')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Parameters (M)')
        axes[1].set_ylabel('Throughput (tokens/s)')
        axes[1].set_title('Throughput vs Model Size')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir + '/model_scaling.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: " + save_dir + "/model_scaling. png")


# =====================================================
# Part 6: 主函数
# =====================================================

def main():
    print("="*80)
    print("线性注意力机制综合实验")
    print("="*80)
    
    # 环境检查
    if torch.cuda.is_available():
        device = 'cuda'
        print("GPU:", torch.cuda.get_device_name(0))
        print("显存:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
    else:
        device = 'cpu'
        print("警告: 未检测到GPU，使用CPU")
    
    print("PyTorch:", torch.__version__)
    
    # 检查依赖
    try:
        from flash_attn import flash_attn_func
        print("Flash Attention:  OK")
    except:
        print("Flash Attention:  Not available")
    
    try:
        from fla. layers import GatedDeltaNet
        print("FLA (Gated DeltaNet): OK")
    except:
        print("FLA (Gated DeltaNet): Not available (using fallback)")
    
    try:
        from datasets import load_dataset
        print("Datasets library: OK")
    except:
        print("Datasets library: Not available (pip install datasets)")
    
    # 配置
    config = Config(
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        vocab_size=256,
        batch_size=8,
        learning_rate=3e-4,
        num_iterations=10
    )
    
    benchmark = Benchmark(config, device=device)
    
    all_results = {}
    
    print("\n" + "#"*80)
    print("开始运行实验...")
    print("#"*80)
    
    # 实验1: 训练效率
    all_results['training'] = benchmark. benchmark_training(seq_lengths=[256, 512, 1024])
    
    # 实验2: 内存扩展性
    all_results['memory'] = benchmark.benchmark_memory(seq_lengths=[256, 512, 1024, 2048])
    
    # 实验3: Copy Task (记忆能力)
    all_results['copy_task'] = benchmark.benchmark_copy_task(copy_lengths=[5, 10, 20, 30], num_epochs=3)
    
    # 实验4: 关联记忆
    all_results['associative'] = benchmark.benchmark_associative_recall(num_pairs_list=[3, 5, 8, 10], num_epochs=3)
    
    # 实验5: 语言建模困惑度 (尝试使用 WikiText-2)
    all_results['perplexity'] = benchmark.benchmark_perplexity(num_epochs=3, use_wikitext=True)
    
    # 实验6: 模型规模扩展性
    all_results['scaling'] = benchmark. benchmark_scaling()
    
    # 保存结果
    os.makedirs('./results', exist_ok=True)
    for name, df in all_results.items():
        df.to_csv('./results/' + name + '_results.csv', index=False)
        print("已保存: ./results/" + name + "_results.csv")
    
    # 绘图
    plot_results(all_results, save_dir='./figures')
    
    # 汇总
    print("\n" + "="*80)
    print("实验完成!  结果汇总")
    print("="*80)
    
    for name, df in all_results. items():
        print("\n【" + name + "】")
        print(df.to_string(index=False))
    
    return all_results


if __name__ == "__main__":
    # 先安装必要依赖
    print("检查依赖...")
    try:
        import datasets
    except ImportError:
        print("正在安装 datasets 库...")
        os.system("pip install datasets -q")
    
    results = main()