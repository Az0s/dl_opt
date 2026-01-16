# =====================================================
# 线性注意力机制性能测试 (修复版)
# =====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import gc
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# =====================================================
# Part 1:注意力层实现
# =====================================================

class StandardAttention(nn.Module):
    """标准 Scaled Dot-Product Attention"""
    def __init__(self, hidden_size:int, num_heads:int, dropout:float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self. num_heads = num_heads
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
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.o_proj(out)


class FlashAttention(nn.Module):
    """Flash Attention 2 实现"""
    def __init__(self, hidden_size:int, num_heads:int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout_p = dropout
        
        try:
            from flash_attn import flash_attn_func
            self.flash_attn = flash_attn_func
            self.use_flash = True
        except ImportError:
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
    """线性注意力"""
    def __init__(self, hidden_size:int, num_heads:int, dropout:float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def feature_map(self, x):
        return F. elu(x) + 1
            
    def forward(self, x, attention_mask=None):
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self. num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        kv = torch.einsum('blhd,blhe->blhde', k, v)
        kv_cumsum = torch.cumsum(kv, dim=1)
        k_cumsum = torch.cumsum(k, dim=1)
        
        out = torch.einsum('blhd,blhde->blhe', q, kv_cumsum)
        normalizer = torch.einsum('blhd,blhd->blh', q, k_cumsum).unsqueeze(-1).clamp(min=1e-6)
        out = out / normalizer
        
        out = out.contiguous().view(B, L, self.hidden_size)
        return self.o_proj(out)


class GatedDeltaNetAttention(nn.Module):
    """Gated DeltaNet"""
    def __init__(self, hidden_size:int, num_heads:int, dropout:float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        try:
            from fla. layers import GatedDeltaNet
            self.gated_deltanet = GatedDeltaNet(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mode='chunk'
            )
            self.use_fla = True
        except ImportError:
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
            kv_cumsum = torch. cumsum(kv * beta. unsqueeze(-1), dim=1)
            out = torch.einsum('blhd,blhde->blhe', q, kv_cumsum)
            
            out = out.contiguous().view(B, L, self.hidden_size)
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
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0):
        self.eval()
        with torch. no_grad():
            for _ in range(max_new_tokens):
                logits = self(input_ids)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# =====================================================
# Part 3:数据集
# =====================================================

class RandomTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


class SyntheticCopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples, copy_len=10):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self. copy_len = copy_len
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = torch.randint(4, self.vocab_size - 2, (self.copy_len,))
        sep = torch.tensor([2])
        padding_len = self.seq_len - 2 * self.copy_len - 1
        padding = torch.zeros(max(0, padding_len), dtype=torch.long)
        
        input_seq = torch.cat([seq, sep, padding, seq])[:self.seq_len]
        return input_seq[:-1], input_seq[1:]


# =====================================================
# Part 4:测试类
# =====================================================

@dataclass
class ExperimentConfig:
    hidden_size:int = 256
    num_layers:int = 4
    num_heads:int = 8
    vocab_size:int = 10000
    max_seq_len: int = 2048
    dropout:float = 0.1
    batch_size:int = 4
    learning_rate:float = 1e-4
    num_warmup: int = 3
    num_iterations:int = 10


class Benchmark:
    
    ATTENTION_CLASSES = {
        'Standard':StandardAttention,
        'Flash':FlashAttention,
        'Linear':LinearAttention,
        'GatedDeltaNet':GatedDeltaNetAttention,
    }
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16 if device == 'cuda' else torch. float32
        
    def clear_cache(self):
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def get_memory_stats(self):
        if self.device != 'cuda':
            return {'allocated':0, 'reserved':0, 'peak':0}
        return {
            'allocated':torch. cuda.memory_allocated() / 1024**2,
            'reserved':torch.cuda.memory_reserved() / 1024**2,
            'peak':torch.cuda.max_memory_allocated() / 1024**2
        }
        
    def reset_peak_memory(self):
        if self.device == 'cuda':
            torch. cuda.reset_peak_memory_stats()
    
    # ==================== 实验1: 训练效率 ====================
    def benchmark_training(self, seq_lengths=[512, 1024, 2048]):
        print("\n" + "="*80)
        print("实验1:训练效率测试")
        print("="*80)
        
        results = []
        
        for seq_len in seq_lengths:
            print("\n序列长度: {}".format(seq_len))
            print("-" * 60)
            
            for attn_name, attn_class in self.ATTENTION_CLASSES.items():
                try:
                    self.clear_cache()
                    self.reset_peak_memory()
                    
                    model = SmallLM(
                        vocab_size=self. config.vocab_size,
                        hidden_size=self.config.hidden_size,
                        num_layers=self.config. num_layers,
                        num_heads=self.config.num_heads,
                        max_seq_len=seq_len + 100,
                        attention_class=attn_class
                    ).to(self.device).to(self.dtype)
                    
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
                    
                    dataset = RandomTextDataset(self.config.vocab_size, seq_len, 1000)
                    dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
                    
                    # 预热
                    model.train()
                    for i, (x, y) in enumerate(dataloader):
                        if i >= self.config.num_warmup:
                            break
                        x, y = x.to(self. device), y.to(self.device)
                        logits = model(x)
                        loss = F.cross_entropy(logits. view(-1, self.config. vocab_size), y.view(-1))
                        loss. backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    # 测试
                    self.reset_peak_memory()
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
                        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                        loss.backward()
                        optimizer. step()
                        optimizer.zero_grad()
                        
                        if self.device == 'cuda':
                            torch.cuda.synchronize()
                        step_time = time.perf_counter() - start
                        
                        step_times.append(step_time * 1000)
                        total_tokens += x.numel()
                    
                    total_time = time.perf_counter() - start_total
                    mem_stats = self.get_memory_stats()
                    
                    avg_step = np.mean(step_times)
                    std_step = np.std(step_times)
                    throughput = total_tokens / total_time
                    peak_mem = mem_stats['peak']
                    
                    result = {
                        'attention': attn_name,
                        'seq_len':seq_len,
                        'step_time_ms':avg_step,
                        'step_time_std':std_step,
                        'throughput':throughput,
                        'memory_peak_mb':peak_mem,
                        'num_params':model.num_params
                    }
                    results. append(result)
                    
                    # 修复: 使用format而不是f-string
                    print("  {:15s} | Step:{:7.2f}±{:5.2f}ms | Throughput:{:8.0f} tok/s | Mem:{:7.1f}MB". format(
                        attn_name, avg_step, std_step, throughput, peak_mem))
                    
                except Exception as e:
                    print("  {:15s} | Error: {}".format(attn_name, str(e)[:50]))
                    results.append({
                        'attention':attn_name, 
                        'seq_len':seq_len,
                        'step_time_ms':float('nan'), 
                        'throughput':0,
                        'memory_peak_mb':float('nan'), 
                        'error':str(e)
                    })
                finally:
                    self.clear_cache()
                    
        return pd.DataFrame(results)
    
    # ==================== 实验2:推理效率 ====================
    def benchmark_inference(self, seq_lengths=[512, 1024], generate_lengths=[50, 100]):
        print("\n" + "="*80)
        print("实验2:推理效率测试")
        print("="*80)
        
        results = []
        
        for seq_len in seq_lengths:
            print("\n输入序列长度:{}". format(seq_len))
            print("-" * 60)
            
            for attn_name, attn_class in self.ATTENTION_CLASSES.items():
                try:
                    self.clear_cache()
                    
                    model = SmallLM(
                        vocab_size=self.config.vocab_size,
                        hidden_size=self.config.hidden_size,
                        num_layers=self. config.num_layers,
                        num_heads=self.config.num_heads,
                        max_seq_len=seq_len + max(generate_lengths) + 100,
                        attention_class=attn_class
                    ).to(self.device).to(self.dtype)
                    model.eval()
                    
                    x = torch.randint(0, self.config.vocab_size, (1, seq_len), device=self.device)
                    
                    # 预热
                    with torch.no_grad():
                        for _ in range(3):
                            _ = model(x)
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    # 前向延迟
                    forward_times = []
                    with torch.no_grad():
                        for _ in range(self.config.num_iterations):
                            if self.device == 'cuda':
                                torch.cuda.synchronize()
                            start = time.perf_counter()
                            _ = model(x)
                            if self.device == 'cuda':
                                torch.cuda.synchronize()
                            forward_times.append((time.perf_counter() - start) * 1000)
                    
                    # 生成速度
                    for gen_len in generate_lengths:
                        gen_times = []
                        with torch.no_grad():
                            for _ in range(3):
                                x_gen = torch.randint(0, self.config.vocab_size, (1, 10), device=self.device)
                                if self.device == 'cuda':
                                    torch.cuda.synchronize()
                                start = time.perf_counter()
                                _ = model. generate(x_gen, max_new_tokens=gen_len)
                                if self.device == 'cuda':
                                    torch.cuda.synchronize()
                                gen_times.append((time.perf_counter() - start) * 1000)
                        
                        result = {
                            'attention': attn_name,
                            'seq_len':seq_len,
                            'generate_len':gen_len,
                            'forward_time_ms':np.mean(forward_times),
                            'generate_time_ms':np.mean(gen_times),
                            'tokens_per_sec':gen_len / (np.mean(gen_times) / 1000)
                        }
                        results.append(result)
                    
                    print("  {:15s} | Forward:{:6.2f}ms | Gen({}):{:7.2f}ms". format(
                        attn_name, np.mean(forward_times), generate_lengths[-1], results[-1]['generate_time_ms']))
                    
                except Exception as e:
                    print("  {:15s} | Error: {}".format(attn_name, str(e)[:50]))
                finally:
                    self.clear_cache()
                    
        return pd.DataFrame(results)
    
    # ==================== 实验3:内存扩展性 ====================
    def benchmark_memory(self, seq_lengths=[256, 512, 1024, 2048, 4096]):
        print("\n" + "="*80)
        print("实验3:内存扩展性测试")
        print("="*80)
        
        results = []
        
        for seq_len in seq_lengths:
            print("\n序列长度: {}".format(seq_len))
            print("-" * 60)
            
            for attn_name, attn_class in self.ATTENTION_CLASSES. items():
                try:
                    self.clear_cache()
                    self.reset_peak_memory()
                    
                    attn = attn_class(
                        hidden_size=self.config.hidden_size,
                        num_heads=self.config.num_heads
                    ).to(self.device).to(self.dtype)
                    
                    x = torch.randn(self.config.batch_size, seq_len, self.config.hidden_size,
                                   device=self. device, dtype=self.dtype, requires_grad=True)
                    
                    # 前向
                    self.reset_peak_memory()
                    out = attn(x)
                    forward_mem = self.get_memory_stats()['peak']
                    
                    # 后向
                    self.reset_peak_memory()
                    loss = out.sum()
                    loss.backward()
                    backward_mem = self.get_memory_stats()['peak']
                    
                    result = {
                        'attention': attn_name,
                        'seq_len':seq_len,
                        'forward_memory_mb':forward_mem,
                        'backward_memory_mb':backward_mem,
                        'total_memory_mb':max(forward_mem, backward_mem)
                    }
                    results.append(result)
                    
                    print("  {:15s} | Forward:{:7.1f}MB | Backward:{:7.1f}MB". format(
                        attn_name, forward_mem, backward_mem))
                    
                except torch.cuda.OutOfMemoryError:
                    print("  {:15s} | OOM".format(attn_name))
                    results.append({
                        'attention':attn_name, 
                        'seq_len':seq_len,
                        'forward_memory_mb':float('inf'), 
                        'backward_memory_mb':float('inf'),
                        'total_memory_mb':float('inf'), 
                        'oom':True
                    })
                except Exception as e:
                    print("  {:15s} | Error:{}".format(attn_name, str(e)[:40]))
                finally:
                    self.clear_cache()
                    
        return pd.DataFrame(results)
    
    # ==================== 实验4:Copy Task ====================
    def benchmark_copy_task(self, copy_lengths=[5, 10, 20], num_train_steps=300):
        print("\n" + "="*80)
        print("实验4:复制任务 (记忆能力测试)")
        print("="*80)
        
        results = []
        
        for copy_len in copy_lengths:
            print("\n复制长度:{}".format(copy_len))
            print("-" * 60)
            
            seq_len = copy_len * 3 + 10
            
            for attn_name, attn_class in self.ATTENTION_CLASSES.items():
                try:
                    self.clear_cache()
                    
                    model = SmallLM(
                        vocab_size=self.config.vocab_size,
                        hidden_size=self.config.hidden_size,
                        num_layers=self.config.num_layers,
                        num_heads=self.config.num_heads,
                        max_seq_len=seq_len + 100,
                        attention_class=attn_class
                    ).to(self.device).to(self.dtype)
                    
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
                    
                    train_dataset = SyntheticCopyDataset(self.config.vocab_size, seq_len, 5000, copy_len)
                    test_dataset = SyntheticCopyDataset(self.config. vocab_size, seq_len, 500, copy_len)
                    train_loader = DataLoader(train_dataset, batch_size=self.config. batch_size, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
                    
                    # 训练
                    model.train()
                    train_losses = []
                    for step, (x, y) in enumerate(train_loader):
                        if step >= num_train_steps:
                            break
                        x, y = x.to(self.device), y.to(self.device)
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
                            copy_start = seq_len - copy_len - 1
                            correct += (preds[:, copy_start:] == y[:, copy_start:]).sum().item()
                            total += y[:, copy_start:].numel()
                    
                    accuracy = correct / total * 100 if total > 0 else 0
                    final_loss = np.mean(train_losses[-50:]) if train_losses else float('nan')
                    
                    result = {
                        'attention':attn_name,
                        'copy_len':copy_len,
                        'accuracy':accuracy,
                        'final_loss':final_loss,
                    }
                    results.append(result)
                    
                    print("  {:15s} | Accuracy:{:6.2f}% | Loss:{:.4f}".format(
                        attn_name, accuracy, final_loss))
                    
                except Exception as e:
                    print("  {:15s} | Error:{}".format(attn_name, str(e)[:50]))
                finally:
                    self.clear_cache()
                    
        return pd.DataFrame(results)
    
    # ==================== 实验5:困惑度 ====================
    def benchmark_perplexity(self, num_train_steps=500):
        print("\n" + "="*80)
        print("实验5:语言建模困惑度测试")
        print("="*80)
        
        results = []
        seq_len = 512
        
        train_dataset = RandomTextDataset(self.config. vocab_size, seq_len, 10000)
        test_dataset = RandomTextDataset(self.config.vocab_size, seq_len, 1000)
        train_loader = DataLoader(train_dataset, batch_size=self. config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        for attn_name, attn_class in self.ATTENTION_CLASSES.items():
            print("\n训练: {}".format(attn_name))
            print("-" * 40)
            
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
                
                # 训练
                model.train()
                train_losses = []
                pbar = tqdm(enumerate(train_loader), total=num_train_steps, desc=attn_name)
                for step, (x, y) in pbar:
                    if step >= num_train_steps:
                        break
                    x, y = x.to(self. device), y.to(self.device)
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, self. config.vocab_size), y.view(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer. zero_grad()
                    train_losses.append(loss.item())
                    
                    if step % 100 == 0:
                        pbar.set_postfix({'loss':'{:.4f}'.format(loss.item())})
                
                # 评估
                model.eval()
                total_loss = 0
                total_tokens = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, self. config.vocab_size), y.view(-1), reduction='sum')
                        total_loss += loss.item()
                        total_tokens += y. numel()
                
                perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
                final_loss = np.mean(train_losses[-100:]) if train_losses else float('nan')
                
                result = {
                    'attention': attn_name,
                    'perplexity':perplexity,
                    'final_train_loss':final_loss,
                }
                results.append(result)
                
                print("  Perplexity:{:.2f} | Final Loss:{:.4f}".format(perplexity, final_loss))
                
            except Exception as e:
                print("  Error:{}".format(str(e)))
            finally:
                self.clear_cache()
                
        return pd.DataFrame(results)
    
    # ==================== 实验6:模型扩展性 ====================
    def benchmark_scaling(self):
        print("\n" + "="*80)
        print("实验6:模型规模扩展性测试")
        print("="*80)
        
        model_configs = [
            {'hidden_size':128, 'num_layers':2, 'num_heads': 4, 'name':'~2M'},
            {'hidden_size':256, 'num_layers': 4, 'num_heads':8, 'name':'~10M'},
            {'hidden_size':512, 'num_layers':6, 'num_heads':8, 'name':'~40M'},
        ]
        
        results = []
        seq_len = 1024
        
        for cfg in model_configs:
            print("\n模型规模:{}".format(cfg['name']))
            print("-" * 60)
            
            for attn_name, attn_class in self.ATTENTION_CLASSES. items():
                try:
                    self.clear_cache()
                    self.reset_peak_memory()
                    
                    model = SmallLM(
                        vocab_size=self.config.vocab_size,
                        hidden_size=cfg['hidden_size'],
                        num_layers=cfg['num_layers'],
                        num_heads=cfg['num_heads'],
                        max_seq_len=seq_len + 100,
                        attention_class=attn_class
                    ).to(self.device).to(self.dtype)
                    
                    optimizer = torch.optim.AdamW(model. parameters(), lr=self.config.learning_rate)
                    
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
                    
                    # 测试
                    self.reset_peak_memory()
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
                    
                    mem_stats = self.get_memory_stats()
                    avg_step = np.mean(step_times)
                    peak_mem = mem_stats['peak']
                    throughput = (self.config.batch_size * seq_len) / (avg_step / 1000)
                    
                    result = {
                        'attention':attn_name,
                        'model_size':cfg['name'],
                        'num_params':model.num_params,
                        'hidden_size': cfg['hidden_size'],
                        'num_layers':cfg['num_layers'],
                        'step_time_ms':avg_step,
                        'memory_peak_mb':peak_mem,
                        'throughput': throughput
                    }
                    results. append(result)
                    
                    print("  {:15s} | Params:{:5.1f}M | Step:{:7.2f}ms | Mem:{:7.1f}MB". format(
                        attn_name, model.num_params/1e6, avg_step, peak_mem))
                    
                except torch.cuda.OutOfMemoryError:
                    print("  {:15s} | OOM".format(attn_name))
                except Exception as e:
                    print("  {:15s} | Error:{}".format(attn_name, str(e)[:40]))
                finally:
                    self.clear_cache()
                    
        return pd.DataFrame(results)


# =====================================================
# Part 5:绘图
# =====================================================

def plot_results(results, save_dir='./figures'):
    os.makedirs(save_dir, exist_ok=True)
    
    colors = {'Standard':'#1f77b4', 'Flash':'#ff7f0e', 'Linear':'#2ca02c', 'GatedDeltaNet':'#d62728'}
    markers = {'Standard':'o', 'Flash':'s', 'Linear':'^', 'GatedDeltaNet':'D'}
    
    # 图1:训练效率
    if 'training' in results and len(results['training']) > 0:
        df = results['training']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for attn in df['attention'].unique():
            data = df[df['attention'] == attn]
            if len(data) > 0 and 'step_time_ms' in data.columns:
                axes[0].plot(data['seq_len'], data['step_time_ms'], 
                            marker=markers. get(attn, 'o'), color=colors.get(attn, 'gray'), 
                            label=attn, linewidth=2, markersize=8)
                axes[1].plot(data['seq_len'], data['throughput'], 
                            marker=markers.get(attn, 'o'), color=colors.get(attn, 'gray'),
                            label=attn, linewidth=2, markersize=8)
                if 'memory_peak_mb' in data. columns:
                    valid_data = data[data['memory_peak_mb']. notna() & (data['memory_peak_mb'] < float('inf'))]
                    if len(valid_data) > 0:
                        axes[2].plot(valid_data['seq_len'], valid_data['memory_peak_mb'], 
                                    marker=markers.get(attn, 'o'), color=colors.get(attn, 'gray'),
                                    label=attn, linewidth=2, markersize=8)
        
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
        plt.savefig('{}/training_efficiency.png'.format(save_dir), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: {}/training_efficiency.png".format(save_dir))
    
    # 图2:内存
    if 'memory' in results and len(results['memory']) > 0:
        df = results['memory']
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for attn in df['attention'].unique():
            data = df[df['attention'] == attn]
            data = data[data['total_memory_mb'] < float('inf')]
            if len(data) > 0:
                ax.plot(data['seq_len'], data['total_memory_mb'], 
                       marker=markers.get(attn, 'o'), color=colors.get(attn, 'gray'),
                       label=attn, linewidth=2, markersize=8)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('{}/memory_scaling.png'.format(save_dir), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved:{}/memory_scaling.png".format(save_dir))
    
    # 图3:Copy Task
    if 'copy_task' in results and len(results['copy_task']) > 0:
        df = results['copy_task']
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for attn in df['attention'].unique():
            data = df[df['attention'] == attn]
            if len(data) > 0:
                ax.plot(data['copy_len'], data['accuracy'], 
                       marker=markers.get(attn, 'o'), color=colors.get(attn, 'gray'),
                       label=attn, linewidth=2, markersize=8)
        
        ax. set_xlabel('Copy Length')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Copy Task Accuracy')
        ax.legend()
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('{}/copy_task.png'.format(save_dir), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved:{}/copy_task.png".format(save_dir))
    
    # 图4:困惑度
    if 'perplexity' in results and len(results['perplexity']) > 0:
        df = results['perplexity']
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(df))
        bar_colors = [colors.get(a, 'gray') for a in df['attention']]
        bars = ax.bar(x, df['perplexity'], color=bar_colors)
        ax.set_xticks(x)
        ax.set_xticklabels(df['attention'], rotation=15)
        ax.set_ylabel('Perplexity')
        ax.set_title('Language Modeling Perplexity (Lower is Better)')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, df['perplexity']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   '{:.1f}'.format(val), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('{}/perplexity.png'.format(save_dir), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved:{}/perplexity.png".format(save_dir))


# =====================================================
# Part 6:主函数
# =====================================================

def main():
    print("="*80)
    print("实验环境检查")
    print("="*80)
    
    if torch.cuda.is_available():
        device = 'cuda'
        print("GPU: {}".format(torch.cuda. get_device_name(0)))
        print("CUDA: {}".format(torch.version. cuda))
        print("显存:{:.1f} GB".format(torch.cuda. get_device_properties(0).total_memory / 1024**3))
    else:
        device = 'cpu'
        print("警告:未检测到GPU")
    
    print("PyTorch: {}".format(torch.__version__))
    
    # 检查依赖
    try:
        from flash_attn import flash_attn_func
        print("Flash Attention: OK")
    except ImportError:
        print("Flash Attention: Not available (using PyTorch SDPA)")
    
    try:
        from fla. layers import GatedDeltaNet
        print("FLA (Gated DeltaNet):OK")
    except ImportError:
        print("FLA (Gated DeltaNet):Not available (using fallback)")
    
    # 配置
    config = ExperimentConfig(
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        vocab_size=10000,
        batch_size=4,
        learning_rate=1e-4,
        num_iterations=10
    )
    
    benchmark = Benchmark(config, device=device)
    
    all_results = {}
    
    print("\n" + "#"*80)
    print("开始运行实验...")
    print("#"*80)
    
    # 实验1
    all_results['training'] = benchmark. benchmark_training(seq_lengths=[512, 1024, 2048])
    
    # 实验2
    all_results['inference'] = benchmark.benchmark_inference(seq_lengths=[512, 1024], generate_lengths=[50, 100])
    
    # 实验3
    all_results['memory'] = benchmark.benchmark_memory(seq_lengths=[256, 512, 1024, 2048])
    
    # 实验4
    all_results['copy_task'] = benchmark.benchmark_copy_task(copy_lengths=[5, 10, 20], num_train_steps=300)
    
    # 实验5
    all_results['perplexity'] = benchmark. benchmark_perplexity(num_train_steps=500)
    
    # 实验6
    all_results['scaling'] = benchmark.benchmark_scaling()
    
    # 保存
    os.makedirs('./results', exist_ok=True)
    for name, df in all_results.items():
        df.to_csv('./results/{}_results.csv'.format(name), index=False)
        print("已保存:./results/{}_results.csv".format(name))
    
    # 绘图
    plot_results(all_results, save_dir='./figures')
    
    # 汇总
    print("\n" + "="*80)
    print("实验完成!  结果汇总")
    print("="*80)
    
    for name, df in all_results. items():
        print("\n【{}】".format(name))
        print(df.to_string(index=False))
    
    return all_results


if __name__ == "__main__":
    results = main()