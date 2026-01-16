# =====================================================
# 线性注意力实验 (使用本地 parquet 数据 + Torch Profiler)
# 修复版：通过调整超参数使各模型参数量对齐
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
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import inspect

# =====================================================
# Part 1: 注意力层
# =====================================================

class SoftmaxAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout
        
    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self. head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                              dropout_p=self. dropout if self.training else 0.0)
        return self. o_proj(out. transpose(1, 2).contiguous().view(B, L, self.hidden_size))


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
        self.dropout = dropout
        self.use_flash = False
        try: 
            from flash_attn import flash_attn_func
            self.flash_attn = flash_attn_func
            self. use_flash = True
        except: 
            pass
            
    def forward(self, x):
        B, L, _ = x. shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self. head_dim)
        if self.use_flash:
            out = self.flash_attn(q, k, v, dropout_p=self.dropout if self. training else 0.0, causal=True)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2)
        return self. o_proj(out.contiguous().view(B, L, self.hidden_size))


class FLALinearAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_fla = False
        
        try: 
            from fla.layers import LinearAttention
            self. attn = LinearAttention(
                hidden_size=hidden_size, 
                num_heads=num_heads, 
                mode='fused_chunk',
                expand_k=1.0,
                expand_v=1.0,
            )
            self.use_fla = True
        except:
            self._init_fallback()
    
    def _init_fallback(self):
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            
    def forward(self, x):
        if self.use_fla:
            result = self.attn(x)
            return result[0] if isinstance(result, tuple) else result
        
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self. head_dim)
        k = self.k_proj(x).view(B, L, self. num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self. head_dim)
        q, k = F. elu(q) + 1, F.elu(k) + 1
        
        kv = torch.einsum('blhd,blhe->blhde', k, v)
        kv_cumsum = torch.cumsum(kv, dim=1)
        k_cumsum = torch.cumsum(k, dim=1)
        out = torch.einsum('blhd,blhde->blhe', q, kv_cumsum)
        out = out / torch.einsum('blhd,blhd->blh', q, k_cumsum).unsqueeze(-1).clamp(min=1e-6)
        return self.o_proj(out. contiguous().view(B, L, self.hidden_size))


class FLAGatedDeltaNet(nn.Module):
    """GatedDeltaNet - 使用 FLA 库实现"""
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self. head_dim = hidden_size // num_heads
        self.use_fla = False
        
        try:
            from fla.layers import GatedDeltaNet
            self.attn = GatedDeltaNet(
                hidden_size=hidden_size, 
                num_heads=num_heads, 
                mode='chunk',
            )
            self.use_fla = True
        except: 
            self._init_fallback()
    
    def _init_fallback(self):
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.beta_proj = nn.Linear(self. hidden_size, self.num_heads, bias=True)
            
    def forward(self, x):
        if self.use_fla:
            result = self.attn(x)
            return result[0] if isinstance(result, tuple) else result
        
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        beta = torch.sigmoid(self.beta_proj(x)).unsqueeze(-1)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        
        kv = torch.einsum('blhd,blhe->blhde', k, v)
        kv_cumsum = torch.cumsum(kv * beta. unsqueeze(-1), dim=1)
        out = torch.einsum('blhd,blhde->blhe', q, kv_cumsum)
        return self.o_proj(out. contiguous().view(B, L, self.hidden_size))


class FLAKimiLinearAttention(nn.Module):
    """Kimi Linear Attention - 使用 FLA 库实现"""
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_fla = False
        self.fla_class_name = None
        try:
            from fla import layers as fla_layers
            self.attn = self._build_fla_attn(fla_layers, dropout)
            self.use_fla = True
        except Exception:
            self._init_fallback()
    
    def _build_fla_attn(self, fla_layers, dropout):
        candidates = [
            "KimiLinearAttention",
            "KimiLinearAttn",
            "KimiAttention",
            "KimiLinear",
        ]
        for name in candidates:
            if not hasattr(fla_layers, name):
                continue
            attn_cls = getattr(fla_layers, name)
            sig = inspect.signature(attn_cls.__init__)
            params = sig.parameters
            kwargs = {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
            }
            if "mode" in params:
                kwargs["mode"] = "fused_chunk"
            if "expand_k" in params:
                kwargs["expand_k"] = 1.0
            if "expand_v" in params:
                kwargs["expand_v"] = 1.0
            if "dropout" in params:
                kwargs["dropout"] = dropout
            if "dropout_p" in params:
                kwargs["dropout_p"] = dropout
            self.fla_class_name = name
            return attn_cls(**kwargs)
        raise ImportError("Kimi linear attention class not found in fla.layers")
    
    def _init_fallback(self):
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        if self.use_fla:
            result = self.attn(x)
            return result[0] if isinstance(result, tuple) else result
        
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        
        kv = torch.einsum('blhd,blhe->blhde', k, v)
        kv_cumsum = torch.cumsum(kv, dim=1)
        k_cumsum = torch.cumsum(k, dim=1)
        out = torch.einsum('blhd,blhde->blhe', q, kv_cumsum)
        out = out / torch.einsum('blhd,blhd->blh', q, k_cumsum).unsqueeze(-1).clamp(min=1e-6)
        return self.o_proj(out.contiguous().view(B, L, self.hidden_size))


# =====================================================
# Part 2: Transformer 模型 (支持不同注意力层配置)
# =====================================================

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_class=SoftmaxAttention, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = attn_class(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size), nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self. norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len, attn_class, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn. Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, attn_class=attn_class, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn. Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_params = sum(p.numel() for p in self.parameters())
        
    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.drop(self.embed(x) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


# =====================================================
# Part 3: 本地数据集
# =====================================================

class LocalWikiTextDataset(Dataset):
    def __init__(self, parquet_path, seq_len=256, char_to_idx=None):
        print("加载本地数据:  " + parquet_path)
        
        df = pd.read_parquet(parquet_path)
        texts = df['text'].tolist()
        text = '\n'.join([t for t in texts if t and t.strip()])
        
        print("文本长度:  " + str(len(text)) + " 字符")
        
        if char_to_idx is None:
            char_freq = {}
            for c in text:
                char_freq[c] = char_freq. get(c, 0) + 1
            chars = sorted(char_freq.keys(), key=lambda x: -char_freq[x])[:5000]
            chars = ['<unk>', '<pad>'] + list(chars)
            self.char_to_idx = {c: i for i, c in enumerate(chars)}
            self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        else:
            self.char_to_idx = char_to_idx
            self.idx_to_char = {i: c for c, i in char_to_idx.items()}
        
        self.vocab_size = len(self.char_to_idx)
        self. unk_idx = self.char_to_idx. get('<unk>', 0)
        self.pad_idx = self.char_to_idx. get('<pad>', 1)
        
        self.tokens = [self.char_to_idx. get(c, self.unk_idx) for c in text]
        self.seq_len = seq_len
        self.n_samples = max(1, (len(self.tokens) - 1) // seq_len)
        
        print("vocab_size: " + str(self.vocab_size) + " | tokens: " + str(len(self.tokens)) + " | samples: " + str(self.n_samples))
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, i):
        s = i * self.seq_len
        t = self.tokens[s:s + self.seq_len + 1]
        if len(t) < self.seq_len + 1:
            t = t + [self.pad_idx] * (self.seq_len + 1 - len(t))
        return torch. tensor(t[:-1], dtype=torch.long), torch.tensor(t[1:], dtype=torch.long)


class CopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, n_samples, copy_len):
        self.vocab_size = vocab_size
        self. seq_len = seq_len
        self.n_samples = n_samples
        self.copy_len = copy_len
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, i):
        seq = torch.randint(10, min(1000, self.vocab_size), (self.copy_len,))
        pad = torch.zeros(max(0, self.seq_len - 2 * self.copy_len - 1), dtype=torch.long)
        full = torch.cat([seq, torch.tensor([0]), pad, seq])[: self.seq_len]
        return full[:-1], full[1:]


class AssocRecallDataset(Dataset):
    def __init__(self, vocab_size, seq_len, n_samples, n_pairs):
        self.vocab_size = vocab_size
        self. seq_len = seq_len
        self.n_samples = n_samples
        self.n_pairs = n_pairs
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, i):
        keys = torch.randint(100, min(2000, self.vocab_size // 2), (self.n_pairs,))
        vals = torch.randint(min(2000, self.vocab_size // 2), min(4000, self.vocab_size), (self.n_pairs,))
        pairs = torch.stack([keys, vals], 1).flatten()
        qi = np.random.randint(self.n_pairs)
        pad = torch.zeros(max(0, self.seq_len - len(pairs) - 2), dtype=torch.long)
        full = torch.cat([pairs, pad, keys[qi: qi+1], vals[qi:qi+1]])[:self.seq_len]
        return full[:-1], full[1:]


# =====================================================
# Part 4: 训练器 (每个模型独立配置)
# =====================================================

class Trainer:
    # 注意力类型映射
    ATTN_CLASSES = {
        'Softmax': SoftmaxAttention,
        'Flash': FlashAttention,
        'Linear': FLALinearAttention,
        'GatedDeltaNet': FLAGatedDeltaNet,
        'KimiLinear': FLAKimiLinearAttention,
    }
    
    # 每个模型的独立配置，调整使参数量接近
    # GatedDeltaNet 参数量较大，需要减小 hidden_size 和 num_layers
    MODEL_CONFIGS = {
        'Softmax': {'hidden_size': 256, 'num_layers': 4, 'num_heads': 4},
        'Flash': {'hidden_size': 256, 'num_layers': 4, 'num_heads': 4},
        'Linear': {'hidden_size': 256, 'num_layers': 4, 'num_heads': 4},
        'GatedDeltaNet': {'hidden_size': 128, 'num_layers': 4, 'num_heads':  4},  # 减小 hidden_size
        'KimiLinear': {'hidden_size': 256, 'num_layers': 4, 'num_heads': 4},
    }
    
    def __init__(self, data_dir, seq_len=256, batch_size=16, lr=5e-4, device='cuda'):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self. lr = lr
        self.device = device
        self.dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        self.models = {}
        self.train_history = {}
        self.vocab_size = None
        
    def clear_cache(self):
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory(self):
        return torch.cuda.max_memory_allocated() / 1024**2 if self.device == 'cuda' else 0
    
    def reset_memory(self):
        if self.device == 'cuda': 
            torch.cuda.reset_peak_memory_stats()
    
    def get_model_config(self, attn_name):
        """获取特定注意力类型的模型配置"""
        return self.MODEL_CONFIGS.get(attn_name, {'hidden_size': 256, 'num_layers': 4, 'num_heads': 4})
    
    def create_model(self, attn_name, vocab_size, max_seq_len):
        """根据注意力类型创建模型"""
        config = self.get_model_config(attn_name)
        attn_class = self.ATTN_CLASSES[attn_name]
        return LanguageModel(
            vocab_size=vocab_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            max_seq_len=max_seq_len,
            attn_class=attn_class
        )
    
    def print_model_params(self):
        """打印各注意力机制的参数量对比"""
        print("\n" + "="*70)
        print("各模型配置与参数量对比")
        print("="*70)
        print("{:<15} {: >12} {:>10} {:>10} {:>15}".format(
            "Attention", "hidden_size", "num_layers", "num_heads", "Total Params"))
        print("-"*70)
        
        for attn_name in self.ATTN_CLASSES. keys():
            try:
                config = self.get_model_config(attn_name)
                model = self.create_model(attn_name, 5000, 512)
                num_params = model.num_params
                print("{:<15} {:>12} {: >10} {:>10} {: >15,}".format(
                    attn_name, config['hidden_size'], config['num_layers'], 
                    config['num_heads'], num_params))
                del model
            except Exception as e: 
                print("{:<15} Error: {}".format(attn_name, str(e)[:40]))
        print("="*70 + "\n")
    
    def calibrate_model_configs(self, target_params=5_000_000, vocab_size=5000):
        """
        自动校准各模型配置，使参数量接近目标值
        """
        print("\n" + "="*70)
        print("自动校准模型配置 (目标参数量: {: ,})".format(target_params))
        print("="*70)
        
        for attn_name, attn_class in self.ATTN_CLASSES.items():
            best_config = None
            best_diff = float('inf')
            
            # 搜索最优配置
            for hidden_size in [64, 96, 128, 160, 192, 224, 256, 288, 320]: 
                for num_layers in [2, 3, 4, 5, 6]: 
                    for num_heads in [2, 4, 8]: 
                        if hidden_size % num_heads != 0:
                            continue
                        try:
                            model = LanguageModel(
                                vocab_size=vocab_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_heads=num_heads,
                                max_seq_len=512,
                                attn_class=attn_class
                            )
                            diff = abs(model.num_params - target_params)
                            if diff < best_diff:
                                best_diff = diff
                                best_config = {
                                    'hidden_size': hidden_size,
                                    'num_layers': num_layers,
                                    'num_heads':  num_heads,
                                    'num_params': model.num_params
                                }
                            del model
                        except: 
                            continue
            
            if best_config:
                self.MODEL_CONFIGS[attn_name] = {
                    'hidden_size':  best_config['hidden_size'],
                    'num_layers':  best_config['num_layers'],
                    'num_heads':  best_config['num_heads']
                }
                print("{: <15}:  h={}, l={}, n={} -> {: ,} params (diff: {:,})".format(
                    attn_name, best_config['hidden_size'], best_config['num_layers'],
                    best_config['num_heads'], best_config['num_params'], 
                    best_config['num_params'] - target_params))
        
        print("="*70 + "\n")
    
    def pretrain_all(self, num_epochs=10):
        print("\n" + "="*80)
        print("阶段1: 预训练 (WikiText-2 本地数据)")
        print("="*80)
        
        checkpoint_dir = './checkpoints'
        checkpoint_paths = {
            attn_name: os.path.join(checkpoint_dir, attn_name + '.pt')
            for attn_name in self.ATTN_CLASSES.keys()
        }
        found_checkpoints = [name for name, path in checkpoint_paths.items() if os.path.isfile(path)]
        if found_checkpoints:
            print("检测到已有 checkpoints，将跳过对应模型的预训练:")
            print("  " + ", ".join(found_checkpoints))

        train_path = os.path.join(self.data_dir, 'train-00000-of-00001.parquet')
        val_path = os.path.join(self.data_dir, 'validation-00000-of-00001.parquet')
        
        train_data = LocalWikiTextDataset(train_path, self.seq_len)
        val_data = LocalWikiTextDataset(val_path, self.seq_len, train_data.char_to_idx)
        self.vocab_size = train_data.vocab_size
        
        train_loader = DataLoader(train_data, batch_size=self. batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=self. batch_size, num_workers=0, drop_last=True)
        
        print("\n基础配置:")
        print("  vocab_size: " + str(self.vocab_size))
        print("  seq_len: " + str(self. seq_len))
        print("  batch_size: " + str(self.batch_size))
        print("  lr: " + str(self.lr))
        
        # 打印各模型配置
        self. print_model_params()
        
        for attn_name, attn_class in self.ATTN_CLASSES.items():
            config = self.get_model_config(attn_name)
            checkpoint_path = checkpoint_paths.get(attn_name)
            print("\n" + "-"*60)
            print("训练:  {} (h={}, l={}, n={})".format(
                attn_name, config['hidden_size'], config['num_layers'], config['num_heads']))
            print("-"*60)

            if checkpoint_path and os.path.isfile(checkpoint_path):
                print("跳过预训练: 发现 checkpoint -> " + checkpoint_path)
                continue
            
            try:
                self.clear_cache()
                model = self.create_model(attn_name, self.vocab_size, self.seq_len + 100)
                model = model.to(self.device).to(self.dtype)
                print("参数量: " + str(round(model.num_params/1e6, 2)) + " M")
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.1)
                total_steps = len(train_loader) * num_epochs
                warmup = min(200, total_steps // 10)
                
                def lr_lambda(step):
                    if step < warmup:
                        return step / warmup
                    return 0.1 + 0.9 * (1 + math.cos(math. pi * (step - warmup) / (total_steps - warmup))) / 2
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                history = {'train_loss': [], 'val_loss': [], 'val_ppl': [], 'config': config}
                
                for epoch in range(num_epochs):
                    model.train()
                    losses = []
                    pbar = tqdm(train_loader, desc="Epoch " + str(epoch+1) + "/" + str(num_epochs))
                    for x, y in pbar:
                        x, y = x.to(self.device), y.to(self.device)
                        loss = F.cross_entropy(model(x).view(-1, self.vocab_size), y.view(-1))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        losses.append(loss.item())
                        pbar.set_postfix({'loss': round(np.mean(losses[-100:]), 4)})
                    
                    history['train_loss'].append(np.mean(losses))
                    
                    model.eval()
                    val_loss, val_n = 0, 0
                    with torch.no_grad():
                        for x, y in val_loader:
                            x, y = x.to(self.device), y.to(self.device)
                            val_loss += F.cross_entropy(model(x).view(-1, self.vocab_size),
                                                        y.view(-1), reduction='sum').item()
                            val_n += y.numel()
                    val_loss /= val_n
                    val_ppl = math.exp(min(val_loss, 10))
                    history['val_loss'].append(val_loss)
                    history['val_ppl'].append(val_ppl)
                    
                    print("Epoch " + str(epoch+1) + " | Train Loss: " + str(round(history['train_loss'][-1], 4)) +
                          " | Val Loss:  " + str(round(val_loss, 4)) + " | Val PPL: " + str(round(val_ppl, 2)))
                
                self.models[attn_name] = model
                self.train_history[attn_name] = history
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print("模型已保存")
                
            except Exception as e:
                print("Error: " + str(e))
                import traceback
                traceback.print_exc()
            finally:
                self.clear_cache()
        
        return self.train_history
    
    # =====================================================
    # Torch Profiler 性能分���
    # =====================================================
    def profile_attention(self, seq_lengths=[256, 512, 1024], num_steps=10):
        """使用 torch. profiler 对各注意力机制进行性能分析"""
        print("\n" + "="*80)
        print("Torch Profiler 性能分析")
        print("="*80)
        
        from torch.profiler import profile, record_function, ProfilerActivity
        
        os.makedirs('./profiler_results', exist_ok=True)
        profile_summary = []
        
        for seq_len in seq_lengths: 
            print("\n序列长度: " + str(seq_len))
            print("-"*60)
            
            for attn_name, attn_class in self.ATTN_CLASSES.items():
                try:
                    self.clear_cache()
                    config = self.get_model_config(attn_name)
                    
                    model = self.create_model(attn_name, self.vocab_size, seq_len + 100)
                    model = model.to(self.device).to(self.dtype)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
                    
                    x = torch.randint(0, self.vocab_size, (self.batch_size, seq_len), device=self.device)
                    y = torch.randint(0, self.vocab_size, (self. batch_size, seq_len), device=self.device)
                    
                    # 预热
                    model.train()
                    for _ in range(3):
                        loss = F. cross_entropy(model(x).view(-1, self.vocab_size), y.view(-1))
                        loss.backward()
                        optimizer. step()
                        optimizer.zero_grad()
                    
                    if self.device == 'cuda': 
                        torch.cuda.synchronize()
                    
                    self.reset_memory()
                    
                    # Profiler
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity. CUDA],
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                    ) as prof:
                        for _ in range(num_steps):
                            with record_function("forward"):
                                logits = model(x)
                            with record_function("loss"):
                                loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
                            with record_function("backward"):
                                loss.backward()
                            with record_function("optimizer"):
                                optimizer.step()
                                optimizer.zero_grad()
                    
                    # 保存 Chrome Trace
                    trace_file = './profiler_results/' + attn_name + '_seq' + str(seq_len) + '_trace.json'
                    prof.export_chrome_trace(trace_file)
                    
                    # 打印汇总表格
                    print("\n  [" + attn_name + "] seq_len=" + str(seq_len))
                    table_str = prof.key_averages().table(sort_by="cpu_time_total", row_limit=15)
                    print(table_str)
                    
                    # 保存表格到文件
                    table_file = './profiler_results/' + attn_name + '_seq' + str(seq_len) + '_table.txt'
                    with open(table_file, 'w') as f:
                        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
                    
                    # 提取关键指标
                    key_avg = prof. key_averages()
                    total_cpu_time = sum([item.cpu_time_total for item in key_avg]) / 1000  # ms
                    
                    # 获取 CUDA 时间和内存
                    total_cuda_time = 0
                    total_memory = 0
                    for item in key_avg: 
                        if hasattr(item, 'self_cuda_time_total'):
                            total_cuda_time += item.self_cuda_time_total
                        elif hasattr(item, 'cuda_time'):
                            total_cuda_time += item.cuda_time
                        
                        if hasattr(item, 'self_cuda_memory_usage'):
                            total_memory = max(total_memory, item.self_cuda_memory_usage)
                        elif hasattr(item, 'cuda_memory_usage'):
                            total_memory = max(total_memory, item.cuda_memory_usage)
                    
                    total_cuda_time = total_cuda_time / 1000  # ms
                    total_memory = total_memory / 1024**2  # MB
                    
                    # 也获取实际 GPU 内存
                    actual_memory = self.get_memory()
                    
                    profile_summary.append({
                        'attention': attn_name,
                        'seq_len': seq_len,
                        'hidden_size': config['hidden_size'],
                        'num_layers': config['num_layers'],
                        'cuda_time_ms': round(total_cuda_time / num_steps, 2) if total_cuda_time > 0 else round(total_cpu_time / num_steps, 2),
                        'cpu_time_ms': round(total_cpu_time / num_steps, 2),
                        'peak_memory_mb': round(actual_memory, 2),
                    })
                    
                    print("  Trace 已保存:  " + trace_file)
                    print("  Peak Memory: " + str(round(actual_memory, 2)) + " MB")
                    
                except Exception as e:
                    print("  " + attn_name + " | Error: " + str(e)[:80])
                    import traceback
                    traceback. print_exc()
                finally: 
                    self.clear_cache()
        
        # 保存汇总
        summary_df = pd.DataFrame(profile_summary)
        summary_df. to_csv('./profiler_results/profile_summary.csv', index=False)
        print("\n汇总已保存:  ./profiler_results/profile_summary.csv")
        
        # 绘制 Profiler 结果图
        self.plot_profiler_results(summary_df)
        
        return summary_df
    
    def plot_profiler_results(self, df, save_dir='./figures'):
        """绘制 Profiler 结果图表"""
        os.makedirs(save_dir, exist_ok=True)
        colors = {'Softmax': '#1f77b4', 'Flash': '#ff7f0e', 'Linear': '#2ca02c', 'GatedDeltaNet': '#d62728', 'KimiLinear': '#9467bd'}
        markers = {'Softmax': 'o', 'Flash': 's', 'Linear': '^', 'GatedDeltaNet': 'D', 'KimiLinear': 'X'}
        
        # 图1:  CUDA/CPU 时间 vs 序列长度
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for attn in df['attention'].unique():
            d = df[df['attention'] == attn]. sort_values('seq_len')
            c, m = colors. get(attn, 'gray'), markers.get(attn, 'o')
            axes[0].plot(d['seq_len'], d['cuda_time_ms'], marker=m, color=c, label=attn, lw=2, ms=8)
            axes[1].plot(d['seq_len'], d['cpu_time_ms'], marker=m, color=c, label=attn, lw=2, ms=8)
            axes[2].plot(d['seq_len'], d['peak_memory_mb'], marker=m, color=c, label=attn, lw=2, ms=8)
        
        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('CUDA Time (ms)')
        axes[0].set_title('CUDA Time per Step')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Sequence Length')
        axes[1].set_ylabel('CPU Time (ms)')
        axes[1].set_title('CPU Time per Step')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Sequence Length')
        axes[2].set_ylabel('Peak Memory (MB)')
        axes[2].set_title('Peak Memory Usage')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir + '/profiler_time_memory.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/profiler_time_memory.pdf', bbox_inches='tight')
        plt.close()
        print("Saved:  profiler_time_memory.png/pdf")
        
        # 图2: 柱状图对比 (固定序列长度)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 选择中间的序列长度
        seq_lens = sorted(df['seq_len'].unique())
        mid_seq_len = seq_lens[len(seq_lens) // 2] if len(seq_lens) > 0 else seq_lens[0]
        df_mid = df[df['seq_len'] == mid_seq_len]
        
        x = np.arange(len(df_mid))
        width = 0.6
        bar_colors = [colors. get(a, 'gray') for a in df_mid['attention']]
        
        # 时间柱状图
        bars1 = axes[0].bar(x, df_mid['cuda_time_ms'], width, color=bar_colors)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df_mid['attention'])
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Step Time Comparison (seq_len=' + str(mid_seq_len) + ')')
        for bar, val in zip(bars1, df_mid['cuda_time_ms']):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(round(val, 1)), ha='center', fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 内存柱状图
        bars2 = axes[1].bar(x, df_mid['peak_memory_mb'], width, color=bar_colors)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_mid['attention'])
        axes[1].set_ylabel('Memory (MB)')
        axes[1].set_title('Memory Comparison (seq_len=' + str(mid_seq_len) + ')')
        for bar, val in zip(bars2, df_mid['peak_memory_mb']):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        str(round(val, 0)), ha='center', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir + '/profiler_comparison_bar.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/profiler_comparison_bar.pdf', bbox_inches='tight')
        plt.close()
        print("Saved: profiler_comparison_bar.png/pdf")
        
        # 图3: 时间复杂度分析 (log-log 图)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for attn in df['attention'].unique():
            d = df[df['attention'] == attn].sort_values('seq_len')
            c, m = colors.get(attn, 'gray'), markers.get(attn, 'o')
            ax.loglog(d['seq_len'], d['cuda_time_ms'], marker=m, color=c, label=attn, lw=2, ms=8)
        
        # 添加参考线 O(n) 和 O(n^2)
        seq_range = np.array(sorted(df['seq_len'].unique()))
        base_time = df[df['seq_len'] == seq_range[0]]['cuda_time_ms'].mean()
        ax.loglog(seq_range, base_time * (seq_range / seq_range[0]), 'k--', alpha=0.5, label='O(n)')
        ax.loglog(seq_range, base_time * (seq_range / seq_range[0])**2, 'k:', alpha=0.5, label='O(n²)')
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Time Complexity Analysis (Log-Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir + '/profiler_complexity.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/profiler_complexity.pdf', bbox_inches='tight')
        plt.close()
        print("Saved: profiler_complexity.png/pdf")
    
    def profile_attention_detailed(self, seq_len=512):
        """详细的单次 profile，生成火焰图数据"""
        print("\n" + "="*80)
        print("详细性能分析 (seq_len=" + str(seq_len) + ")")
        print("="*80)
        
        from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
        
        os.makedirs('./profiler_logs', exist_ok=True)
        os.makedirs('./profiler_results', exist_ok=True)
        
        for attn_name, attn_class in self.ATTN_CLASSES.items():
            try:
                self.clear_cache()
                print("\n分析:  " + attn_name)
                
                model = self.create_model(attn_name, self.vocab_size, seq_len + 100)
                model = model.to(self.device).to(self.dtype)
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
                
                x = torch.randint(0, self.vocab_size, (self. batch_size, seq_len), device=self.device)
                y = torch.randint(0, self.vocab_size, (self.batch_size, seq_len), device=self.device)
                
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
                    on_trace_ready=tensorboard_trace_handler('./profiler_logs/' + attn_name),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                    for step in range(10):
                        loss = F.cross_entropy(model(x).view(-1, self.vocab_size), y.view(-1))
                        loss.backward()
                        optimizer. step()
                        optimizer.zero_grad()
                        prof.step()
                
                print("  TensorBoard 日志已保存:  ./profiler_logs/" + attn_name)
                
                # 导出 stacks
                try:
                    stacks_file = './profiler_results/' + attn_name + '_stacks.txt'
                    prof.export_stacks(stacks_file, metric="self_cpu_time_total")
                    print("  Stacks 已保存: " + stacks_file)
                except Exception as e:
                    print("  Stacks 导出跳过: " + str(e)[:50])
                
            except Exception as e:
                print("  " + attn_name + " | Error: " + str(e)[:60])
            finally:
                self.clear_cache()
        
        print("\n可以使用以下命令查看 TensorBoard:")
        print("  tensorboard --logdir=./profiler_logs")
    
    def test_training_efficiency(self, seq_lengths=[256, 512, 1024]):
        print("\n" + "="*80)
        print("测试1: 训练效率")
        print("="*80)
        results = []
        for seq_len in seq_lengths: 
            print("\n序列长度: " + str(seq_len))
            print("-"*60)
            for attn_name, attn_class in self.ATTN_CLASSES.items():
                try:
                    self.clear_cache()
                    self.reset_memory()
                    config = self.get_model_config(attn_name)
                    
                    model = self. create_model(attn_name, self.vocab_size, seq_len + 100)
                    model = model.to(self.device).to(self.dtype)
                    optimizer = torch. optim.AdamW(model.parameters(), lr=self.lr)
                    x = torch.randint(0, self.vocab_size, (self.batch_size, seq_len), device=self.device)
                    y = torch.randint(0, self.vocab_size, (self.batch_size, seq_len), device=self.device)
                    
                    model.train()
                    for _ in range(5):
                        loss = F. cross_entropy(model(x).view(-1, self.vocab_size), y.view(-1))
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    if self.device == 'cuda': 
                        torch.cuda.synchronize()
                    
                    self.reset_memory()
                    times = []
                    for _ in range(20):
                        if self.device == 'cuda': 
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        loss = F.cross_entropy(model(x).view(-1, self.vocab_size), y.view(-1))
                        loss.backward()
                        optimizer. step()
                        optimizer.zero_grad()
                        if self.device == 'cuda': 
                            torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1000)
                    
                    r = {
                        'attention': attn_name, 
                        'seq_len': seq_len, 
                        'hidden_size': config['hidden_size'],
                        'num_params': model.num_params,
                        'step_time_ms': np.mean(times),
                        'step_std_ms': np.std(times),
                        'throughput':  (self.batch_size * seq_len) / (np.mean(times) / 1000),
                        'memory_mb': self.get_memory()
                    }
                    results.append(r)
                    print("  " + attn_name + " | " + str(round(r['step_time_ms'], 2)) + "±" + str(round(r['step_std_ms'], 2)) + "ms | " +
                          str(int(r['throughput'])) + " tok/s | " + str(round(r['memory_mb'], 1)) + "MB")
                except Exception as e:
                    print("  " + attn_name + " | Error: " + str(e)[:50])
                finally:
                    self.clear_cache()
        return pd.DataFrame(results)
    
    def test_memory_scaling(self, seq_lengths=[256, 512, 1024, 2048, 4096]):
        print("\n" + "="*80)
        print("测试2: 内存扩展性")
        print("="*80)
        results = []
        for seq_len in seq_lengths:
            print("\n序列长度: " + str(seq_len))
            print("-"*60)
            for attn_name, attn_class in self.ATTN_CLASSES.items():
                try:
                    self. clear_cache()
                    self.reset_memory()
                    config = self.get_model_config(attn_name)
                    
                    attn = attn_class(config['hidden_size'], config['num_heads']).to(self.device).to(self.dtype)
                    x = torch.randn(self.batch_size, seq_len, config['hidden_size'],
                                    device=self.device, dtype=self.dtype, requires_grad=True)
                    
                    self.reset_memory()
                    out = attn(x)
                    fwd = self.get_memory()
                    self.reset_memory()
                    out. sum().backward()
                    bwd = self.get_memory()
                    
                    results.append({
                        'attention': attn_name, 
                        'seq_len': seq_len,
                        'hidden_size':  config['hidden_size'],
                        'forward_mb': fwd, 
                        'backward_mb': bwd, 
                        'total_mb': max(fwd, bwd)
                    })
                    print("  " + attn_name + " | Fwd: " + str(round(fwd, 1)) + "MB | Bwd: " + str(round(bwd, 1)) + "MB")
                except torch.cuda.OutOfMemoryError:
                    print("  " + attn_name + " | OOM")
                    results.append({'attention': attn_name, 'seq_len': seq_len, 'total_mb': float('inf')})
                except Exception as e:
                    print("  " + attn_name + " | Error: " + str(e)[:50])
                finally:
                    self.clear_cache()
        return pd.DataFrame(results)
    
    def test_copy_task(self, copy_lengths=[5, 10, 20, 30, 50], num_epochs=5):
        print("\n" + "="*80)
        print("测试3: Copy Task (记忆能力)")
        print("="*80)
        results = []
        for copy_len in copy_lengths:
            print("\n复制长度: " + str(copy_len))
            print("-"*60)
            seq_len = copy_len * 3 + 20
            train_loader = DataLoader(CopyDataset(self.vocab_size, seq_len, 5000, copy_len),
                                       batch_size=self.batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(CopyDataset(self.vocab_size, seq_len, 1000, copy_len),
                                      batch_size=self. batch_size, drop_last=True)
            
            for attn_name, attn_class in self.ATTN_CLASSES.items():
                try:
                    self. clear_cache()
                    config = self.get_model_config(attn_name)
                    
                    model = self. create_model(attn_name, self.vocab_size, seq_len + 100)
                    model = model.to(self.device).to(self.dtype)
                    optimizer = torch. optim.AdamW(model.parameters(), lr=self.lr)
                    
                    model.train()
                    losses = []
                    for _ in range(num_epochs):
                        for x, y in train_loader: 
                            x, y = x.to(self. device), y.to(self.device)
                            loss = F.cross_entropy(model(x).view(-1, self.vocab_size), y.view(-1))
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            optimizer. zero_grad()
                            losses.append(loss.item())
                    
                    model.eval()
                    correct, total = 0, 0
                    with torch.no_grad():
                        for x, y in test_loader:
                            x, y = x.to(self.device), y.to(self.device)
                            preds = model(x).argmax(-1)
                            start = seq_len - copy_len - 1
                            correct += (preds[: , start:] == y[:, start:]).sum().item()
                            total += y[: , start:].numel()
                    
                    acc = correct / total * 100
                    results.append({
                        'attention': attn_name, 
                        'copy_len': copy_len,
                        'hidden_size': config['hidden_size'],
                        'accuracy': acc, 
                        'final_loss': np.mean(losses[-100:])
                    })
                    print("  " + attn_name + " | Acc: " + str(round(acc, 2)) + "%")
                except Exception as e:
                    print("  " + attn_name + " | Error: " + str(e)[:50])
                finally:
                    self.clear_cache()
        return pd.DataFrame(results)
    
    def test_associative_recall(self, n_pairs_list=[3, 5, 8, 10, 15], num_epochs=5):
        print("\n" + "="*80)
        print("测试4: 关联记忆")
        print("="*80)
        results = []
        seq_len = 256
        for n_pairs in n_pairs_list:
            print("\nKV对数: " + str(n_pairs))
            print("-"*60)
            train_loader = DataLoader(AssocRecallDataset(self.vocab_size, seq_len, 5000, n_pairs),
                                       batch_size=self.batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(AssocRecallDataset(self.vocab_size, seq_len, 1000, n_pairs),
                                      batch_size=self.batch_size, drop_last=True)
            
            for attn_name, attn_class in self.ATTN_CLASSES.items():
                try:
                    self. clear_cache()
                    config = self.get_model_config(attn_name)
                    
                    model = self. create_model(attn_name, self.vocab_size, seq_len + 100)
                    model = model.to(self.device).to(self.dtype)
                    optimizer = torch. optim.AdamW(model.parameters(), lr=self.lr)
                    
                    model.train()
                    losses = []
                    for _ in range(num_epochs):
                        for x, y in train_loader:
                            x, y = x.to(self.device), y.to(self.device)
                            loss = F.cross_entropy(model(x).view(-1, self.vocab_size), y.view(-1))
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            losses.append(loss.item())
                    
                    model.eval()
                    correct, total = 0, 0
                    with torch.no_grad():
                        for x, y in test_loader:
                            x, y = x.to(self.device), y.to(self.device)
                            preds = model(x)[: , -1, : ].argmax(-1)
                            correct += (preds == y[: , -1]).sum().item()
                            total += y. shape[0]
                    
                    acc = correct / total * 100
                    results.append({
                        'attention': attn_name, 
                        'n_pairs': n_pairs,
                        'hidden_size':  config['hidden_size'],
                        'accuracy': acc, 
                        'final_loss': np.mean(losses[-100:])
                    })
                    print("  " + attn_name + " | Acc: " + str(round(acc, 2)) + "%")
                except Exception as e:
                    print("  " + attn_name + " | Error: " + str(e)[:50])
                finally:
                    self.clear_cache()
        return pd.DataFrame(results)
    
    def test_perplexity_summary(self):
        print("\n" + "="*80)
        print("测试5: 困惑度汇总")
        print("="*80)
        results = []
        for name, h in self.train_history.items():
            config = h. get('config', {})
            r = {
                'attention': name, 
                'hidden_size': config.get('hidden_size', 'N/A'),
                'num_layers': config.get('num_layers', 'N/A'),
                'final_train_loss': h['train_loss'][-1],
                'final_val_loss': h['val_loss'][-1], 
                'final_ppl': h['val_ppl'][-1],
                'best_ppl': min(h['val_ppl'])
            }
            results.append(r)
            print(name + " | Final PPL: " + str(round(r['final_ppl'], 2)) +
                  " | Best PPL: " + str(round(r['best_ppl'], 2)))
        return pd.DataFrame(results)
    
    def test_model_scaling(self):
        print("\n" + "="*80)
        print("测试6: 模型扩展性")
        print("="*80)
        
        # 为每种注意力类型定义不同规模的配置
        scale_configs = {
            'Softmax': [
                {'h':  128, 'l': 2, 'n': 4, 'name': '~1M'},
                {'h': 256, 'l': 4, 'n': 8, 'name': '~5M'},
                {'h': 384, 'l': 6, 'n': 8, 'name': '~15M'},
                {'h': 512, 'l': 8, 'n': 8, 'name': '~30M'},
            ],
            'Flash': [
                {'h': 128, 'l': 2, 'n': 4, 'name': '~1M'},
                {'h': 256, 'l': 4, 'n': 8, 'name': '~5M'},
                {'h': 384, 'l': 6, 'n': 8, 'name': '~15M'},
                {'h': 512, 'l': 8, 'n': 8, 'name': '~30M'},
            ],
            'Linear': [
                {'h': 128, 'l': 2, 'n': 4, 'name': '~1M'},
                {'h': 256, 'l': 4, 'n': 8, 'name': '~5M'},
                {'h': 384, 'l': 6, 'n': 8, 'name': '~15M'},
                {'h': 512, 'l': 8, 'n': 8, 'name': '~30M'},
            ],
            'GatedDeltaNet': [
                {'h': 64, 'l': 2, 'n': 4, 'name': '~1M'},
                {'h': 128, 'l': 4, 'n': 4, 'name': '~5M'},
                {'h': 192, 'l': 6, 'n': 4, 'name': '~15M'},
                {'h': 256, 'l': 8, 'n': 4, 'name': '~30M'},
            ],
            'KimiLinear': [
                {'h': 128, 'l': 2, 'n': 4, 'name': '~1M'},
                {'h': 256, 'l': 4, 'n': 8, 'name': '~5M'},
                {'h': 384, 'l': 6, 'n': 8, 'name': '~15M'},
                {'h': 512, 'l': 8, 'n': 8, 'name': '~30M'},
            ],
        }
        
        results = []
        seq_len = 512
        
        for attn_name, attn_class in self.ATTN_CLASSES.items():
            configs = scale_configs.get(attn_name, scale_configs['Softmax'])
            
            for cfg in configs:
                print("\n" + attn_name + " 规模: " + cfg['name'])
                print("-"*60)
                try:
                    self.clear_cache()
                    self.reset_memory()
                    
                    model = LanguageModel(
                        self.vocab_size, cfg['h'], cfg['l'], cfg['n'],
                        seq_len + 100, attn_class
                    ).to(self.device).to(self.dtype)
                    
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
                    x = torch.randint(0, self.vocab_size, (self.batch_size, seq_len), device=self.device)
                    y = torch.randint(0, self.vocab_size, (self.batch_size, seq_len), device=self.device)
                    
                    for _ in range(3):
                        loss = F.cross_entropy(model(x).view(-1, self.vocab_size), y.view(-1))
                        loss.backward()
                        optimizer.step()
                        optimizer. zero_grad()
                    if self.device == 'cuda': 
                        torch.cuda.synchronize()
                    
                    self.reset_memory()
                    times = []
                    for _ in range(10):
                        if self.device == 'cuda': 
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        loss = F.cross_entropy(model(x).view(-1, self.vocab_size), y.view(-1))
                        loss.backward()
                        optimizer. step()
                        optimizer.zero_grad()
                        if self.device == 'cuda': 
                            torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1000)
                    
                    r = {
                        'attention':  attn_name, 
                        'model_size': cfg['name'], 
                        'hidden_size': cfg['h'],
                        'num_layers':  cfg['l'],
                        'num_heads': cfg['n'],
                        'num_params': model.num_params,
                        'step_time_ms': np.mean(times), 
                        'memory_mb': self.get_memory(),
                        'throughput':  (self.batch_size * seq_len) / (np.mean(times) / 1000)
                    }
                    results.append(r)
                    print("  " + attn_name + " | " + str(round(model.num_params/1e6, 2)) + "M | " +
                          str(round(r['step_time_ms'], 2)) + "ms | " + str(round(r['memory_mb'], 1)) + "MB")
                except torch.cuda.OutOfMemoryError:
                    print("  " + attn_name + " | OOM")
                except Exception as e:
                    print("  " + attn_name + " | Error: " + str(e)[:50])
                finally: 
                    self.clear_cache()
        
        return pd. DataFrame(results)


# =====================================================
# Part 5: 绘图
# =====================================================

def plot_results(results, history, save_dir='./figures'):
    os.makedirs(save_dir, exist_ok=True)
    colors = {'Softmax': '#1f77b4', 'Flash': '#ff7f0e', 'Linear': '#2ca02c', 'GatedDeltaNet': '#d62728', 'KimiLinear': '#9467bd'}
    markers = {'Softmax': 'o', 'Flash': 's', 'Linear': '^', 'GatedDeltaNet': 'D', 'KimiLinear': 'X'}
    
    # 1. 训练曲线
    if history: 
        fig, axes = plt. subplots(1, 2, figsize=(12, 4))
        for name, h in history.items():
            c, m = colors. get(name, 'gray'), markers.get(name, 'o')
            axes[0].plot(range(1, len(h['train_loss'])+1), h['train_loss'], marker=m, color=c, label=name, lw=2, ms=6)
            axes[1].plot(range(1, len(h['val_ppl'])+1), h['val_ppl'], marker=m, color=c, label=name, lw=2, ms=6)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train Loss'); axes[0].set_title('Training Loss')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation Perplexity'); axes[1].set_title('Validation Perplexity')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir + '/training_curves.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/training_curves.pdf', bbox_inches='tight')
        plt.close()
        print("Saved:  training_curves.png/pdf")
    
    # 2. 训练效率
    if 'efficiency' in results:
        df = results['efficiency']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for attn in df['attention'].unique():
            d = df[df['attention'] == attn]
            c, m = colors.get(attn, 'gray'), markers.get(attn, 'o')
            axes[0].plot(d['seq_len'], d['step_time_ms'], marker=m, color=c, label=attn, lw=2, ms=8)
            axes[1].plot(d['seq_len'], d['throughput'], marker=m, color=c, label=attn, lw=2, ms=8)
            axes[2].plot(d['seq_len'], d['memory_mb'], marker=m, color=c, label=attn, lw=2, ms=8)
        for i, (yl, ti) in enumerate([('Step Time (ms)', 'Training Step Time'),
                                       ('Throughput (tokens/s)', 'Training Throughput'),
                                       ('Memory (MB)', 'GPU Memory Usage')]):
            axes[i].set_xlabel('Sequence Length'); axes[i].set_ylabel(yl); axes[i].set_title(ti)
            axes[i].legend(); axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir + '/efficiency.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/efficiency.pdf', bbox_inches='tight')
        plt.close()
        print("Saved: efficiency.png/pdf")
    
    # 3. 内存扩展性
    if 'memory' in results:
        df = results['memory']
        df = df[df['total_mb'] < 1e9]
        fig, ax = plt.subplots(figsize=(8, 5))
        for attn in df['attention'].unique():
            d = df[df['attention'] == attn]
            ax.plot(d['seq_len'], d['total_mb'], marker=markers.get(attn, 'o'),
                    color=colors.get(attn, 'gray'), label=attn, lw=2, ms=8)
        ax.set_xlabel('Sequence Length'); ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Scaling with Sequence Length')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir + '/memory.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/memory.pdf', bbox_inches='tight')
        plt.close()
        print("Saved: memory.png/pdf")
    
    # 4. Copy Task
    if 'copy' in results:
        df = results['copy']
        fig, ax = plt.subplots(figsize=(8, 5))
        for attn in df['attention'].unique():
            d = df[df['attention'] == attn]
            ax.plot(d['copy_len'], d['accuracy'], marker=markers.get(attn, 'o'),
                    color=colors.get(attn, 'gray'), label=attn, lw=2, ms=8)
        ax.set_xlabel('Copy Length'); ax.set_ylabel('Accuracy (%)')
        ax.set_title('Copy Task (Memory Retrieval Ability)')
        ax.legend(); ax.set_ylim([0, 105]); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir + '/copy_task.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/copy_task.pdf', bbox_inches='tight')
        plt.close()
        print("Saved: copy_task.png/pdf")
    
    # 5. 关联记忆
    if 'assoc' in results:
        df = results['assoc']
        fig, ax = plt.subplots(figsize=(8, 5))
        for attn in df['attention'].unique():
            d = df[df['attention'] == attn]
            ax. plot(d['n_pairs'], d['accuracy'], marker=markers.get(attn, 'o'),
                    color=colors.get(attn, 'gray'), label=attn, lw=2, ms=8)
        ax.set_xlabel('Number of Key-Value Pairs'); ax.set_ylabel('Accuracy (%)')
        ax.set_title('Associative Recall (In-Context Learning)')
        ax.legend(); ax.set_ylim([0, 105]); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir + '/associative_recall.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/associative_recall.pdf', bbox_inches='tight')
        plt.close()
        print("Saved: associative_recall.png/pdf")
    
    # 6. 困惑度
    if 'ppl' in results:
        df = results['ppl']
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(df))
        bar_colors = [colors.get(a, 'gray') for a in df['attention']]
        bars = ax.bar(x, df['final_ppl'], color=bar_colors)
        ax.set_xticks(x); ax.set_xticklabels(df['attention'])
        ax.set_ylabel('Perplexity'); ax.set_title('Language Modeling Perplexity (Lower is Better)')
        for bar, val in zip(bars, df['final_ppl']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(round(val, 2)), ha='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_dir + '/perplexity.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/perplexity.pdf', bbox_inches='tight')
        plt.close()
        print("Saved:  perplexity.png/pdf")
    
    # 7. 模型扩展性
    if 'scaling' in results:
        df = results['scaling']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for attn in df['attention'].unique():
            d = df[df['attention'] == attn]. sort_values('num_params')
            c, m = colors.get(attn, 'gray'), markers.get(attn, 'o')
            axes[0].plot(d['num_params']/1e6, d['step_time_ms'], marker=m, color=c, label=attn, lw=2, ms=8)
            axes[1].plot(d['num_params']/1e6, d['throughput'], marker=m, color=c, label=attn, lw=2, ms=8)
        axes[0].set_xlabel('Parameters (Million)'); axes[0].set_ylabel('Step Time (ms)')
        axes[0].set_title('Training Time vs Model Size'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel('Parameters (Million)'); axes[1].set_ylabel('Throughput (tokens/s)')
        axes[1].set_title('Throughput vs Model Size'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir + '/scaling.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/scaling.pdf', bbox_inches='tight')
        plt.close()
        print("Saved: scaling.png/pdf")
    
    # 8. Profiler 汇总图
    if 'profile' in results:
        df = results['profile']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for attn in df['attention'].unique():
            d = df[df['attention'] == attn]
            c, m = colors.get(attn, 'gray'), markers.get(attn, 'o')
            axes[0]. plot(d['seq_len'], d['cuda_time_ms'], marker=m, color=c, label=attn, lw=2, ms=8)
            axes[1].plot(d['seq_len'], d['peak_memory_mb'], marker=m, color=c, label=attn, lw=2, ms=8)
        axes[0].set_xlabel('Sequence Length'); axes[0].set_ylabel('CUDA Time (ms)')
        axes[0].set_title('CUDA Time per Step'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel('Sequence Length'); axes[1].set_ylabel('Peak Memory (MB)')
        axes[1].set_title('Peak Memory Usage'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir + '/profiler_summary.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_dir + '/profiler_summary.pdf', bbox_inches='tight')
        plt.close()
        print("Saved: profiler_summary.png/pdf")
    
    # 9. 综合对比雷达图
    if 'ppl' in results and 'efficiency' in results:
        try:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            categories = ['Perplexity\n(lower=better)', 'Speed\n(higher=better)', 
                         'Memory Eff.\n(lower=better)', 'Copy Task\n(higher=better)', 
                         'Assoc Recall\n(higher=better)']
            num_vars = len(categories)
            angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
            angles += angles[:1]
            
            for attn_name in results['ppl']['attention'].unique():
                values = []
                
                # PPL (归一化，越低越好，所以取倒数)
                ppl_val = results['ppl'][results['ppl']['attention'] == attn_name]['final_ppl']. values[0]
                values. append(100 / ppl_val)
                
                # Speed (throughput)
                if 'efficiency' in results:
                    eff_df = results['efficiency']
                    speed = eff_df[(eff_df['attention'] == attn_name) & (eff_df['seq_len'] == 256)]['throughput'].values
                    values.append(speed[0] / 10000 if len(speed) > 0 else 0)
                else:
                    values.append(0)
                
                # Memory (越低越好，取倒数)
                if 'memory' in results: 
                    mem_df = results['memory']
                    mem = mem_df[(mem_df['attention'] == attn_name) & (mem_df['seq_len'] == 256)]['total_mb'].values
                    values.append(100 / mem[0] if len(mem) > 0 and mem[0] > 0 else 0)
                else: 
                    values.append(0)
                
                # Copy Task
                if 'copy' in results:
                    copy_df = results['copy']
                    copy_acc = copy_df[(copy_df['attention'] == attn_name) & (copy_df['copy_len'] == 20)]['accuracy'].values
                    values.append(copy_acc[0] / 100 if len(copy_acc) > 0 else 0)
                else: 
                    values.append(0)
                
                # Assoc Recall
                if 'assoc' in results:
                    assoc_df = results['assoc']
                    assoc_acc = assoc_df[(assoc_df['attention'] == attn_name) & (assoc_df['n_pairs'] == 5)]['accuracy'].values
                    values.append(assoc_acc[0] / 100 if len(assoc_acc) > 0 else 0)
                else:
                    values. append(0)
                
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=attn_name, color=colors.get(attn_name, 'gray'))
                ax.fill(angles, values, alpha=0.1, color=colors.get(attn_name, 'gray'))
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            plt.title('Comprehensive Comparison', size=14, y=1.08)
            plt.tight_layout()
            plt.savefig(save_dir + '/radar_comparison.png', dpi=150, bbox_inches='tight')
            plt.savefig(save_dir + '/radar_comparison.pdf', bbox_inches='tight')
            plt.close()
            print("Saved: radar_comparison.png/pdf")
        except Exception as e:
            print("Radar chart error:  " + str(e))
    
    print("\n所有图表已保存到 " + save_dir)


# =====================================================
# Main
# =====================================================

def main():
    print("="*80)
    print("线性注意力机制综合实验 (含 Torch Profiler)")
    print("="*80)
    
    device = 'cuda' if torch.cuda. is_available() else 'cpu'
    if device == 'cuda':
        print("GPU:  " + torch.cuda.get_device_name(0))
        print("显存: " + str(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)) + " GB")
    print("PyTorch:  " + torch.__version__)
    
    print("\n检查依赖...")
    try:
        from fla.layers import GatedDeltaNet, LinearAttention
        print("FLA:  OK")
    except:
        print("FLA: Not found (will use fallback)")
    
    try:
        from fla import layers as fla_layers
        kimi_candidates = [
            "KimiDeltaAttention",
        ]
        if any(hasattr(fla_layers, name) for name in kimi_candidates):
            print("FLA Kimi Linear: OK")
        else:
            print("FLA Kimi Linear: Not found (will use fallback)")
    except:
        print("FLA Kimi Linear: Not found (will use fallback)")
    
    try:
        from flash_attn import flash_attn_func
        print("Flash Attention: OK")
    except:
        print("Flash Attention:  Not found (will use PyTorch SDPA)")
    
    data_dir = '/root/autodl-tmp/GatedDeltaNet'
    
    trainer = Trainer(
        data_dir=data_dir,
        seq_len=256,
        batch_size=128,
        lr=5e-4,
        device=device
    )
    
    # 可选：自动校准模型配置使参数量接近
    # trainer.calibrate_model_configs(target_params=5_000_000, vocab_size=5000)
    
    # 打印当前配置
    print("\n当前模型配置:")
    for name, cfg in trainer.MODEL_CONFIGS.items():
        print("  {}: {}".format(name, cfg))
    
    # ========== 阶段1: 预训练 ==========
    history = trainer.pretrain_all(num_epochs=5)
    
    # ========== 阶段2: 测试 ==========
    results = {}
    results['efficiency'] = trainer.test_training_efficiency([256, 512, 1024])
    results['memory'] = trainer.test_memory_scaling([256, 512, 1024, 2048])
    results['copy'] = trainer.test_copy_task([5, 10, 20, 30, 50], num_epochs=5)
    results['assoc'] = trainer.test_associative_recall([3, 5, 8, 10, 15], num_epochs=5)
    results['ppl'] = trainer.test_perplexity_summary()
    results['scaling'] = trainer.test_model_scaling()
    
    # ========== 阶段3: Torch Profiler ==========
    results['profile'] = trainer.profile_attention(seq_lengths=[256, 512, 1024])
    trainer.profile_attention_detailed(seq_len=512)
    
    # 保存结果
    os.makedirs('./results', exist_ok=True)
    for name, df in results.items():
        df.to_csv('./results/' + name + '.csv', index=False)
        print("Saved: ./results/" + name + ".csv")
    
    with open('./results/history.json', 'w') as f:
        # 移除不可序列化的 config 中的类对象
        history_save = {}
        for k, v in history.items():
            history_save[k] = {
                'train_loss': v['train_loss'],
                'val_loss': v['val_loss'],
                'val_ppl': v['val_ppl'],
                'config': v. get('config', {})
            }
        json. dump(history_save, f, indent=2)
    print("Saved: ./results/history.json")
    
    # 保存模型配置
    with open('./results/model_configs. json', 'w') as f:
        json.dump(trainer.MODEL_CONFIGS, f, indent=2)
    print("Saved: ./results/model_configs.json")
    
    # 绘图
    plot_results(results, history)
    
    # 打印汇总
    print("\n" + "="*80)
    print("实验完成!  结果汇总")
    print("="*80)
    
    for name, df in results.items():
        print("\n[" + name + "]")
        print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("模型配置汇总")
    print("="*80)
    for name, cfg in trainer.MODEL_CONFIGS.items():
        print("  {}: hidden_size={}, num_layers={}, num_heads={}".format(
            name, cfg['hidden_size'], cfg['num_layers'], cfg['num_heads']))
    
    print("\n" + "="*80)
    print("Profiler 文件说明")
    print("="*80)
    print("1. Chrome Trace (可用 chrome://tracing 打开):")
    print("   ./profiler_results/*_trace.json")
    print("")
    print("2. TensorBoard 日志 (可用 tensorboard --logdir=./profiler_logs 查看):")
    print("   ./profiler_logs/")
    print("")
    print("3. 汇总表格:")
    print("   ./profiler_results/profile_summary.csv")
    print("   ./profiler_results/*_table.txt")
    print("")
    print("4. Stacks 文件 (可生成火焰图):")
    print("   ./profiler_results/*_stacks.txt")
    print("   使用:  https://www.speedscope.app/ 打开")
    
    return results, history


if __name__ == "__main__":
    results, history = main()
