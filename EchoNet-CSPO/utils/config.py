"""YAML 配置加载: dict -> 可点取属性的命名空间, 并保留原始 dict 以便保存。"""
import os
import yaml
from types import SimpleNamespace


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_ns(x) for x in d]
    return d


def _resolve_paths(d, base_dir):
    """递归地把 dict 中所有相对路径字符串解析为绝对路径。

    base_dir 为配置文件所在目录（即项目的 configs/ 目录）。
    只处理以 '.' 或 '..' 开头的字符串，其余值原样保留。
    这样无论从哪个工作目录启动脚本，路径都能正确解析。
    """
    if isinstance(d, dict):
        return {k: _resolve_paths(v, base_dir) for k, v in d.items()}
    if isinstance(d, list):
        return [_resolve_paths(x, base_dir) for x in d]
    if isinstance(d, str) and (d.startswith('./') or d.startswith('../')
                               or d.startswith('.\\') or d.startswith('..\\')):
        return os.path.normpath(os.path.join(base_dir, d))
    return d


def load_config(path):
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)          # configs/ 目录
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    raw = _resolve_paths(raw, base_dir)       # 相对路径 → 绝对路径
    return _to_ns(raw), raw
