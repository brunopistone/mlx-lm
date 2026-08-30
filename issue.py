import mlx.core as mx

from mlx_lm.models.cache import KVCache, RotatingKVCache

cache = RotatingKVCache(max_size=128)
for step in range(1, 40_001):
    row = mx.full((1, 1, 1, 512), float(step % 7), mx.bfloat16)
    mx.eval(row)
    keys, _ = cache.update_and_fetch(row, row)
    mx.eval(keys)
    cache.values = mx.zeros_like(cache.keys)
    if step % 10_000 == 0:
        print(step, mx.get_active_memory() // 1024, "KB active")
