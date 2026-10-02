"""
Two-view blocker for ONE country's pool at a time:
  - name-only index: catches strong name matches despite noisy addresses
  - blend (name+address) index: disambiguates common names via address
Built per-country (not for all countries at once) so peak memory is bounded
by the single largest country's pool, not the sum of all countries.
"""
import sys
sys.path.insert(0, "/home/claude/ber/src")
from token_blocking import TokenIndex, tokenize_for_blocking


class SingleCountryBlocker:
    def __init__(self, ids, names, addrs):
        """ids/names/addrs: parallel lists for one country's pool only."""
        self.name_idx = TokenIndex(ids, lambda: (n.split() if n else [] for n in names))
        self.blend_idx = TokenIndex(ids, lambda: (tokenize_for_blocking(n, a)
                                                    for n, a in zip(names, addrs)))
        self.pool_name = dict(zip(ids, names))
        self.pool_addr = dict(zip(ids, addrs))

    def query_one(self, norm_name, norm_address, k=15):
        merged = {}
        for pid, raw, norm in self.name_idx.query(norm_name.split() if norm_name else [], k=k):
            merged.setdefault(pid, [0.0, 0.0])
            merged[pid][0] = max(merged[pid][0], norm)
        toks = tokenize_for_blocking(norm_name, norm_address)
        for pid, raw, norm in self.blend_idx.query(toks, k=k):
            merged.setdefault(pid, [0.0, 0.0])
            merged[pid][1] = max(merged[pid][1], norm)
        return [(pid, v[0], v[1]) for pid, v in merged.items()]
