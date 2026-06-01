#!/usr/bin/env python3
"""Find CNN padding/pool yielding flatten size 13080 (Sun et al. Table 1)."""
import torch
import torch.nn as nn

TARGET = 13080


def flat_size(p1, p2, pk):
    m = nn.Sequential(
        nn.Conv2d(1, 16, (10, 10), padding=p1),
        nn.MaxPool2d(pk),
        nn.Conv2d(16, 12, (15, 15), padding=p2),
    )
    x = torch.zeros(1, 1, 251, 15)
    with torch.no_grad():
        y = m(x)
    return tuple(y.shape), y.numel()


found = []
for p1 in range(0, 12):
    for p2 in range(0, 12):
        for pk in [(10, 1), (5, 1), (2, 1), (3, 1), (10, 2)]:
            try:
                sh, n = flat_size(p1, p2, pk)
                if n == TARGET:
                    found.append((p1, p2, pk, sh))
            except Exception:
                pass
print("matches:", len(found))
for f in found[:30]:
    print(f)
