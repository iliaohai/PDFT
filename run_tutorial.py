"""Run the original PDFT functions without the unavailable project package/assets."""
import argparse
import json
from pathlib import Path
import random
import sys
import types

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aps', type=int, default=9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='tutorial_result.json')
    args = parser.parse_args()
    if args.aps < 1:
        parser.error('--aps must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Alias the original project's imports; leave algorithm files unchanged.
    import util
    for name in ('deploy', 'deploy.grad', 'deploy.grad.algorithm'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules['deploy.grad.algorithm.util'] = util
    sys.modules['deploy.grad.common_utils'] = util
    import common
    sys.modules['deploy.grad.algorithm.common'] = common
    import proposed_common as pdf

    circles, bounds, radii = pdf.init_circles(args.aps)
    initial = [tuple(c) for c in circles]
    pd = pdf.optimize_circles(circles, display=False)
    pd_snapshot = [tuple(c) for c in pd]
    final = pdf.optimize_circles_bfgs(pd, bounds, radii, display=False)
    coords = np.asarray([(x, y) for x, y, _ in final]).ravel()
    rss, coverage = pdf.total_overlap_vectorized(coords, display=False, data=True)
    result = dict(seed=args.seed, aps=args.aps, initial=initial,
                  predeployment=pd_snapshot, final=final, rss=rss, coverage=coverage)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'rss': rss, 'coverage': coverage}, indent=2))
    print(f'Saved {path.resolve()}')


if __name__ == '__main__':
    main()
