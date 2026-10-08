"""Read-only independent row/edge-chain oracles for staged software repairs."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import random
import sys
import unittest

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGED = HERE.parent / 'code_audit/staged/main_scripts'
sys.path.insert(0, str(ROOT / 'main_scripts'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ht = load('review_hierarchy_table', STAGED / 'region_analysis/hierarchy_table.py')
tracer = load('review_neuro_tracer', STAGED / 'neuro_tracer.py')


def graph_oracle(parent_ordered):
    """Enumerate maximal edge chains from degree changes; no tracer traversal."""
    parent = dict(parent_ordered)
    root = next(node for node, p in parent.items() if p == -1)
    children = {node: [] for node in parent}
    for node, p in parent_ordered:
        if p != -1:
            children[p].append(node)
    orders, paths = {}, {}
    # Derive each node's order and DFS ranking from its ancestor sequence.
    for node in parent:
        ancestors, current = [], node
        while parent[current] != -1:
            ancestors.append((parent[current], current))
            current = parent[current]
        orders[node] = sum(len(children[p]) > 1 for p, _ in ancestors)
        paths[node] = tuple(children[p].index(c) for p, c in reversed(ancestors))
    starts = [(root, [root])] + [(node, [p, node]) for node, p in parent_ordered
                                if p != -1 and len(children[p]) > 1]
    chains = []
    for start, chain in starts:
        while len(children[chain[-1]]) == 1:
            chain.append(children[chain[-1]][0])
        if len(chain) > 1 or not children[start]:
            chains.append((paths[start], chain))
    chains.sort(key=lambda pair: pair[0])
    return [chain for _, chain in chains], orders


class IndependentStagedReview(unittest.TestCase):
    def test_real_table_all_explicit_ancestors_and_no_arbitrary_descendants(self):
        self.tables_checked = 0
        for filename in ('CHARM_key_table_v2.csv', 'SARM_key_table_v2.csv'):
            df = pd.read_csv(ROOT / 'atlas' / filename)
            table = ht.HierarchyTable(df)
            expected = {}
            for _, row in df.iterrows():
                names = [row['Level_0']] + [row[f'Level_{level}_abbr'] for level in range(1, 7)]
                indices = [None] + [int(row[f'Level_{level}_index']) for level in range(1, 7)]
                for values in (names, indices):
                    for identity in set(values) - {None}:
                        finest_explicit = max(i for i, v in enumerate(values) if v == identity)
                        for level in range(7):
                            key = (str(identity), level)
                            result = names[level] if level <= finest_explicit else None
                            if key in expected:
                                previous = expected[key]
                                if previous is not None and result is not None:
                                    self.assertEqual(previous, result)
                                result = previous if previous is not None else result
                            expected[key] = result
            for (identity, level), wanted in expected.items():
                self.assertEqual(table.get_at_level(identity, level), wanted,
                                 (filename, identity, level))
            self.tables_checked += 1

    def test_prefix_retention_and_opt_in_loss(self):
        table = ht.DualHierarchyTable.load(ROOT/'atlas/CHARM_key_table_v2.csv',
                                           ROOT/'atlas/SARM_key_table_v2.csv')
        source = {'CL_Pi': 2., 'SL_Pi': 7., 'CR_Pi': 3., 'SR_Pi': 11.}
        result, unmapped = table.aggregate_to_level(source, 6)
        self.assertEqual(result, source)
        self.assertEqual(unmapped, [])
        stripped, unmapped = table.aggregate_to_level(source, 6, strip_prefixes=True)
        self.assertEqual(stripped, {'Pi': 23.})
        self.assertEqual(unmapped, [])

    def test_random_unsorted_and_adversarial_valid_trees(self):
        rng = random.Random(20261009)
        cases = [[(91, -1)]]
        # A long unary chain and deep alternating fork/continuation tree.
        cases.append([(1, -1)] + [(i, i-1) for i in range(2, 3002)])
        comb = [(1, -1)]
        continuation = 1
        for i in range(1200):
            a, b = 2*i+2, 2*i+3
            comb.extend([(a, continuation), (b, continuation)])
            continuation = a
        cases.append(comb)
        for _ in range(256):
            size = rng.randrange(2, 220)
            ids = rng.sample(range(2, 100000), size)
            tree = [(ids[0], -1)] + [(ids[i], ids[rng.randrange(i)])
                                         for i in range(1, size)]
            rng.shuffle(tree)
            cases.append(tree)
        for number, tree in enumerate(cases):
            expected_chains, expected_orders = graph_oracle(tree)
            neuron = tracer.neuro_tracer()
            neuron.nodes = {node: neuron.Node([node, 2, 0, 0, 0, 1, parent])
                            for node, parent in tree}
            neuron.root = next(n for n in neuron.nodes.values() if n.parent == -1)
            for node in neuron.nodes.values():
                if node.parent != -1:
                    neuron.nodes[node.parent].children.append(node)
            with contextlib.redirect_stdout(io.StringIO()):
                neuron._construct_branches()
            actual = [[node.id for node in branch] for branch in neuron.branches]
            self.assertEqual(actual, expected_chains, number)
            self.assertEqual({node.id: node.order for node in neuron.nodes.values()},
                             expected_orders, number)
            edges = [(a, b) for branch in actual for a, b in zip(branch, branch[1:])]
            self.assertCountEqual(edges, [(p, node) for node, p in tree if p != -1])
            self.assertEqual(len(edges), len(set(edges)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
