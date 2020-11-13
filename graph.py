'''
Generate a graphviz dot graph from the YAML states data.
'''

# pylint: disable=missing-docstring
# pylint: disable=line-too-long
# pylint: disable=len-as-condition
# pylint: disable=C0330

from typing import Any
from typing import List
import re
import sys
import yaml

def print_dot(data: Any, cluster_by_agents: List[str]):
    print('digraph G {')

    name_by_index = {}
    for node_index, node_data in data.items():
        name_by_index[node_index] = node_data['name']

    if len(cluster_by_agents) == 0:
        for node_index, node_name in name_by_index.items():
            print_node(node_index, node_name)
    else:
        patterns = [re.compile('(%s @ [^,;/]*)(?: [,;/]|$)' % name) for name in cluster_by_agents]
        for node_index, node_name in name_by_index.items():
            for pattern in patterns:
                if not pattern.search(node_name + ' ;'):
                    print(node_name)
                    assert False

        current_clusters = [] # type: List[str]
        for value in sorted([tuple(pattern.search(node_name).groups()[0] for pattern in patterns)
                             + (node_name, node_index)
                             for node_index, node_name in name_by_index.items()]):
            node_name, node_index = value[-2:]
            value_clusters = value[:-2]
            remaining_clusters = current_clusters + []

            while len(remaining_clusters) > 0 and len(value_clusters) > 0 and remaining_clusters[0] == value_clusters[0]:
                remaining_clusters = remaining_clusters[1:]
                value_clusters = value_clusters[1:]

            while len(remaining_clusters) > 0:
                remaining_clusters.pop()
                current_clusters.pop()
                print('}')

            for cluster_name in value_clusters:
                current_clusters.append(cluster_name)
                print('subgraph "cluster_%s" {' % ' , '.join(current_clusters))
                print('label = "%s";' % cluster_name)

            print_node(node_index, node_name)

        while len(current_clusters) > 0:
            current_clusters.pop()
            print('}')

    print_edges(data)

def print_node(node_index: int, node_name: str) -> None:
    print('%s [ label="%s" ];' % (node_index, node_name.replace(' , ', '\n').replace(' ; ', '\n')))
    if 'MISSING LOGIC' in node_name:
        node_name = 'MISSING LOGIC'
        shape = 'doubleoctagon'
    else:
        shape = 'box'
    print('%s [ label="%s", shape=%s ];' % (node_index, node_name.replace(' , ', '\n').replace(' ; ', '\n'), shape))

def print_edges(data: Any) -> None:
    for node_index, node_data in data.items():
        for incoming_index, transition in node_data['incoming'].items():
            transition_name = '%s\n-> %s ->\n%s' % (transition['source'], transition['message'], transition['target'])
            print('%s -> %s [ label="%s" ];' % (incoming_index, node_index, transition_name))

    print('}')

print_dot(yaml.safe_load(sys.stdin.read()), sys.argv[1:])
