import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

def dfs_iterative(graph, start_vertex):
    visited = set()
    stack = [start_vertex]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            print(vertex, end=' ')
            visited.add(vertex)
            stack.extend(reversed(graph[vertex]))


def bfs_iterative(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            print(vertex, end=" ")
            visited.add(vertex)
            queue.extend(set(graph[vertex]) - visited)

    return visited


def dijkstra_algorithm(graph, start_node):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    visited = set()

    while len(visited) < len(graph.nodes):
        current_node = min(
            (node for node in graph.nodes if node not in visited), key=lambda x: distances[x])
        visited.add(current_node)

        for neighbor, weight in graph[current_node].items():
            potential_distance = distances[current_node] + weight['weight']
            if potential_distance < distances[neighbor]:
                distances[neighbor] = potential_distance

    return distances

# Створення графа "Штучна нейронна мережа"
G = nx.DiGraph()

# Додавання вершин та ребер з вагами
layers = {'Input': 3, 'Hidden': 4, 'Output': 2}

for layer, num_neurons in layers.items():
    color = 'red' if layer == 'Input' else 'blue' if layer == 'Hidden' else 'green'
    G.add_nodes_from([(f'{layer}{i+1}', {'layer': layer, 'color': color})
                     for i in range(num_neurons)])

G.add_edge('Input1', 'Hidden1', weight=1)
G.add_edge('Input1', 'Hidden2', weight=0)
G.add_edge('Input1', 'Hidden3', weight=1)
G.add_edge('Input1', 'Hidden4', weight=1)
G.add_edge('Input2', 'Hidden1', weight=0)
G.add_edge('Input2', 'Hidden2', weight=1)
G.add_edge('Input2', 'Hidden3', weight=0)
G.add_edge('Input2', 'Hidden4', weight=1)
G.add_edge('Input3', 'Hidden1', weight=1)
G.add_edge('Input3', 'Hidden2', weight=0)
G.add_edge('Input3', 'Hidden3', weight=1)
G.add_edge('Input3', 'Hidden4', weight=0)
G.add_edge('Hidden1', 'Output1', weight=1)
G.add_edge('Hidden1', 'Output2', weight=0)
G.add_edge('Hidden2', 'Output1', weight=0)
G.add_edge('Hidden2', 'Output2', weight=1)
G.add_edge('Hidden3', 'Output1', weight=1)
G.add_edge('Hidden3', 'Output2', weight=0)
G.add_edge('Hidden4', 'Output1', weight=0)
G.add_edge('Hidden4', 'Output2', weight=1)

labels = {node: node for node in G.nodes()}
colors = [data['color'] for _, data in G.nodes(data=True)]

# Визуалізація графа
pos = {'Input1': (1, 2.5), 'Input2': (1, 1.5), 'Input3': (1, 0.5),
       'Hidden1': (2, 3), 'Hidden2': (2, 2), 'Hidden3': (2, 1), 'Hidden4': (2, 0),
       'Output1': (3, 2), 'Output2': (3, 1)}

nx.draw_networkx(G, pos, with_labels=True, labels=labels, node_color=colors,
                 node_size=800, font_size=8, font_color='black', font_weight='bold', arrowsize=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# Інформація про граф
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Вивід кількості вершин і ребер
print(f"Кількість вершин: {num_nodes}")
print(f"Кількість ребер: {num_edges}")

# Ступінь вершин
degree_sequence = [G.degree(node) for node in G.nodes]
average_degree = sum(degree_sequence) / num_nodes

# Вивід ступенів вершин
print(f"Ступінь вершин: {degree_sequence}")
print(f"Середній ступінь вершин: {average_degree}")

# Використовуємо DFS для знаходження шляхів
dfs_paths = list(nx.dfs_edges(G, source='Input1'))

print("Алгоритм DFS:")
for path in dfs_paths:
    print(path)

# Використовуємо BFS для знаходження шляхів
bfs_paths = list(nx.bfs_edges(G, source='Input1'))

print("Алгоритм BFS:")
for path in bfs_paths:
    print(path)

# Виведення найкоротших шляхів між всіма вершинами
shortest_paths = nx.single_source_dijkstra_path(G, source='Input2')

print(f"Найкоротші шляхи між всіма вершинами: {shortest_paths}")
