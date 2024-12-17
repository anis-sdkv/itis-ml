import networkx as nx
import random

import numpy as np
import pygame


class ClusteredGraph:
    def __init__(self):
        self.graph = self.generate_random_weighted_graph(10, np.random.randint(30, 50) / 100)
        spanning_tree = nx.minimum_spanning_tree(self.graph, algorithm="kruskal")
        clusters, clustered_forest = self.get_clusters(3, spanning_tree)

        colors = [np.random.uniform(0, 1, size=3) for i in range(len(clusters))]
        self.node_color_map = {}
        for i, cluster in enumerate(clusters):
            for node in cluster:
                self.node_color_map[node] = colors[i]

        self.edge_labels = nx.get_edge_attributes(self.graph, 'weight')

    @staticmethod
    def generate_random_weighted_graph(n_vertices, edge_probability):
        # graph = nx.erdos_renyi_graph(n_vertices, edge_probability)
        total_edges = n_vertices * (n_vertices - 1) // 2
        n_edges = int(total_edges * edge_probability)
        graph = nx.gnm_random_graph(n_vertices, n_edges)
        for u, v in graph.edges():
            graph[u][v]["weight"] = random.randint(1, 200)
        return graph

    @staticmethod
    def get_clusters(n_clusters, spanning_tree):
        spanning_tree_copy = spanning_tree.copy()
        sorted_edges = sorted(spanning_tree.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
        edges_to_remove = sorted_edges[:n_clusters - 1]
        for u, v, _ in edges_to_remove:
            spanning_tree_copy.remove_edge(u, v)

        return list(nx.connected_components(spanning_tree_copy)), spanning_tree_copy


class GraphViewModel:
    NODE_RADIUS = 10

    def __init__(self, width, height):
        self.graph_data = ClusteredGraph()
        self.graph_node_positions = {node: np.array(((x + 1) / 2 * width, (y + 1) / 2 * height))
                                     for node, (x, y) in nx.spring_layout(self.graph_data.graph, seed=42).items()}

    # Сила отталкивания (между всеми узлами)
    @staticmethod
    def repulsion_force(node1, node2, positions, k):
        pos1 = positions[node1]
        pos2 = positions[node2]
        distance = np.linalg.norm(pos1 - pos2)
        if distance < 10:
                distance = 10
        force_direction = (pos1 - pos2) / distance  # нормализованный вектор силы отталкивания
        magnitude = k / (distance ** 2)  # сила отталкивания пропорциональна квадрату расстояния
        return force_direction * magnitude

    # Сила притяжения (для рёбер)
    @staticmethod
    def attraction_force(node1, node2, positions, k):
        pos1 = positions[node1]
        pos2 = positions[node2]
        distance = np.linalg.norm(pos1 - pos2)
        force_direction = (pos2 - pos1) / distance  # нормализованный вектор притяжения
        if distance < 10:
            distance = 10
        magnitude = distance ** 2 / k  # сила притяжения пропорциональна квадрату расстояния
        return force_direction * magnitude

    def update_positions(self, link_distance):
        for node in self.graph_data.graph.nodes():
            total_repulsion = np.array([0.0, 0.0])
            for other_node in self.graph_data.graph.nodes():
                if node != other_node:
                    total_repulsion += GraphViewModel.repulsion_force(node, other_node, self.graph_node_positions,
                                                                      link_distance * 5)

            total_attraction = np.array([0.0, 0.0])
            for neighbor in self.graph_data.graph.neighbors(node):
                total_attraction += GraphViewModel.attraction_force(node, neighbor, self.graph_node_positions,
                                                                    link_distance)

            self.graph_node_positions[node] += total_repulsion + total_attraction

    def draw_graph(self, screen):
        pos = self.graph_node_positions
        graph = self.graph_data.graph
        node_colors = self.graph_data.node_color_map
        weights = self.graph_data.edge_labels

        for u, v in graph.edges():
            pygame.draw.line(screen, (100, 100, 100), pos[u], pos[v], 2)

        for node, (x, y) in pos.items():
            pygame.draw.circle(screen, node_colors[node] * 255, (x, y), GraphViewModel.NODE_RADIUS)

        for u, v in graph.edges():
            weight = weights[(u, v)]
            font = pygame.font.SysFont(None, 18)
            label = font.render(str(weight), True, (0, 0, 0))

            x1, y1 = pos[u]
            x2, y2 = pos[v]
            mid_x = (x1 + x2 - label.get_width()) / 2
            mid_y = (y1 + y2 - label.get_height()) / 2
            box_offset = 4
            label_rect = (mid_x - box_offset, mid_y - box_offset, label.get_width() + box_offset * 2,
                          label.get_height() + box_offset * 2)
            pygame.draw.rect(screen, "white", label_rect)
            pygame.draw.rect(screen, "black", label_rect, 2)

            screen.blit(label, (mid_x, mid_y))

    def try_get_node_by_pos(self, pos):
        for node, (x, y) in self.graph_node_positions.items():
            distance = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
            if distance <= GraphViewModel.NODE_RADIUS * 2:
                return node
        return None


class App:
    BACKGROUND_COLOR = (255, 255, 255)

    def __init__(self, width=800, height=600):
        pygame.init()
        pygame.display.set_caption("GraphVisualization")
        self.width = width
        self.height = height
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)

        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 60

        self.key_f_down = False

        self.graph_view_model = GraphViewModel(width, height)
        self.captured_node = None

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.captured_node = self.graph_view_model.try_get_node_by_pos(pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONUP:
                self.captured_node = None
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)

    def update(self):
        self.graph_view_model.update_positions(30000)
        self._handle_mouse_input()
        self._handle_key_actions()

    def draw(self):
        self.screen.fill(App.BACKGROUND_COLOR)
        self.graph_view_model.draw_graph(self.screen)

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(self.fps)

        self.quit()

    def quit(self):
        self.running = False
        pygame.quit()

    def _handle_mouse_input(self):
        if self.captured_node is not None:
            self.graph_view_model.graph_node_positions[self.captured_node] = np.array(pygame.mouse.get_pos(),
                                                                                      dtype=np.float64)

    def _handle_key_actions(self):
        if self.key_f_down:
            self.key_f_down = False
            self._toggle_fullscreen()

    def _handle_keydown(self, key):
        if key == pygame.K_f:
            self.key_f_down = True
        elif key == pygame.K_q:
            self.key_q_down = True

    def _toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen

        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)


app = App()
app.run()
# pos = nx.spring_layout(random_graph, seed=42)
# plt.figure(figsize=(8, 6))  # Размер рисунка
# nx.draw(
#     random_graph,
#     pos=pos,
#     with_labels=True,  # Показывать метки узлов
#     node_color="lightblue",  # Цвет узлов
#     edge_color="gray",  # Цвет рёбер
#     node_size=200,  # Размер узлов
#     font_size=10,  # Размер шрифта меток
# )
# nx.draw(
#     mst,
#     pos=pos,
#     with_labels=True,  # Показывать метки узлов
#     node_color=node_colors,  # Цвет узлов
#     edge_color="red",  # Цвет рёбер
#     node_size=200,  # Размер узлов
#     font_size=10,  # Размер шрифта меток
# )
#
# nx.draw(
#     clustered_forest,
#     pos=pos,
#     with_labels=True,  # Показывать метки узлов
#     node_color=node_colors,  # Цвет узлов
#     edge_color="blue",  # Цвет рёбер
#     node_size=200,  # Размер узлов
#     font_size=10,  # Размер шрифта меток
# )
#
# edge_labels = nx.get_edge_attributes(random_graph, 'weight')
# nx.draw_networkx_edge_labels(
#     random_graph, pos, edge_labels=edge_labels, font_size=10, font_color="black"
# )
#
# plt.title("Случайный граф (Erdos-Renyi Model)")
# plt.show()
