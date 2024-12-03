from cProfile import label

import pygame
import random
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, another_point):
        return np.sqrt((self.x - another_point.x) ** 2 + (self.y - another_point.y) ** 2)

    def generate_points_in_neighborhood(self, max_distance, num_points):
        points = []
        for _ in range(num_points):
            angle = np.random.uniform(0, 2 * np.pi)  # случайный угол от 0 до 2*pi
            distance = np.random.uniform(0, max_distance)  # случайное расстояние от 0 до max_distance

            dx = distance * np.cos(angle)
            dy = distance * np.sin(angle)

            new_point = (self.x + dx, self.y + dy)
            points.append(Point(*new_point))
        return points


def generate_labels_points(n_clusters):
    colors = [np.random.randint(0, 256, size=3) for i in range(n_clusters - 1)]
    colors.append(np.array([255, 0, 0]))  # red
    return colors


WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)

n_samples = 300
dbscan_eps = 25
dbscan_min_samples = 5
dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)

points_to_draw = []
labels = None
label_colors = None

# Инициализация pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DBSCAN Clustering Visualization")
pygame.display.set_mode((0,0), pygame.RESIZABLE)
clock = pygame.time.Clock()


def draw_points():
    screen.fill(BACKGROUND_COLOR)
    color = "black"
    for i in range(len(points_to_draw)):
        if not labels is None:
            color = label_colors[labels[i]]
        point = points_to_draw[i]
        pygame.draw.circle(screen, color, (int(point.x), int(point.y)), 5)


# Главный цикл
mouse_down = False

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
        elif event.type == pygame.MOUSEMOTION and mouse_down:
            position = Point(*event.pos)
            last_position = points_to_draw[-1] if len(points_to_draw) > 0 else None
            if not last_position or last_position.distance_to(position) > 50:
                if not labels is None:
                    labels = None
                last_position = position
                points_to_draw.append(position)
                for gen_point in position.generate_points_in_neighborhood(20, 3):
                    points_to_draw.append(gen_point)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                labels = dbscan.fit_predict(np.array([[point.x, point.y] for point in points_to_draw]))
                label_colors = generate_labels_points(len(np.unique(labels)))

    # Отображение точек
    draw_points()

    # Обновление экрана
    pygame.display.flip()
    clock.tick(30)

# Завершение работы
pygame.quit()
