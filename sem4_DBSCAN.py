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


WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)

n_samples = 300
dbscan_eps = 25
dbscan_min_samples = 5
dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)

points_to_draw = []
labels = []
label_colors = {}

# Инициализация pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DBSCAN Clustering Visualization")
clock = pygame.time.Clock()


#
# X, _ = make_blobs(n_samples=n_samples, centers=5, cluster_std=20, center_box=(100, 700), random_state=42)
# X = np.clip(X, 0, min(WIDTH, HEIGHT))
# points_to_draw.append(list(X))


# def apply_dbscan():
#     global labels
#     if len(points_to_draw) == 0:
#         return
#     labels = list(dbscan.fit_predict(np.array(points_to_draw)))
#     for label in set(labels):
#         if label == -1:
#             label_colors[label] = (255, 255, 255)
#         else:
#             label_colors[label] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
#

def draw_points():
    screen.fill(BACKGROUND_COLOR)
    for point in points_to_draw:
        pygame.draw.circle(screen, "black", (int(point.x), int(point.y)), 5)

    # for point, label in zip(points_to_draw, labels):
    #     pygame.draw.circle(screen, label_colors[label], (int(point[0]), int(point[1])), 5)


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
            position = event.pos
            points_to_draw.append(Point(*position))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                print(event)

    # Отображение точек
    draw_points()

    # Обновление экрана
    pygame.display.flip()
    clock.tick(30)

# Завершение работы
pygame.quit()
