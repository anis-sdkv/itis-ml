import pygame
import numpy as np
from sklearn.cluster import DBSCAN


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


class App:
    BACKGROUND_COLOR = (255, 255, 255)
    DEFAULT_POINT_COLOR = "black"

    def __init__(self, dbscan_objects, width=800, height=600):
        pygame.init()
        pygame.display.set_caption("DBSCAN Clustering Visualization")
        self.width = width
        self.height = height
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)

        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 60

        self.mouse_down = False
        self.key_f_down = False
        self.key_1_down = False
        self.key_2_down = False
        self.key_3_down = False

        self.last_position = None
        self.labels = None
        self.label_colors = None
        self.color_by_type = None
        self.points_to_draw = []

        self.dbscan_objects = dbscan_objects

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)

    def update(self):
        self._handle_mouse_input()
        self._handle_key_actions()

    def draw(self):
        self.screen.fill(App.BACKGROUND_COLOR)
        color = App.DEFAULT_POINT_COLOR
        for i in range(len(self.points_to_draw)):
            if not self.labels is None:
                color = self.label_colors[self.labels[i]]
            if not self.color_by_type is None:
                color = self.color_by_type[i]
            point = self.points_to_draw[i]
            pygame.draw.circle(self.screen, color, (int(point.x), int(point.y)), 5)

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
        if self.mouse_down:
            cursor_position = Point(*pygame.mouse.get_pos())
            if not self.last_position or self.last_position.distance_to(cursor_position) > 50:
                self._reset()
                self.last_position = cursor_position
                self.points_to_draw.append(cursor_position)
                self.points_to_draw.extend(cursor_position.generate_points_in_neighborhood(30, 3))

    def _handle_key_actions(self):
        if self.key_f_down:
            self.key_f_down = False
            self._toggle_fullscreen()

        if self.key_1_down:
            self.key_1_down = False
            self._cluster_points(self.dbscan_objects[0])

        if self.key_2_down and len(self.dbscan_objects) > 1:
            self.key_2_down = False
            self._cluster_points(self.dbscan_objects[1])

        if self.key_3_down and len(self.dbscan_objects) > 1:
            self.key_3_down = False
            self._cluster_and_paint_by_type(self.dbscan_objects[1])

    def _handle_keydown(self, key):

        if key == pygame.K_f:
            self.key_f_down = True
        elif key == pygame.K_1:
            self.key_1_down = True
            self._reset()
        elif key == pygame.K_2:
            self.key_2_down = True
            self._reset()
        elif key == pygame.K_3:
            self.key_3_down = True
            self._reset()

    def _reset(self):
        self.labels = None
        self.label_colors = None
        self.color_by_type = None

    def _toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen

        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)

    def _cluster_points(self, dbscan):
        self.labels = dbscan.fit_predict(np.array([[point.x, point.y] for point in self.points_to_draw]))
        self.label_colors = self._generate_random_label_colors(len(np.unique(self.labels)))

    def _cluster_and_paint_by_type(self, dbscan):
        dbscan.fit(np.array([[point.x, point.y] for point in self.points_to_draw]))
        conversion_dict = {1: 'green', 0: 'yellow', -1: 'red'}
        self.color_by_type = [conversion_dict[x] for x in dbscan.point_types]

    @staticmethod
    def _generate_random_label_colors(n_clusters):
        colors = [
            np.array([np.random.randint(32, 128),
                      np.random.randint(32, 256),
                      np.random.randint(32, 256)])
            for i in range(n_clusters - 1)
        ]
        colors.append(np.array([255, 0, 0]))  # red
        return colors


if __name__ == "__main__":
    dbscan_eps = 25
    dbscan_min_samples = 5

    app = App([DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)])
    app.run()
