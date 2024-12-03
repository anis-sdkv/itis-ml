import pygame

def draw_point(screen, position, color=(255, 255, 255), radius=5):
    """
    Рисует точку на экране pygame.

    Parameters:
    - screen: объект pygame.Surface, на котором будет рисоваться точка.
    - position: tuple, координаты точки (x, y).
    - color: tuple, цвет точки в формате RGB. По умолчанию белый (255, 255, 255).
    - radius: int, радиус точки. По умолчанию 5.
    """
    pygame.draw.circle(screen, color, position, radius)
