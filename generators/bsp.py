import generators.search as search
import numpy as np
import random


class BspNode():
    def __init__(self, parent, depth, x1, y1, x2, y2, min_room_size):
        self._parent = parent
        self._depth = depth
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._min_room_size = min_room_size

        self._rooms = []

        self._split()


    def _check_size(self, dim):
        if dim == 'x':
            return self._x2 - self._x1 >= 2 * (self._min_room_size + 2)
        elif dim == 'y':
            return self._y2 - self._y1 >= 2 * (self._min_room_size + 2)
        else:
            return False


    def _init_room(self):
        x1 = random.randint(self._x1 + 1, self._x2 - self._min_room_size)
        y1 = random.randint(self._y1 + 1, self._y2 - self._min_room_size)
        x2 = random.randint(x1 + self._min_room_size - 1, self._x2 - 1)
        y2 = random.randint(y1 + self._min_room_size - 1, self._y2 - 1)

        self._rooms = [(x1, y1, x2, y2)]


    def _dist(self, l, r, dim):
        if dim == 'x':
            intersection_start = max(l[1], r[1])
            intersection_end = min(l[3], r[3])
            if intersection_start <= intersection_end:
                return r[0] - l[2]
            elif r[3] < l[1]:
                return r[0] - l[2] + l[1] - r[3]
            else:
                return r[0] - l[2] + r[1] - l[3]
        else:
            intersection_start = max(l[0], r[0])
            intersection_end = min(l[2], r[2])
            if intersection_start <= intersection_end:
                return r[1] - l[3]
            elif r[2] < l[0]:
                return r[1] - l[3] + l[0] - r[2]
            else:
                return r[1] - l[3] + r[0] - l[2]


    def _split(self):
        if self._depth == 0 or (not self._check_size('x') and not self._check_size('y')):
            self._init_room()
            return

        if not self._check_size('x'):
            split_dim = 'y'
        elif not self._check_size('y'):
            split_dim = 'x'
        else:
            split_dim = random.choice('xy')

        if split_dim == 'x':
            split_val = random.randint(self._x1 + self._min_room_size + 1, self._x2 - self._min_room_size - 2)
            self._left = BspNode(self, self._depth - 1, self._x1, self._y1, split_val, self._y2, self._min_room_size)
            self._right = BspNode(self, self._depth - 1, split_val + 1, self._y1, self._x2, self._y2, self._min_room_size)
        else:
            split_val = random.randint(self._y1 + self._min_room_size + 1, self._y2 - self._min_room_size - 2)
            self._left = BspNode(self, self._depth - 1, self._x1, self._y1, self._x2, split_val, self._min_room_size)
            self._right = BspNode(self, self._depth - 1, self._x1, split_val + 1, self._x2, self._y2, self._min_room_size)

        self._rooms = self._left._rooms + self._right._rooms

        # connect siblings
        min_dist = 10000000000000000000
        closest_rooms = []
        for room_left in self._left._rooms:
            for room_right in self._right._rooms:
                d = self._dist(room_left, room_right, split_dim)
                if d < min_dist:
                    min_dist = d
                    closest_rooms = []
                if d == min_dist:
                    closest_rooms.append((room_left, room_right))

        room_left, room_right = random.choice(closest_rooms)
        min_x_left, min_y_left, max_x_left, max_y_left = room_left
        min_x_right, min_y_right, max_x_right, max_y_right = room_right

        if split_dim == 'x':
            intersection_y_min = max(min_y_left, min_y_right)
            intersection_y_max = min(max_y_left, max_y_right)

            if intersection_y_min <= intersection_y_max:
                y = random.randint(intersection_y_min, intersection_y_max)
                self._rooms.append((max_x_left, y, min_x_right, y))
            elif max_y_right < min_y_left:
                self._rooms.append((max_x_left, min_y_left, min_x_right, min_y_left))
                self._rooms.append((min_x_right, max_y_right, min_x_right, min_y_left))
            else:
                self._rooms.append((max_x_left, max_y_left, min_x_right, max_y_left))
                self._rooms.append((min_x_right, max_y_left, min_x_right, min_y_right))
        else:
            intersection_x_min = max(min_x_left, min_x_right)
            intersection_x_max = min(max_x_left, max_x_right)

            if intersection_x_min <= intersection_x_max:
                x = random.randint(intersection_x_min, intersection_x_max)
                self._rooms.append((x, max_y_left, x, min_y_right))
            elif max_x_right < min_x_left:
                self._rooms.append((min_x_left, max_y_left, min_x_left, min_y_right))
                self._rooms.append((max_x_right, min_y_right, min_x_left, min_y_right))
            else:
                self._rooms.append((max_x_left, max_y_left, max_x_left, min_y_right))
                self._rooms.append((max_x_left, min_y_right, min_x_right, min_y_right))



class Generator(search.Generator):
    def reset(self, **kwargs):
        super().reset(**kwargs)

        self._width = self._env._problem._width
        self._height = self._env._problem._height
        self._target = self._env._problem._target
        self._depth = kwargs.get('depth', 4)
        self._min_room_size = kwargs.get('min_room_size', 3)

        self._generate()


    def update(self):
        self._generate()


    def _generate(self):
        root = BspNode(None, self._depth, 0, 0, self._width - 1, self._height - 1, self._min_room_size)

        chromosome = search.Chromosome(self._random)
        chromosome.random(self._env)
        chromosome._control['path'] = np.int64(self._target)
        for y in range(self._height):
            for x in range(self._width):
                chromosome._content[y][x] = np.int64(0)

        for x1, y1, x2, y2 in root._rooms:
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    chromosome._content[y][x] = np.int64(1)

        self._chromosomes = [chromosome]
        search.evaluateChromosomes(self._env, self._chromosomes)
