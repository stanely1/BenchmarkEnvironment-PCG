import generators.search as search
import numpy as np
import random


class BspNode():
    # TODO: parametrize this size
    MIN_SIZE = 5

    def __init__(self, parent, depth, x1, y1, x2, y2):
        self._parent = parent
        self._depth = depth
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2

        self._rooms = []

        self._split()


    def _check_size(self, dim):
        if dim == 'x':
            return self._x2 - self._x1 >= 2 * BspNode.MIN_SIZE
        elif dim == 'y':
            return self._y2 - self._y1 >= 2 * BspNode.MIN_SIZE
        else:
            return False


    def _init_room(self):
        # TODO: parametrize room size
        x1 = random.randint(self._x1 + 1, self._x2 - 3)
        y1 = random.randint(self._y1 + 1, self._y2 - 3)
        x2 = random.randint(x1 + 2, self._x2 - 1)
        y2 = random.randint(y1 + 2, self._y2 - 1)

        self._rooms = [(x1, y1, x2, y2)]


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
            split_val = random.randint(self._x1 + BspNode.MIN_SIZE - 1, self._x2 - BspNode.MIN_SIZE)
            self._left = BspNode(self, self._depth - 1, self._x1, self._y1, split_val, self._y2)
            self._right = BspNode(self, self._depth - 1, split_val + 1, self._y1, self._x2, self._y2)
        else:
            split_val = random.randint(self._y1 + BspNode.MIN_SIZE - 1, self._y2 - BspNode.MIN_SIZE)
            self._left = BspNode(self, self._depth - 1, self._x1, self._y1, self._x2, split_val)
            self._right = BspNode(self, self._depth - 1, self._x1, split_val + 1, self._x2, self._y2)

        self._rooms = self._left._rooms + self._right._rooms

        # connect siblings
        if split_dim == 'x':
            idx_left, max_x_left = max(enumerate(self._left._rooms), key=lambda r: r[1][2])
            max_x_left = max_x_left[2]
            min_y_left, max_y_left = self._left._rooms[idx_left][1], self._left._rooms[idx_left][3]

            idx_right, min_x_right = min(enumerate(self._right._rooms), key=lambda r: r[1][0])
            min_x_right = min_x_right[0]
            min_y_right, max_y_right = self._right._rooms[idx_right][1], self._right._rooms[idx_right][3]

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
            idx_left, max_y_left = max(enumerate(self._left._rooms), key=lambda r: r[1][3])
            max_y_left = max_y_left[3]
            min_x_left, max_x_left = self._left._rooms[idx_left][0], self._left._rooms[idx_left][2]

            idx_right, min_y_right = min(enumerate(self._right._rooms), key=lambda r: r[1][1])
            min_y_right = min_y_right[1]
            min_x_right, max_x_right = self._right._rooms[idx_right][0], self._right._rooms[idx_right][2]

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

        self._generate()


    def update(self):
        self._generate()


    def _generate(self):
        root = BspNode(None, self._depth, 0, 0, self._width - 1, self._height - 1)

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
