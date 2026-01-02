import generators.search as search
import numpy as np
import random


class Generator(search.Generator):
    def reset(self, **kwargs):
        super().reset(**kwargs)

        self._width = self._env._problem._width
        self._height = self._env._problem._height
        self._target = self._env._problem._target
        self._min_room_size = kwargs.get('min_room_size', 3)
        self._max_room_size = kwargs.get('max_room_size', 15)
        self._max_agent_moves = kwargs.get('max_agent_moves', 200)

        self._generate()


    def update(self):
        self._generate()


    def _check_collisions(self, rooms, new_room):
        for room in rooms:
            if new_room[2] < room[0] - 1 or room[2] + 1 < new_room[0]:
                continue
            if new_room[3] < room[1] - 1 or room[3] + 1 < new_room[1]:
                continue
            return True
        return False


    def _place_room(self, agent_x, agent_y, chromosome, rooms):
        for _ in range(10):
            room_width = random.randint(self._min_room_size // 2, self._max_room_size // 2)
            room_height = random.randint(self._min_room_size // 2, self._max_room_size // 2)
            if 0 < agent_x - room_width and agent_x + room_width < self._width - 1 and 0 < agent_y - room_height and agent_y + room_height < self._height - 1:
                new_room = (agent_x - room_width, agent_y - room_height, agent_x + room_width, agent_y + room_height)
                if not self._check_collisions(rooms, new_room):
                    for y in range(agent_y - room_height, agent_y + room_height + 1):
                        for x in range(agent_x - room_width, agent_x + room_width + 1):
                            chromosome._content[y][x] = np.int64(1)
                    rooms.append(new_room)
                    return


    def _generate(self):
        chromosome = search.Chromosome(self._random)
        chromosome.random(self._env)
        chromosome._control['path'] = np.int64(self._target)
        for y in range(self._height):
            for x in range(self._width):
                chromosome._content[y][x] = np.int64(0)

        rooms = []
        agent_x, agent_y = random.randint(2, self._width - 3), random.randint(2, self._height - 3)
        self._place_room(agent_x, agent_y, chromosome, rooms)

        last_dim = ''
        last_dir = 0

        for _ in range(self._max_agent_moves):
            if random.random() < 0.2:
                self._place_room(agent_x, agent_y, chromosome, rooms)
            else:
                d = 1 if random.random() < 0.5 else -1
                v = random.randint(1, 12)
                if 1 < agent_x + d * v < self._width - 2 and random.random() < 0.5:
                    if last_dim == 'x' and last_dir == -d:
                        self._place_room(agent_x, agent_y, chromosome, rooms)
                    for _ in range(v):
                        agent_x += d
                        chromosome._content[agent_y][agent_x] = np.int64(1)
                    last_dim = 'x'
                if 1 < agent_y + d * v < self._height - 2 and random.random() < 0.5:
                    if last_dim == 'y' and last_dir == -d:
                        self._place_room(agent_x, agent_y, chromosome, rooms)
                    for _ in range(v):
                        agent_y += d
                        chromosome._content[agent_y][agent_x] = np.int64(1)
                    last_dim = 'y'
                last_dir = d

        self._place_room(agent_x, agent_y, chromosome, rooms)
        self._chromosomes = [chromosome]
        search.evaluateChromosomes(self._env, self._chromosomes)
