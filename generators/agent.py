import generators.search as search
import numpy as np
import random


class Generator(search.Generator):
    def reset(self, **kwargs):
        super().reset(**kwargs)

        self._width = self._env._problem._width
        self._height = self._env._problem._height
        self._target = self._env._problem._target
        self._depth = kwargs.get('depth', 4)
        self._min_room_size = kwargs.get('min_room_size', 3)
        self._max_agent_moves = kwargs.get('max_agent_moves', 200)

        self._generate()


    def update(self):
        self._generate()


    def _generate(self):
        chromosome = search.Chromosome(self._random)
        chromosome.random(self._env)
        chromosome._control['path'] = np.int64(self._target)
        for y in range(self._height):
            for x in range(self._width):
                chromosome._content[y][x] = np.int64(0)

        # TODO: add some constraints on movement/room placement
        agent_x, agent_y = random.randint(1, self._width - 2), random.randint(1, self._height - 2)
        chromosome._content[agent_y][agent_x] = np.int64(1)

        for _ in range(self._max_agent_moves):
            if random.random() < 0.17:
                # place room
                try:
                    room_width = random.randint(self._min_room_size // 2, min(agent_x - 1, self._width - agent_x - 2) // 4)
                    room_height = random.randint(self._min_room_size // 2, min(agent_y - 1, self._height - agent_y - 2) // 4)
                    for y in range(agent_y - room_height, agent_y + room_height + 1):
                        for x in range(agent_x - room_width, agent_x + room_width + 1):
                            chromosome._content[y][x] = np.int64(1)
                except:
                    pass
            else:
                # move
                d = 1 if random.random() < 0.5 else -1
                v = random.randint(1, 12)
                if 1 < agent_x + d * v < self._width - 1 and random.random() < 0.5:
                    for i in range(v):
                        agent_x += d
                        chromosome._content[agent_y][agent_x] = np.int64(1)
                if 1 < agent_y + d * v < self._height - 1 and random.random() < 0.5:
                    for i in range(v):
                        agent_y += d
                        chromosome._content[agent_y][agent_x] = np.int64(1)

        self._chromosomes = [chromosome]
        search.evaluateChromosomes(self._env, self._chromosomes)
