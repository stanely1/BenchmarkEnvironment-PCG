import generators.search as search

class Generator(search.Generator):
    def reset(self, **kwargs):
        super().reset(**kwargs)

        self._problem = kwargs.get('problem')
        self._lambda_size = kwargs.get('lambda_size', 100)
        self._mut_rate = kwargs.get('mut_rate', 0.05)

        chromosomes = []
        for _ in range(self._lambda_size):
            chromosomes.append(search.Chromosome(self._random))
            chromosomes[-1].random(self._env)
        search.evaluateChromosomes(self._env, chromosomes)

        self._map_elites = {}
        self._update_map_elites(chromosomes)


    def update(self):
        # TODO: GA?
        curr_chromosomes = list(self._map_elites.values())
        new_chromosomes = []
        for i in range(self._lambda_size):
            index = self._random.integers(len(curr_chromosomes))
            new_chromosomes.append(curr_chromosomes[index][0].mutation(self._env, self._mut_rate))
        search.evaluateChromosomes(self._env, new_chromosomes)
        self._update_map_elites(new_chromosomes)

        quality_passing_cells = len(list(filter(lambda c: c.quality() == 1, self._chromosomes)))
        print(f'cells that pass quality test: {quality_passing_cells}/{len(self._chromosomes)} ({100.0*quality_passing_cells/len(self._chromosomes):.2f}%)')


    def _update_map_elites(self, new_chromosomes):
        for chromosome in new_chromosomes:
            cell_id = self._get_cell_id(chromosome)
            fitness = self._fitness_fn(chromosome)
            if (cell_id not in self._map_elites) or fitness > self._map_elites[cell_id][1]:
                self._map_elites[cell_id] = (chromosome, fitness)

        self._chromosomes = list(map(lambda c: c[0],
                                     sorted(self._map_elites.values(), key=lambda c: c[1], reverse=True)))


    def _get_cell_id(self, chromosome):
        info = chromosome._info

        # TODO: use problem specific features
        if self._problem.startswith('binary-'):
            # return (info['path'], info['regions'])
            # number of empty tiles
            return sum(info['flat'])

        if self._problem.startswith('sokoban-'):
            return (info['crates'], info['targets'])

        if self._problem.startswith('ddave-'):
            player_locations = info['player_locations']
            player_location = player_locations[0] if player_locations else -1

            exit_locations = info['exit_locations']
            exit_location = exit_locations[0] if exit_locations else -1

            return (player_location, exit_location)

        if self._problem.startswith('building-'):
            return (info['blocks'])#, tuple(info['heights']))

        raise RuntimeError(f'Unknown problem: {self._problem}')
