import generators.search as search

class Generator(search.Generator):
    def reset(self, **kwargs):
        super().reset(**kwargs)

        self._problem = kwargs.get('problem')
        self._pop_size = kwargs.get('pop_size', 100)
        self._tournment_size = kwargs.get('tournment_size', 7)
        self._cross_rate = kwargs.get('cross_rate', 0.5)
        self._mut_rate = kwargs.get('mut_rate', 0.05)

        chromosomes = []
        for _ in range(self._pop_size):
            chromosomes.append(search.Chromosome(self._random))
            chromosomes[-1].random(self._env)
        search.evaluateChromosomes(self._env, chromosomes)

        self._map_elites = {}
        self._update_map_elites(chromosomes)


    def update(self):
        new_chromosomes = []
        while len(new_chromosomes) < self._pop_size:
            child = self._select()
            if self._random.random() < self._cross_rate:
                parent = self._select()
                child = child.crossover(parent)
            child = child.mutation(self._env, self._mut_rate)
            new_chromosomes.append(child)

        search.evaluateChromosomes(self._env, new_chromosomes)
        self._update_map_elites(new_chromosomes)

        quality_passing_cells = len(list(filter(lambda c: c.quality() == 1, self._chromosomes)))
        print(f'cells that pass quality test: {quality_passing_cells}/{len(self._chromosomes)} ({100.0*quality_passing_cells/len(self._chromosomes):.2f}%)')


    def _select(self):
        size = self._tournment_size
        if size > len(self._chromosomes):
            size = len(self._chromosomes)

        tournment = list(range(len(self._chromosomes)))
        self._random.shuffle(tournment)

        chromosomes = []
        for i in range(size):
            chromosomes.append(self._chromosomes[tournment[i]])
        chromosomes.sort(key=lambda c: self._fitness_fn(c), reverse=True)

        return chromosomes[0]


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

        # OK
        if self._problem.startswith('binary-'):
            # (number of empty tiles, number of 4-long empty spaces / 10)
            arr = info['flat']
            return (sum(arr), sum(1 for i in range(len(arr) - 3) if all(arr[i:i+4])) // 10)

        # OK
        if self._problem.startswith('mdungeons-'):
            return (info['potions'] // 3, info['treasures'] // 3, info['enemies'] // 3)

        # OK
        if self._problem.startswith('zelda-'):
            player_location = info['pk_path'][0] if info['pk_path'] else -1
            key_location = info['kd_path'][0] if info['kd_path'] else -1
            door_location = info['kd_path'][-1] if info['kd_path'] else -1
            return (player_location, key_location, door_location)

        # OK
        if self._problem.startswith('isaac-'):
            return (info['dead_end'], info['locations'], sum(1 for x in info['flat'] if x == 0))

        # Can't find solution that passes quality
        if self._problem.startswith('talakat-'):
            return (info['script_connectivity'], round(info['bullet_coverage'], 1))



        if self._problem.startswith('building-'):
            return (info['blocks'], sum(1 for h in info['heights'] if h != 0))

        if self._problem.startswith('ddave-'):
            player_locations = info['player_locations']
            player_location = player_locations[0] if player_locations else -1

            exit_locations = info['exit_locations']
            exit_location = exit_locations[0] if exit_locations else -1

            return (player_location, exit_location)

        if self._problem.startswith('sokoban-'):
            return (info['crates'], info['targets'])

        raise RuntimeError(f'Unknown problem: {self._problem}')
