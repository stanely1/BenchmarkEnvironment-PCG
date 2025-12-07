import argparse
import pcg_benchmark
from importlib import import_module
import fire

def isFloat(number):
    try:
        test = float(number)
        return True
    except:
        return False

def convert2Dic(commands):
    if len(commands) % 2 == 1:
        raise ValueError("inputs have to be tuples example (--fitness quality).")
    result = {}
    for i in range(0, len(commands), 2):
        key = commands[i]
        if key.startswith('--'):
            key = key.split('--')[-1]
        if commands[i+1].isnumeric():
            result[key] = int(commands[i+1])
        elif isFloat(commands[i+1]):
            result[key] = float(commands[i+1])
        else:
            result[key] = commands[i+1]
    return result

def main(folder: str = 'outputs',
         problem: str = 'binary-v0',
         generator: str = 'random',
         steps: int = 100,
         early_stop: bool = False,
         seed: int = None,
         **kwargs):
    """
    Run a generator from the list of generators.

    Args:
        folder (str): The folder to save the search results
        problem (str): Problem to solve ('binary-v0', 'zelda-v0', 'sokoban-v0', etc.)
        generator (str): Generator file name from generators folder ('ga', 'es', 'random', etc.)
        steps (int): Number of iterations to run the generator
        early_stop (bool): Stop generation when the best fitness reaches 1
        seed (int): Random seed for the environment (optional)
        **kwargs: Additional arguments passed to the generator (e.g., fitness='quality_control')

    Examples:
        python run.py --folder=outputs --problem=sokoban-v0 --generator=ga --steps=100
        python run.py --folder=outputs --problem=sokoban-v0 --generator=ga --steps=100 --fitness=quality_control
        python run.py --folder=outputs --problem=sokoban-v0 --generator=ga --steps=100 --early_stop=True --seed=42
    """

    outputfolder = folder

    # Create environment
    env = pcg_benchmark.make(problem)
    if seed is not None:
        env.seed(seed)

    # Import and initialize generator
    module = import_module(f"generators.{generator}")
    if not hasattr(module, "Generator"):
        raise ValueError(f"generators.{generator}.Generator doesn't exist.")
    gen = module.Generator(env)

    # Start generation
    print(f"Starting {generator}:")
    print(f"  Problem: {problem}")
    print(f"  Output folder: {outputfolder}")
    print(f"  Steps: {steps}")
    print(f"  Early stop: {early_stop}")
    if seed is not None:
        print(f"  Seed: {seed}")
    if kwargs:
        print(f"  Additional parameters: {kwargs}")
    print()

    kwargs['problem'] = problem

    # Reset generator with additional kwargs
    gen.reset(**kwargs)
    print(f"  Iteration 0: {gen.best():.2f}")
    gen.save(f"{outputfolder}/iter_0")

    # Run iterations
    for i in range(steps):
        gen.update()
        print(f"  Iteration {i+1}: {gen.best():.2f}")
        gen.save(f"{outputfolder}/iter_{i+1}")

        if early_stop and gen.best() >= 1:
            print(f"\nEarly stopping at iteration {i+1} (fitness >= 1.0)")
            break

    print(f"\nGeneration complete! Results saved to '{outputfolder}'")

if __name__ == "__main__":
    fire.Fire(main)
