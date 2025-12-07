import json
import os
from typing import List, Optional, Tuple, Dict

from itertools import cycle
import fire
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import pcg_benchmark



def compute_fitness(quality_value, controllability_value, diversity_value,
                    fitness_type):
    if fitness_type == 'quality':
        return quality_value
    elif fitness_type == 'quality_control':
        result = quality_value
        if quality_value >= 1:
            result += controllability_value
        return result / 2.0
    elif fitness_type == 'quality_control_diversity':
        result = quality_value
        if quality_value >= 1:
            result += controllability_value
        if quality_value >= 1 and controllability_value >= 1:
            result += diversity_value
        return result / 3.0


def compute_population_fitness_stats(folder_path: str,
                                     fitness_type: str = 'quality_control') -> Dict[int, Dict[str, float]]:
    """
    Compute best and average fitness for each generation in a folder.

    Args:
        folder_path (str): Path to folder containing generation subfolders (iter_0, iter_1, etc.)
        fitness_type (str): Type of fitness ('quality', 'quality_control', or 'quality_control_diversity')

    Returns:
        dict: Dictionary mapping generation number to {'best_fitness', 'avg_fitness', 'std_fitness',
                                                       'best_quality', 'avg_quality',
                                                       'best_diversity', 'avg_diversity',
                                                       'best_controlability', 'avg_controlability',
                                                       'percentage_quality_pass'}
    """
    generation_stats = {}

    # Get all iteration folders
    iter_folders = [f for f in os.listdir(folder_path) if f.startswith('iter_') and
                    os.path.isdir(os.path.join(folder_path, f))]

    # Sort folders by iteration number
    iter_folders.sort(key=lambda x: int(x.replace('iter_', '')))

    for iter_folder in iter_folders:
        iter_path = os.path.join(folder_path, iter_folder)
        iter_num = int(iter_folder.replace('iter_', ''))

        # Collect all fitness values and metrics for this generation
        fitness_values = []
        quality_values = []
        diversity_values = []
        controlability_values = []

        for json_file in os.listdir(iter_path):
            if json_file.endswith(".json"):
                file_path = os.path.join(iter_path, json_file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                    quality = data["quality"]
                    diversity = data["diversity"]
                    controlability = data["controlability"]

                    fitness = compute_fitness(quality, controlability, diversity, fitness_type)

                    fitness_values.append(fitness)
                    quality_values.append(quality)
                    diversity_values.append(diversity)
                    controlability_values.append(controlability)

        # Compute statistics
        if fitness_values:
            generation_stats[iter_num] = {
                'best_fitness': max(fitness_values),
                'avg_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'median_fitness': np.median(fitness_values),
                'min_fitness': min(fitness_values),

                'best_quality': max(quality_values),
                'avg_quality': np.mean(quality_values),

                'best_diversity': max(diversity_values),
                'avg_diversity': np.mean(diversity_values),

                'best_controlability': max(controlability_values),
                'avg_controlability': np.mean(controlability_values),

                'population_size': len(fitness_values),
                'percentage_quality_pass': 100.0 * len([q for q in quality_values if q == 1]) / len(quality_values)
            }

    return generation_stats


def get_best_chromosome_from_generation(folder_path: str,
                                        iter_num: int,
                                        fitness_type: str = 'quality_control') -> Tuple[Optional[dict], Optional[str]]:
    """
    Get the best chromosome from a specific generation.

    Args:
        folder_path (str): Path to folder containing generation subfolders
        iter_num (int): Generation number
        fitness_type (str): Type of fitness to compute

    Returns:
        tuple: (best_chromosome_data, best_filename) or (None, None) if not found
    """
    iter_path = os.path.join(folder_path, f'iter_{iter_num}')

    if not os.path.exists(iter_path):
        return None, None

    best_fitness = -np.inf
    best_chromosome = None
    best_filename = None

    for json_file in os.listdir(iter_path):
        if json_file.endswith(".json"):
            file_path = os.path.join(iter_path, json_file)
            with open(file_path, "r") as f:
                data = json.load(f)

                quality = data["quality"]
                diversity = data["diversity"]
                controlability = data["controlability"]

                fitness = compute_fitness(quality, controlability, diversity, fitness_type)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_chromosome = data
                    best_filename = json_file

    return best_chromosome, best_filename


def render_chromosome(chromosome, env, target_dir, file_name):
    # Render based on environment type
    if env._name != 'talakat-v0':
        # For most environments, render returns an image
        img = env.render(chromosome['content'])
        output_filename = os.path.join(target_dir, f'{file_name}.png')
        img.save(output_filename)
    else:
        # For talakat-v0, render returns frames for a GIF
        frames = env.render(chromosome['content'])
        output_filename = os.path.join(target_dir, f'{file_name}.gif')
        frames[0].save(
            output_filename,
            append_images=frames[1:],
            save_all=True,
            duration=100,
            loop=0
        )


def render_best_individuals(folder_path: str,
                            env_name: str,
                            results_dir: str,
                            fitness_type: str = 'quality_control'):
    """
    Render the best individual from each generation.
    Additionally, render all individuals from last generation, that pass the quality test.

    Args:
        folder_path (str): Path to folder containing generation subfolders
        env_name (str): Environment name (e.g., 'sokoban-v0', 'smb-v0')
        results_dir (str): Directory to save rendered images
        fitness_type (str): Type of fitness to compute
    """
    # Create renders subdirectories
    renders_dir = os.path.join(results_dir, 'renders'+'_'+fitness_type)
    best_renders_dir = os.path.join(renders_dir, 'best_evolution')
    last_renders_dir = os.path.join(renders_dir, 'last_gen')
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(best_renders_dir, exist_ok=True)
    os.makedirs(last_renders_dir, exist_ok=True)

    # Initialize environment
    env = pcg_benchmark.make(env_name)

    # Get all iteration folders
    iter_folders = [f for f in os.listdir(folder_path) if f.startswith('iter_') and
                    os.path.isdir(os.path.join(folder_path, f))]

    # Sort folders by iteration number
    iter_folders.sort(key=lambda x: int(x.replace('iter_', '')))

    print(f"\n=== Rendering Best Individuals ===")
    for iter_folder in tqdm(iter_folders, desc="Rendering generations"):
        iter_num = int(iter_folder.replace('iter_', ''))

        best_chromosome, best_filename = get_best_chromosome_from_generation(
            folder_path, iter_num, fitness_type
        )

        if best_chromosome is None:
            continue

        try:
            render_chromosome(best_chromosome, env, best_renders_dir, f'gen_{iter_num:04d}_best')
        except Exception as e:
            print(f"Error rendering generation {iter_num}: {e}")

    print(f"\n=== Rendering Last Generation ===")
    iter_path = os.path.join(folder_path, f'iter_{iter_num}')
    for json_file in os.listdir(iter_path):
        if json_file.endswith(".json"):
            file_path = os.path.join(iter_path, json_file)
            with open(file_path, "r") as f:
                data = json.load(f)
                quality = data["quality"]
                if quality == 1:
                    try:
                        render_chromosome(data, env, last_renders_dir, json_file.replace('.json', ''))
                    except Exception as e:
                        print(f"Error rendering last generation: {e}")


    print(f"Rendered images saved to '{renders_dir}'")


def plot_fitness_over_generations(stats_dict: Dict[int, Dict[str, float]],
                                  output_path: str = 'fitness.png'):
    """
    Plot fitness evolution over generations.

    Args:
        stats_dict: Dictionary from compute_population_fitness_stats
        output_path: Where to save the plot
    """
    generations = sorted(stats_dict.keys())
    best_fitness = [stats_dict[g]['best_fitness'] for g in generations]
    avg_fitness = [stats_dict[g]['avg_fitness'] for g in generations]
    std_fitness = [stats_dict[g]['std_fitness'] for g in generations]
    percentage_quality_pass = [stats_dict[g]['percentage_quality_pass'] for g in generations]

    plt.figure(figsize=(12, 6))

    ax1 = plt.gca()
    ax1.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
    ax1.plot(generations, avg_fitness, 'r-', label='Average Fitness', linewidth=2)

    avg_array = np.array(avg_fitness)
    std_array = np.array(std_fitness)
    ax1.fill_between(generations,
                    avg_array - std_array,
                    avg_array + std_array,
                    alpha=0.3,
                    color='red')

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(generations, percentage_quality_pass, 'g-', label='% Passing Quality Test', linewidth=2)
    ax2.set_ylabel('Percentage (%)')
    ax2.tick_params(axis='y')

    plt.title('Fitness Evolution Over Generations')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")


def main(folder: str = 'results',
         fitness: str = 'quality_control',
         problem: str = 'sokoban-v0',
         render: bool = True):
    """
    Main function to analyze fitness statistics and generate plots.

    Args:
        folder (str): Name of the folder containing generation data
        fitness (str): Type of fitness to compute ('quality', 'quality_control', or 'quality_control_diversity')
        problem (str): Environment name for rendering (e.g., 'sokoban-v0', 'smb-v0', 'zelda-v0')
        render (bool): Whether to render best individuals from each generation
    """
    folder_name = folder
    fitness_type = fitness
    env_name = problem

    folder_path = './' + folder_name + '/'

    # Create results directory
    results_dir = folder_name + '_processed'
    os.makedirs(results_dir, exist_ok=True)

    stats = compute_population_fitness_stats(folder_path, fitness_type=fitness_type)

    # Create summary file
    summary_file_path = os.path.join(results_dir, f'summary_{fitness_type}.txt')

    # Print statistics to console and file
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("=== Fitness Statistics by Generation ===")
    summary_lines.append("=" * 60)
    summary_lines.append(f"\nFolder: {folder_name}")
    summary_lines.append(f"Fitness Type: {fitness_type}")
    summary_lines.append(f"Environment: {env_name}")
    summary_lines.append(f"\n{'=' * 60}\n")

    for gen_num, gen_stats in sorted(stats.items()):
        gen_summary = f"\nGeneration {gen_num}:\n"
        gen_summary += f"  Best Fitness: {gen_stats['best_fitness']:.4f}\n"
        gen_summary += f"  Avg Fitness: {gen_stats['avg_fitness']:.4f} ± {gen_stats['std_fitness']:.4f}\n"
        gen_summary += f"  Median Fitness: {gen_stats['median_fitness']:.4f}\n"
        gen_summary += f"  Min Fitness: {gen_stats['min_fitness']:.4f}\n"
        gen_summary += f"  Population Size: {gen_stats['population_size']}\n"
        gen_summary += f"  Best Quality: {gen_stats['best_quality']:.4f}\n"
        gen_summary += f"  Avg Quality: {gen_stats['avg_quality']:.4f}\n"
        gen_summary += f"  Best Diversity: {gen_stats['best_diversity']:.4f}\n"
        gen_summary += f"  Avg Diversity: {gen_stats['avg_diversity']:.4f}\n"
        gen_summary += f"  Best Controlability: {gen_stats['best_controlability']:.4f}\n"
        gen_summary += f"  Avg Controlability: {gen_stats['avg_controlability']:.4f}\n"

        summary_lines.append(gen_summary)
        print(gen_summary)

    # Add overall statistics
    if stats:
        all_best_fitness = [stats[g]['best_fitness'] for g in sorted(stats.keys())]
        all_avg_fitness = [stats[g]['avg_fitness'] for g in sorted(stats.keys())]

        overall_summary = f"\n{'=' * 60}\n"
        overall_summary += "=== Overall Statistics ===\n"
        overall_summary += f"{'=' * 60}\n"
        overall_summary += f"\nTotal Generations: {len(stats)}\n"
        overall_summary += f"Overall Best Fitness: {max(all_best_fitness):.4f}\n"
        overall_summary += f"Overall Worst Best Fitness: {min(all_best_fitness):.4f}\n"
        overall_summary += f"Average of Best Fitness: {np.mean(all_best_fitness):.4f} ± {np.std(all_best_fitness):.4f}\n"
        overall_summary += f"Average of Avg Fitness: {np.mean(all_avg_fitness):.4f} ± {np.std(all_avg_fitness):.4f}\n"

        # Find generation with best fitness
        best_gen = max(stats.keys(), key=lambda g: stats[g]['best_fitness'])
        overall_summary += f"\nBest Generation: {best_gen} (fitness: {stats[best_gen]['best_fitness']:.4f})\n"

        summary_lines.append(overall_summary)
        print(overall_summary)

    # Write to file
    with open(summary_file_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"\nSummary saved to '{summary_file_path}'")

    # Plot if there are multiple generations
    if len(stats) > 1:
        plot_output_path = os.path.join(results_dir, 'fitness_'+fitness_type+'.png')
        plot_fitness_over_generations(stats, plot_output_path)

    # Save to CSV
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.index.name = 'generation'
    csv_output_path = os.path.join(results_dir, 'generation_statistics.csv')
    df.to_csv(csv_output_path)
    print(f"Statistics saved to '{csv_output_path}'")
    print(f"Results directory: '{results_dir}'")

    # Render best individuals if requested
    if render:
        render_best_individuals(folder_path, env_name, results_dir, fitness_type)

# Example usage:
if __name__ == "__main__":
    fire.Fire(main)
