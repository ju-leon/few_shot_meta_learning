from mtutils.mtutils import BM_DICT, collate_data

from metalearning_benchmarks.benchmarks.util import normalize_benchmark
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
import numpy as np

from few_shot_meta_learning.sinusoid_affine_benchmark import SinusoidAffineBenchmark

def create_benchmarks(config: dict):
    # extend benchmark dict
    BM_DICT['SinusoidAffine1D'] = SinusoidAffineBenchmark
    # create benchmarks
    bm_meta = BM_DICT[config["benchmark"]](
        n_task=config["minibatch"],
        n_datapoints_per_task=config["points_per_minibatch"],
        output_noise=config["noise_stddev"],
        seed_task=config["seed_offset"],
        seed_x=config["seed_offset"] + 1,
        seed_noise=config["seed_offset"] + 2,
    )
    bm_test = BM_DICT[config["benchmark"]](
        n_task=config["minbatch_test"],
        n_datapoints_per_task=config["points_per_minibatch_test"],
        output_noise=config["noise_stddev"],
        seed_task=config["seed_offset_test"],
        seed_x=config["seed_offset_test"] + 1,
        seed_noise=config["seed_offset_test"] + 2,
    )
    if config['normalize_benchmark']:
        bm_meta = normalize_benchmark(bm_meta)
        bm_test = normalize_benchmark(bm_test)
    return bm_meta, bm_test


def _prepare_benchmark(bm: MetaLearningBenchmark, n_points_pred: int, n_task: int):
    x, y = collate_data(bm)
    bounds = bm.x_bounds[0, :]
    lower = bounds[0] - 0.1 * (bounds[1] - bounds[0])
    higher = bounds[1] + 0.1 * (bounds[1] - bounds[0])
    x_pred = np.linspace(lower, higher, n_points_pred)[
        None, :, None].repeat(n_task, axis=0)
    return x, y, x_pred


def create_extracted_benchmarks(config: dict):
    bm_meta, bm_test = create_benchmarks(config)
    x_meta, y_meta, x_pred_meta = _prepare_benchmark(
        bm_meta, config['n_points_pred'], config['minibatch'])
    x_test, y_test, x_pred_test = _prepare_benchmark(
        bm_test, config['n_points_pred'], config['minibatch_test'])
    return x_meta, y_meta, x_test, y_test, x_pred_meta, x_pred_test
