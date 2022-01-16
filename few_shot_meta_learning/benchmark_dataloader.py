import torch
import typing
from few_shot_meta_learning.create_benchmarks import create_benchmarks

"""
    This Dataset is a wrapper for a benchmark (all tasks)
"""
class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, benchmark) -> None:
        super().__init__()
        self.n_tasks = benchmark.n_task
        self.tasks = [None] * benchmark.n_task
        for i in range(benchmark.n_task):
            self.tasks[i] = benchmark.get_task_by_index(i)

    def __len__(self) -> int:
        return self.n_tasks

    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        task = self.tasks[index]
        x = torch.tensor(task.x, dtype=torch.float32).squeeze()
        y = torch.tensor(task.y, dtype=torch.float32).squeeze()
        return [x, y]

def create_benchmark_dataloaders(config):
    bm_meta, bm_val, bm_test = create_benchmarks(config)
    train_data_loader = torch.utils.data.DataLoader(BenchmarkDataset(bm_meta))
    val_data_loader = torch.utils.data.DataLoader(BenchmarkDataset(bm_val))
    test_data_loader = torch.utils.data.DataLoader(BenchmarkDataset(bm_test))
    return train_data_loader, val_data_loader, test_data_loader
    

