import matplotlib.pyplot as plt
import torch

def plot_prediction(dataset, config, maml, model):
    x_test, sort_indices = torch.sort(dataset[0])
    y_test = dataset[1][sort_indices]
    split_data = config['train_val_split_function'](
        eps_data=dataset, k_shot=config['k_shot'])

    # move data to GPU (if there is a GPU)
    x_t = split_data['x_t'].to(config['device'])  # k_shot=8
    y_t = split_data['y_t'].to(config['device'])  # k_shot=8

    # MAML
    adapted_hyper_net = maml.adaptation(
        x=x_t[:, None], y=y_t[:, None], model=model)
    y_maml = maml.prediction(
        x=x_test[:, None], adapted_hyper_net=adapted_hyper_net, model=model)

    # plot
    plt.figure(figsize=(4, 4))
    plt.scatter(x=x_t.cpu().numpy(), y=y_t.cpu().numpy(),
                s=80, marker='^', color='C0')  # samples
    plt.plot(x_test.cpu().numpy(), y_test, color='black',
             linewidth=1, linestyle='-')  # true task data
    plt.plot(x_test.cpu().numpy(), y_maml.detach(
    ).cpu().numpy(), color='C2', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()
