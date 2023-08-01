import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import gaussian_filter1d

from models.gnn_ca_simple import GNNCASimple
from util.graphs import get_cloud
from util.init_state import SphericalizeState
from util.state_cache import StateCache

# Setting up the device
USE_CUDA = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

if USE_CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def normalize_sphere(graph):
    offset = torch.mean(graph.x, dim=-2, keepdim=True)
    scale = torch.abs((graph.x)).max()
    graph.x = (graph.x - offset) / scale

    return graph

def run(graph, args):
    def train_step(x, edge_index, steps):
        #print(f"Tracing for {steps} steps")
        model.train()
        optimizer.zero_grad()

        out = model(x, edge_index, steps)
        out = out.reshape(args.batch_size, -1, out.shape[-1])
        loss = loss_fn(y.expand(args.batch_size, y.shape[0], y.shape[1]), out)

        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        return out, loss.item()

    # Prepare inputs for training
    y = graph.x
    edge_index_list = []
    for i in range(args.batch_size):
        edge_index_list.append(graph.edge_index)

    # Concatenate edge indices in batch
    edge_indices = [edge_index + i * y.shape[0] for i, edge_index in enumerate(edge_index_list)]
    edge_index = torch.cat(edge_indices, dim=1)
    state_cache = StateCache(
        SphericalizeState(y.cpu().numpy()),
        size=args.cache_size,
        reset_every=args.cache_reset_every,
    )

    # Train
    history = {
        "loss": [],
        "best_loss": np.inf,
        "best_model": None,
        "steps_avg": [],
        "steps_std": [],
        "steps_max": [],
        "steps_min": [],
    }
    best_loss = np.inf
    current_es_patience = args.es_patience
    current_lr_patience = args.lr_patience

    for i in range(args.epochs):
        loss = 0
        for _ in range(args.batches_in_epoch):
            x, idxs = state_cache.sample(args.batch_size)
            x = torch.tensor(x, dtype=torch.float32).view(-1, x.shape[-1])
            steps = torch.tensor(np.random.randint(args.min_steps, args.max_steps), dtype=torch.long)

            out, loss_step = train_step(x, edge_index, steps)
            out = out.detach().cpu().numpy()
            loss += loss_step

            # Update cache
            state_cache.update(idxs, out, steps.item())
            history["steps_avg"].append(np.mean(state_cache.counter))
            history["steps_std"].append(np.std(state_cache.counter))
            history["steps_max"].append(np.max(state_cache.counter))
            history["steps_min"].append(np.min(state_cache.counter))

        loss /= args.batches_in_epoch
        history["loss"].append(loss)

        print(
            f"Iter {i} - Steps: {steps.item():3d} - Loss: {loss:.10e} - "
            f"ES pat. {current_es_patience} - LR pat {current_lr_patience}"
        )

        if loss + args.tol < best_loss:
            best_loss = loss
            best_model = model.state_dict()
            current_es_patience = args.es_patience
            current_lr_patience = args.lr_patience
            print(f"Loss improved ({best_loss})")
        else:
            current_es_patience -= 1
            current_lr_patience -= 1
            if current_es_patience == 0:
                print("Early stopping")
                model.load_state_dict(best_model)
                break
            if current_lr_patience == 0:
                print(f"Reducing LR to {optimizer.param_groups[0]['lr'] * args.lr_red_factor}")
                current_lr_patience = args.lr_patience
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_red_factor
                model.load_state_dict(best_model)

    return history, state_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-3, type=float, help="Initial LR")
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Size of the mini-batches"
    )
    parser.add_argument("--epochs", default=100000, type=int, help="Training epochs")
    parser.add_argument(
        "--batches_in_epoch", default=10, type=int, help="Batches in an epoch"
    )
    parser.add_argument(
        "--es_patience", default=1000, type=int, help="Patience for early stopping"
    )
    parser.add_argument(
        "--lr_patience", default=750, type=int, help="Patience for LR annealing"
    )
    parser.add_argument(
        "--tol", default=1e-6, type=float, help="Tolerance for improvements"
    )
    parser.add_argument(
        "--lr_red_factor", default=0.1, type=float, help="Rate for LR annealing"
    )
    parser.add_argument("--min_steps", default=10, type=int, help="Minimum n. of steps")
    parser.add_argument("--max_steps", default=11, type=int, help="Maximum n. of steps")
    parser.add_argument(
        "--activation", default="relu", type=str, help="Activation for the GNCA"
    )
    parser.add_argument("--grad_clip", default=False, action=argparse.BooleanOptionalAction, help="Use gradient clipping")
    parser.add_argument(
        "--cache_size", default=1024, type=int, help="Size of the cache"
    )
    parser.add_argument(
        "--cache_reset_every",
        default=32,
        type=int,
        help="How often to reset one state in cache",
    )
    parser.add_argument(
        "--outdir", default="results", type=str, help="Where to save output"
    )
    parser.add_argument(
        "--graphs", default="Grid2d", type=str, help="Which graphs to use: Grid2d, Bunny, Minnesota, Logo, SwissRoll"
    )
    args = parser.parse_args()

    graphs = []
    if "Grid2d" in args.graphs:
        graphs.append(get_cloud("Grid2d", N1=20, N2=20))
    if "Bunny" in args.graphs:
        graphs.append(get_cloud("Bunny"))
    if "Minnesota" in args.graphs:
        graphs.append(get_cloud("Minnesota"))
    if "Logo" in args.graphs:
        graphs.append(get_cloud("Logo"))
    if "SwissRoll" in args.graphs:
        graphs.append(get_cloud("SwissRoll", N=200))

    if len(graphs) == 0:
        print("No graphs selected")
        exit()

    for graph in graphs:
        normalized_graph = normalize_sphere(graph)

        model = GNNCASimple(graph.x.shape[-1], activation=args.activation, batch_norm=False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        history, state_cache = run(graph, args)

        # Unpack data
        y = graph.x.cpu().numpy()
        edge_index = graph.edge_index

        # Run model for the twice the maximum number of steps in the cache
        x = state_cache.initial_state()
        steps = 2 * int(np.max(state_cache.counter))
        zs = [x]
        model.eval()
        with torch.no_grad():
            for _ in range(steps):
                z = model(torch.tensor(zs[-1]), edge_index, 1)
                zs.append(z.detach().cpu().numpy())
        zs = np.stack(zs, 0)
        z = zs[-1]

        out_dir = f"{args.outdir}/{graph.name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/config.txt", "w") as f:
            f.writelines([f"{k}={v}\n" for k, v, in vars(args).items()])
        np.savez(f"{out_dir}/run_point_cloud.npz", y=y, z=z, history=history, zs=zs)

        # Plot difference between target and output points
        plt.figure(figsize=(2.5, 2.5))
        cmap = plt.get_cmap("Set2")
        plt.scatter(*y[:, :2].T, color=cmap(0), marker=".", s=1)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/target.pdf")

        plt.figure(figsize=(2.5, 2.5))
        cmap = plt.get_cmap("Set2")
        plt.scatter(*z[:, :2].T, color=cmap(1), marker=".", s=1)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/endstate.pdf")

        # Plot loss and loss trend
        plt.figure(figsize=(2.6, 2.5))
        cmap = plt.get_cmap("Set2")
        plt.plot(history["loss"], alpha=0.3, color=cmap(0), label="Real")
        plt.plot(gaussian_filter1d(history["loss"], 50), color=cmap(0), label="Trend")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/loss.pdf")

        # Plot change between consecutive state
        plt.figure(figsize=(2.5, 2.5))
        cmap = plt.get_cmap("Set2")
        change = np.abs(zs[:-1] - zs[1:]).mean((1, 2))
        loss = np.array([loss_fn(torch.tensor(y), torch.tensor(zs[i])).cpu().numpy() for i in range(len(zs))])
        plt.plot(change, label="Abs. change", color=cmap(0))
        plt.plot(loss, label="Loss", color=cmap(1))
        plt.xlabel("Step")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/change.pdf")

        # Plot evolution of states
        n_states = 10
        plt.figure(figsize=(n_states * 2.0, 2.1))
        for i in range(n_states):
            plt.subplot(1, n_states, i + 1)
            plt.scatter(*zs[i, :, :2].T, color=cmap(1), marker=".", s=1)
            plt.title(f"t={i}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/evolution.pdf")

        # Plot the average number of steps for the states in the cache
        plt.figure(figsize=(2.5, 2.5))
        cmap = plt.get_cmap("Set2")
        s_avg, s_std = np.array(history["steps_avg"]), np.array(history["steps_std"])
        s_max, s_min = np.array(history["steps_max"]), np.array(history["steps_min"])
        plt.plot(s_avg, label="Avg.", color=cmap(0))
        plt.fill_between(
            np.arange(len(s_std)),
            s_avg - s_std,
            s_avg + s_std,
            alpha=0.5,
            color=cmap(0),
        )
        plt.plot(s_max, linewidth=0.5, linestyle="--", color="k", label="Max")
        plt.xlabel("Epoch")
        plt.ylabel("Number of steps in cache")
        plt.legend()
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/steps_in_cache.pdf")

    plt.show()
