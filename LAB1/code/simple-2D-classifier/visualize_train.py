from lib.dataset import Simple2DDataset, Simple2DTransformDataset
from lib.networks import LinearClassifier, MLPClassifier

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

BATCH_SIZE = 500
NUM_WORKERS = 1

if __name__ == "__main__":
    valid_dataset = Simple2DDataset(split='valid')
    valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )

    net = LinearClassifier()
    net.load_state_dict(torch.load("best-linear.pth")["net"])
    net.eval()

    for batch in valid_dataloader:
        with torch.no_grad():
            output = net(batch['input'])
        fig, axes = plt.subplots(1, 2)
        axes[0].scatter(batch['input'][:,0], batch['input'][:,1], c=batch["annotation"], vmin=0, vmax=1)
        axes[0].set_title("Ground Truth of Validation Data")
        axes[0].set_aspect(1)

        axes[1].scatter(batch['input'][:,0],batch['input'][:,1], c=output, vmin=0, vmax=1)
        axes[1].set_title("Predicted Result")
        axes[1].set_aspect(1)

        # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdBu))
        plt.savefig("valid_result.pdf", bbox_inches="tight")
        plt.show()


