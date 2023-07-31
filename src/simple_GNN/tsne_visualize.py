import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_visualize(config="3_layer_lr_00001_long_long_deep", valid="valid", perplexity=30):
    # load torch tensors from /output/3_layer_lr_00001_long_long_deep_test_embedding.pt
    # load torch tensors from /output/3_layer_lr_00001_long_long_deep_test_embedding.pt
    embedding = torch.load(f"output/{config}_{valid}_embedding.pt",
                            map_location=torch.device('cpu')).detach().numpy() 
    label = torch.load(f"output/{config}_{valid}_label.pt",
                        map_location=torch.device('cpu')).detach().numpy() 
    pred = torch.load(f"output/{config}_{valid}_pred.pt",
                        map_location=torch.device('cpu')).detach().numpy() 

    # Extract positive labels index
    pos_idx = label == 1
    pos_idx = pos_idx.nonzero()

    # extract negative labels
    neg_idx = label == 0
    neg_idx = neg_idx.nonzero()

    # import tsne
    # visualize embedding with tsne
    tsne = TSNE(n_components=2, random_state=0, perplexity = perplexity)

    # fit tsne
    x_embedded = tsne.fit_transform(embedding)

    # plot tsne
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(x_embedded[pos_idx, 0], x_embedded[pos_idx, 1], c='Blue', alpha = 0.5, label="Bot", cmap="jet")
    ax.scatter(x_embedded[neg_idx, 0], x_embedded[neg_idx, 1], c='Red', alpha = 0.5, label="Human", cmap="jet")

    # Add legend of color
    ax.legend()

    # save plot
    plt.savefig(f"output/{config}_{valid}_embedding_perp_{perplexity}.png")

if __name__ == "__main__":
    tsne_visualize(config="3_layer_lr_00001_long_long_deep", valid="valid", perplexity=30)
    tsne_visualize(config="3_layer_lr_00001_long_long_deep", valid="test", perplexity=30)
    tsne_visualize(config="3_layer_lr_00001_long_long_deep", valid="train", perplexity=30)