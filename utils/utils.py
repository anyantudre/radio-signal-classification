import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms as T

from .spec_augment import TimeMask, FreqMask
from dataset.dataset import CLASSES


def print_one_image(dataset, idx):
    row = dataset.iloc[idx]

    image_pixels = np.array(row.iloc[:8192], dtype=np.float64)
    label = row.labels

    image = np.resize(image_pixels, (64, 128))  # 64 * 128 = 8192

    plt.imshow(image)
    plt.title(label)
    plt.show()


def get_train_tranform():
    return T.Compose([
        TimeMask(T = 15, num_masks = 4),
        FreqMask(F = 15, num_masks = 4)
    ])


def multiclass_accuracy(y_pred,y_true):
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))
    
    
def view_classify(img, ps, true_label = None):
    classes = CLASSES

    ps = ps.data.cpu().numpy().squeeze()
    img = img.numpy()
   
    fig, (ax1, ax2) = plt.subplots(figsize=(12,8), ncols=2)
    ax1.imshow(img)
    ax1.axis('off')
    if true_label is not None:
        ax1.set_title(f'Ground-Truth : {true_label}')
    ax2.barh(np.arange(len(classes)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None


def confusion_matrix(y_true, y_pred, num_classes = 4):
    # rows are true labels, columns are predictions
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, classes = CLASSES, title = None, save_path = None):
    cmap = LinearSegmentedColormap.from_list('blues', ['#ffffff', '#cde2fb', '#6da7ec', '#256abf', '#0d366b'])
    row_share = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ink = '#ffffff' if row_share[i, j] > 0.55 else '#1f1f1e'
            ax.text(j, i - 0.08, f'{cm[i, j]}', ha='center', va='center', color=ink, fontsize=13, fontweight='bold')
            ax.text(j, i + 0.2, f'{row_share[i, j]:.0%}', ha='center', va='center', color=ink, fontsize=9)

    # thin white gaps between cells
    ax.set_xticks(np.arange(-0.5, len(classes)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(classes)), minor=True)
    ax.grid(which='minor', color='#ffffff', linewidth=2)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=0, colors='#4a4a48')
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted class', color='#4a4a48')
    ax.set_ylabel('True class', color='#4a4a48')
    if title:
        ax.set_title(title, color='#1f1f1e', loc='left', fontsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#ffffff')
    return fig
