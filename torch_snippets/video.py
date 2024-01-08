from torch_snippets import *
from IPython.display import display, clear_output
from matplotlib.animation import FuncAnimation
from matplotlib import rc

rc("animation", html="jshtml")


def get_sz(w, sz=None):
    if sz is None:
        if w < 129:
            sz = 2
        elif w < 1025:
            sz = 5
        else:
            sz = 10
    if isinstance(sz, int):
        sz = (sz, sz)
    return sz


def denormalize_image(img):
    Debug(f"DeNormalizing...")
    if img.min() < 0:
        img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return img


def animate(frames, sz=None):
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
        Info(f"Animating {torch.Tensor(frames)}")

    # if isinstance(frames, list):
    #     frames = np.array(frames)
    if isinstance(frames, np.ndarray):
        if len(frames.shape) == 5:
            # assuming BXXXX
            Warn(f"Received a 5D tensor. Animating only the first one in the batch")
            frames = frames[0]

        if len(frames.shape) == 4:
            if frames.shape[1] == 3:
                # assuming NCHW
                frames = frames.transpose(0, 2, 3, 1)  # Convert to NHWC format
            if frames.shape[0] == 3:
                # assuming CNHW
                frames = frames.transpose(1, 2, 3, 0)  # Convert to NHWC format

        if frames.min() < 0 or frames.max() > 1:
            frames = denormalize_image(frames)

    w = frames[0].shape[1]
    sz = get_sz(w, sz)

    def init():
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, 1)
        return []

    def update(frame):
        ax.clear()
        img = frames[frame]
        ax.imshow(img)
        ax.axis("off")
        return []

    fig, ax = plt.subplots(figsize=sz)
    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True)
    plt.close()
    return ani
