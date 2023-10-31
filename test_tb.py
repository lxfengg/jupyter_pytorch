from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img_path = "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
img_np = np.array(img_PIL)

writer.add_image("test", img_np, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()