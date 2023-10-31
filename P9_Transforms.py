from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
tensor = transforms.ToTensor()
img_tensor = tensor(img_PIL)

writer = SummaryWriter("logs")
writer.add_image("tensor_img", img_tensor, 1)
writer.close()