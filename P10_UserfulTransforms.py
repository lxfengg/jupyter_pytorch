from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

image_path = "images/0013035.jpg"
img_PIL = Image.open(image_path)

tensor = transforms.ToTensor()
img_tensor = tensor(img_PIL)

normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_normalize = normalize(img_tensor)

transforms_resize = transforms.Resize((512, 512))
img_resize = transforms_resize(img_PIL)
img_resize = tensor(img_resize)
print(img_resize)

resize = transforms.Resize(256)
compose = transforms.Compose([resize, tensor])
img_com = compose(img_PIL)
print(img_com)

writer = SummaryWriter("logs")
writer.add_image("ToTensor", img_tensor)
writer.add_image("Norm", img_normalize)
writer.add_image("Resize", img_resize, 0)
writer.add_image("Resize", img_com, 1)
writer.close()