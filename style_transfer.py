import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Image loader
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    size = min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Show tensor image
def imshow(tensor, title=None):
    image = tensor.clone().squeeze()
    image = image.mul(torch.tensor([0.229, 0.224, 0.225]).view(3,1,1))
    image = image.add(torch.tensor([0.485, 0.456, 0.406]).view(3,1,1))
    image = image.clamp(0, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    if title:
        plt.title(title)
    plt.pause(0.001)

# Define loss functions
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a*b, c*d)
    G = torch.mm(features, features.t())
    return G.div(a*b*c*d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images
content_image = load_image("content.jpg")
style_image = load_image("style.jpg")

# Display
plt.figure()
imshow(content_image, title='Content Image')
plt.figure()
imshow(style_image, title='Style Image')

# VGG Model
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Layers to extract
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Build model
content_losses = []
style_losses = []
model = nn.Sequential()

i = 0
for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = f"conv_{i}"
    elif isinstance(layer, nn.ReLU):
        name = f"relu_{i}"
        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = f"pool_{i}"
    elif isinstance(layer, nn.BatchNorm2d):
        name = f"bn_{i}"
    else:
        raise RuntimeError("Unrecognized layer")

    model.add_module(name, layer)

    if name in content_layers:
        target = model(content_image).detach()
        content_loss = ContentLoss(target)
        model.add_module(f"content_loss_{i}", content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        target = model(style_image).detach()
        style_loss = StyleLoss(target)
        model.add_module(f"style_loss_{i}", style_loss)
        style_losses.append(style_loss)

# Trim off the unnecessary layers after last content/style loss
for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], (ContentLoss, StyleLoss)):
        break
model = model[:(i + 1)]

# Input image (copy of content)
input_img = content_image.clone().requires_grad_(True)

# Optimizer
optimizer = optim.LBFGS([input_img])

# Run style transfer
run = [0]
while run[0] <= 300:
    def closure():
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_score * 1e6 + content_score
        loss.backward()
        run[0] += 1
        return loss
    optimizer.step(closure)

# Result
plt.figure()
imshow(input_img, title="Output Image")
plt.show()
