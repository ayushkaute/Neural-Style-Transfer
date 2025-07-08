# Neural-Style-Transfer

*COMPANY: CODTECH IT SOLUTIONS

*NAME: AYUSH MACHHINDRA KAUTE

*INTERN ID: CT04DF1740

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEEKS

*MENTOR: NEELA SANTOSH

Description:

This script performs Neural Style Transfer — a technique that blends the content of one image with the style of another using a deep neural network. It’s based on a 2015 paper by Gatys et al., where they showed that convolutional neural networks (CNNs) could separate and recombine content and style from images.

📁 1. Importing Libraries
The script imports:
torch, torch.nn, and torch.optim for deep learning operations.
torchvision for pre-trained models and image transforms.
PIL.Image for image loading.
matplotlib.pyplot to visualize the results.

🖼 2. Image Loading and Preprocessing

load_image()

This function loads an image from a given path and:
Resizes it to a manageable size (max 400 px).
Converts it into a normalized tensor suitable for the VGG19 model.
Normalization uses ImageNet’s mean and std values because VGG was trained on that dataset.

🖼 3. Displaying Images

imshow()

This helper function denormalizes and displays a PyTorch tensor image using matplotlib.

🧠 4. Loss Functions
ContentLoss: Measures how different the content of the generated image is from the content image using Mean Squared Error (MSE).
StyleLoss: Computes the style difference using Gram matrices. A Gram matrix is a way to capture the texture (style) by comparing feature correlations within a layer.

💻 5. Load and Display Images

content_image = load_image("content.jpg")
style_image = load_image("style.jpg")

Images are loaded and shown using imshow().

🧠 6. Load Pretrained VGG19

cnn = models.vgg19(pretrained=True).features.to(device).eval()

We load the feature extractor part of the pre-trained VGG19 model. It’s used because it’s deep and proven effective at capturing hierarchical visual features.

🔍 7. Select Content and Style Layers

content_layers = ['conv_4']
style_layers = ['conv_1', ..., 'conv_5']

These are specific VGG layers where we compute style and content losses. Lower layers capture textures (style), and deeper layers capture structure (content).

🧱 8. Build the Model
The model is created by looping over layers in VGG19 and:
Adding each to a new nn.Sequential model.
Inserting ContentLoss and StyleLoss modules at the specified layers.
Stopping after the last loss layer to save computation.

🧑‍🎨 9. Create the Input Image

input_img = content_image.clone().requires_grad_(True)

The input image is a clone of the content image. It will be updated by the optimizer to match the target content and style.

🔁 10. Optimize the Input Image
The optimizer (LBFGS) iteratively adjusts the input image:
Computes losses using the model.
Combines content loss and style loss (weighted).
Backpropagates the error and updates the image.
Runs for 300 iterations.

🖼 11. Display Result
Finally, the stylized output image is shown using imshow.

🧾 Summary
This PyTorch script implements Neural Style Transfer using a pre-trained VGG19 model. It works by:
Extracting content and style features from two images.
Computing losses to measure how well the generated image matches those features.
Optimizing a copy of the content image to minimize style and content losses.
Producing a beautiful image with the content of one and the style of another.
This showcases the creative power of deep learning in art and design!
