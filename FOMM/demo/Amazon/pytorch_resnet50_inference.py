import torch, torcheia, torchvision
import PIL
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import time

def get_image(filename):
  im = Image.open(filename)
  # ImageNet pretrained models required input images to have width/height of 224
  # and color channels normalized according to ImageNet distribution.
  im_process = transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
  im = im_process(im) # 3 x 224 x 224
  return im.unsqueeze(0) # Add dimension to become 1 x 3 x 224 x 224

im = get_image('kitten.jpg')

# eval() toggles inference mode
#model = torchvision.models.resnet50(pretrained=True).eval()

# Convert model to torchscript
#model = torch.jit.script(model)

# Serialize model
#torch.jit.save(model, 'resnet50_traced.pt')

# Deserialize model
model = torch.jit.load('resnet50_traced.pt', map_location=torch.device('cpu'))

# Disable profiling executor. This is required for Elastic inference.
#torch._C._jit_set_profiling_executor(False)

# Attach Accelerator using device ordinal
#eia_model = torcheia.jit.attach_eia(model, 0)
eia_model = model
# Perform inference. Make sure to disable autograd.
with torch.no_grad():
    with torch.jit.optimized_execution(True):
        total_frames = 169
        pbar = tqdm(total=total_frames, desc=f"Elapsed time:0.000s")
        for i in range(total_frames):
            tic = time.time()
            probs = eia_model.forward(im)
            toc = time.time()
            pbar.desc = f"Elapsed time:{round(toc - tic, 3)}s"
            pbar.update(1)
        pbar.close()

# Torchvision implementation doesn't have Softmax as last layer.
# Use Softmax to convert activations to range 0-1 (probabilities)
probs = torch.nn.Softmax(dim=1)(probs)

# Get top 5 predicted classes
classes = eval(open('imagenet_classes.txt').read())
pred_probs, pred_indices = torch.topk(probs, 5)
pred_probs = pred_probs.squeeze().numpy()
pred_indices = pred_indices.squeeze().numpy()

for i in range(len(pred_indices)):
    curr_class = classes[pred_indices[i]]
    curr_prob = pred_probs[i]
    print('{}: {:.4f}'.format(curr_class, curr_prob))