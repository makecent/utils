import requests
import time
from torchvision.transforms import ToTensor
from PIL import Image
import io
import torch
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10000)
model.to(device)
model.load_state_dict(torch.load('trained_ori.pth'))

for i in range(10000):
    img_data = requests.get("https://hk.sz.gov.cn:8118/user/getVerify?0.6619971119181707").content
    image = Image.open(io.BytesIO(img_data))
    image.show()
    img = ToTensor()(image).to(device)
    with torch.no_grad():
        model.eval()
        output = model(img.unsqueeze(0)).max(1, keepdim=True)[1]
    print(output[0][0])
    time.sleep(1)