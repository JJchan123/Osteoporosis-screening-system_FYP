from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18
import cv2
import os
import pandas as pd
import os
from PIL import Image,ImageFilter
import numpy as np
from django.conf import settings
import base64
import io





# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)

num_classes = 1 

model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9 ,weight_decay=0.01)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) 

# Load the pre-trained weight
model.load_state_dict(torch.load("/Users/USER/OneDrive/桌面/MY_FYP/osteoporosis_detector/PTH_files/resnet18_best.pth"))
model = model.to(device)
model.eval()
target_layers = [model.layer4[-1]]

# Create your views here.
def index(request):
    http = "<h1> Osteoporosis Detector </h1>"
    return HttpResponse(http)


def upload_image(request):
    return render(request, 'upload_image_fin.html')


def predict(request):
    if request.method == 'POST' and request.FILES['image']:
        try:
            # Get the uploaded image from the request
            uploaded_image = request.FILES['image']
            
            # Preprocess the image (resize, normalize, etc.)
            img = Image.open(uploaded_image)
            img = img.resize((224, 224),Image.ANTIALIAS)
            img= img.convert('RGB')
            transform = transforms.Compose([transforms.ToTensor()])
            predict_image = transform(img)
            gradcam_image = transform(img)
            #==========================================================================(gradcam process)
            img_float_np = np.float32(img)/255
            input_tensor = gradcam_image.to(device)
            input_tensor = input_tensor.unsqueeze(0)
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            targets = None
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0,:]
            visualization = show_cam_on_image(img_float_np, grayscale_cam, use_rgb=True)
            #=============================================================================
            with torch.no_grad():
                model.eval()
                inputs = predict_image.unsqueeze(0).to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs).round().long()
    
            preds_numpy = preds.cpu().numpy()
            # Map the diagnosis to class index
            if preds_numpy[0] == 1 :
                prediction = "初步診斷為患有骨質疏鬆症 ， 建議到醫院做進一步的檢查!"
                gradcam = visualization
                gradcam_image = Image.fromarray(gradcam)
                buffer = io.BytesIO()
                gradcam_image.save(buffer, format="PNG")  # 保存为PNG格式
                gradcam_base64 = base64.b64encode(buffer.getvalue()).decode()
                return render(request, 'prediction.html', {'prediction': prediction ,'gradcam_base64': gradcam_base64,'show_image': True})
            elif preds_numpy[0] == 0:
                prediction = "初步結果尚未發現骨質疏鬆症的跡象!"
                return render(request, 'prediction.html', {'prediction': prediction,'noshow_image': True})
            #response_content = f"<h2>診斷結果:</h2> {prediction}"

        

        except Exception as e:
            return JsonResponse({'error': str(e)})
    else:
        return HttpResponse("<h3>Invalid request method or missing image file.</h3>", status=400)


def gradcam(request):
    if request.method == 'POST' and request.FILES['image']:
        try:
            # Get the uploaded image from the request
            uploaded_image = request.FILES['image']        
            # Preprocess the image (resize, normalize, etc.)
            img = Image.open(uploaded_image)
            image = img.resize((224, 224))
            image = image.convert('RGB')
            transform = transforms.Compose([transforms.ToTensor()])
            img_float_np = np.float32(img)/255
            input_tensor = transform(image)
            input_tensor = input_tensor.to(device)
            input_tensor = input_tensor.unsqueeze(0)
            
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            targets = None
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0,:]
            visualization = show_cam_on_image(img_float_np, grayscale_cam, use_rgb=True)
            
            return render(request, 'prediction.html', {'gradcam': visualization})
            save_path = os.path.join(settings.STATIC_ROOT, 'image', 'gradcam.png')
            plt.imsave(save_path, visualization)

        
        except Exception as e:
            return JsonResponse({'error': str(e)})
