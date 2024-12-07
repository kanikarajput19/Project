
from django.shortcuts import render,redirect, HttpResponseRedirect
from django.contrib.auth.forms import AuthenticationForm,UserCreationForm
from django.contrib.auth import authenticate , login
from django.contrib.auth import authenticate, login as auth_login
from django.core.files.storage import FileSystemStorage
from .diagnostic_data import DIAGNOSTIC_STEPS

from joblib import load
import cv2
import numpy as np
 
import torch
# from transformers import ViTForImageClassification

from torchvision.transforms import transforms, ToTensor, Normalize, Resize


model = torch.load('C:/Users/harsh/OneDrive/Desktop/B14 projects/trained_vit_model.pth')
model.eval()  # Set the model to evaluation mode


 
# Create your views here.
def home(request):
    return render(request,'home.html')
 

def user_login(request):  # Renamed the function to avoid conflict
    if request.user.is_authenticated:
        return redirect('/profile')  # Redirect authenticated users to their profile

    if request.method == "POST":
        un = request.POST['username']
        pw = request.POST['password']
        
        user = authenticate(request, username=un, password=pw)
        if user is not None:
            auth_login(request, user)  # Use the correct login function to log in the user
            return redirect('/profile')  # Redirect to profile after successful login
        else:
            msg = 'Invalid Username/Password'
            form = AuthenticationForm(request.POST)
            return render(request, 'login.html', {'form': form, 'msg': msg})
    else:
        form = AuthenticationForm()  # Create a blank form for GET requests

    return render(request, 'login.html', {'form': form})  # Render the login form



def register(request):
    if request.user.is_authenticated:
        return redirect('/')

    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
              
            # Authenticate and log the user in
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)  # Log the user in after registration
                return redirect('/')  # Redirect to a desired page after login

    else:
        form = UserCreationForm()  # Create a blank form for GET requests

    return render(request, 'register.html', {'form': form})  # Render the registration form


# def profile(request):
#     if request.method == "POST" and request.FILES.get('uploadImage'):
#         img = request.FILES['uploadImage']
        
#         # Save the uploaded image to the media folder
#         fs = FileSystemStorage()
#         filename = fs.save(img.name, img)
        
#         # Generate the URL to access the uploaded image
#         img_url = fs.url(filename)
        
#         # Pass the image URL to the template
#         return render(request, 'profile.html', {'img_url': img_url})
#     else:
#         return render(request, 'profile.html')
    

# def profile(request):
#     if request.method == "POST" and request.FILES.get('uploadImage'):
#         img = request.FILES['uploadImage']
        
#         fs = FileSystemStorage()
#         filename = fs.save(img.name, img)
#         img_url = fs.url(filename)
#         img_path = fs.path(filename)
#         img_opencv = cv2.imread(img_path)

#         img_resize = cv2.resize(img_opencv, (600, 315))
#         cv2.imshow('without nor',img_resize)

#         img_nor = img_resize / 255.0
#         cv2.imshow('with nor',img_nor)
        
#         img_dim = np.expand_dims(img_nor, axis=0)
        
#         print(img_dim)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         return render(request, 'profile.html', {'img_url': img_url, 'img_dim': img_dim})
    
#     else:
#         return render(request, 'profile.html')




# def profile(request):
#     if request.method == "POST" and request.FILES.get('uploadImage'):
#         img = request.FILES['uploadImage']
        
        
#         fs = FileSystemStorage()
#         filename = fs.save(img.name, img)
#         img_url = fs.url(filename)
#         img_path = fs.path(filename)

        
#         img_opencv = cv2.imread(img_path)

        
#         img_resize = cv2.resize(img_opencv, (600, 315))
#         img_nor = img_resize / 255.0

    
#         gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (3, 3), 0)

        
#         _, threshold = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    
#         contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

#         disease_area = sum(cv2.contourArea(contour) for contour in contours)

    
#         img_with_contours = img_resize.copy()
#         cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)  # Green contours

    
#         processed_img_path = fs.path(f"processed_{filename}")
#         cv2.imwrite(processed_img_path, img_with_contours)
#         processed_img_url = fs.url(f"processed_{filename}")

    
#         img_dim = np.expand_dims(img_nor, axis=0)

#         # Pass the image URLs and calculated area to the template
#         return render(request, 'profile.html', {
#             'img_url': img_url,
#             'processed_img_url': processed_img_url,
#             'disease_area': disease_area,
#             'img_dim': img_dim
#         })
    
#     return render(request, 'profile.html')


from PIL import Image  # Add this import

def process_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Convert the OpenCV image (NumPy array) to a PIL image
    pil_img = Image.fromarray(img)

    # Define the transformations
    transform = transforms.Compose([
        Resize((224, 224)),  # Resize to match ViT input size
        ToTensor(),          # Convert to tensor
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
    ])

    # Apply the transformations to the PIL image
    img_tensor = transform(pil_img)
    return img_tensor.unsqueeze(0)  # Add batch dimension


def make_prediction(model, img_tensor):
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(img_tensor)  # Forward pass through the model
        prediction = torch.argmax(output.logits, dim=1).item()  # Get the class index
    return prediction




def profile(request):
    # if(request.method=="POST"):
    #     if(request.FILES.get('uploadImage')):
    #         img_name = request.FILES['uploadImage'].read()
    #         encode = base64.b64encode(img_name).decode('utf-8')
    #         img_url = f"data:image/jpeg;base64,{encode}"
    #         return render(request,'profile.html',{'img':img_url})
    # else:
    #     return render(request,'profile.html')
    if(request.method=="POST"):
        if(request.FILES.get('uploadImage')):
            img_name = request.FILES['uploadImage']
            # create a variable for our FileSystem package
            fs = FileSystemStorage()
            filename = fs.save(img_name.name,img_name)
            #urls
            img_url = fs.url(filename)
            #find the path of the image
            img_path = fs.path(filename)
 
            #start implementing the opencv condition
            img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            #resize the image for a constant use
            img = cv2.resize(img,(64,64))
            #flatten the image for the better clear shape of the disease spread on the skin
            img = img.flatten()
            #using the normalization predefined function to find the value
            img = np.expand_dims(img,axis=0)
 
            #we sill start executing with our model
            img_tensor = process_image(img_path)  # Preprocess uploaded image
            prediction = make_prediction(model, img_tensor)  # Predict with the model
 
            skin_disease_names = ['BA- cellulitis','BA-impetigo','FU-athlete-foot','FU-nail-fungus','FU-ringworm','PA-cutaneous-larva-migrans','VI-chickenpox','VI-shingles']
            # diagnosis = ['']
 
            result1 = skin_disease_names[prediction]
            # result2 = diagnosis[predict]
 
            # return render(request,'profile.html',{'img':img_url,'obj1':result1,'obj2':result2})
            # return render(request,'profile.html',{'img':img_url,'obj1':result1})
             # Retrieve diagnostic steps from the dictionary
            diagnostic_steps = DIAGNOSTIC_STEPS.get(result1, ["No diagnostic steps available."])

            # Pass disease name and diagnostic steps to the template
            return render(request, 'profile.html', {
                'img': img_url,
                'obj1': result1,
                'diagnostic_steps': diagnostic_steps
            })

    else:
        return render(request,'profile.html')



def diagnose_skin_disease(request):
    if request.method == "POST":
        uploaded_image = request.FILES['uploadImage']
        # Process the uploaded image and get the disease prediction
        predicted_disease = "FU-nail-fungus"  # Replace with your model's prediction logic
        
        # Fetch diagnostic steps for the predicted disease
        diagnostic_steps = DIAGNOSTIC_STEPS.get(predicted_disease, ["No diagnostic steps available."])
        
        return render(request, 'diagnostic_page.html', {
            'img_url': uploaded_image.url,
            'diagnostic_steps': diagnostic_steps,
        })

    return render(request, 'upload_page.html')




