
from django.shortcuts import render,redirect, HttpResponseRedirect
from django.contrib.auth.forms import AuthenticationForm,UserCreationForm
from django.contrib.auth import authenticate , login
from django.contrib.auth import authenticate, login as auth_login
 
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

   
def profile(request):
    return render(request,'profile.html')
