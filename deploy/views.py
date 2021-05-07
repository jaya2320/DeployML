from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request,'index.html')

import math
import pandas as pd
import pickle
model=pickle.load(open('./model/breast_cancer.pkl','rb+'))

def predict(request):
     if request.method=='POST':
        
        temp={}
        temp['texture']=float(request.POST.get('texture'))  
        temp['radius']=float(request.POST.get('radius'))
        temp['perimeter']=float(request.POST.get('perimeter'))
        temp['smoothness']=float(request.POST.get('smoothness'))
        temp['area']=float(request.POST.get('area'))
        
        testdata=pd.DataFrame({'x':temp}).transpose()
        scoreval=model.predict(testdata)[0]
        if scoreval==0:
            ans="Benign"
        else:
            ans="Malignant"
        return render(request,'result.html',{'result':ans})