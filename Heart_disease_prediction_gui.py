#!/usr/bin/env python
# coding: utf-8

# # Gui Of The Model
# 
#    Author : Sourav Dey
# 

# In[46]:


#importing the necessary modules

from tkinter import *
from tkinter import messagebox
from PIL import ImageTk
import pickle


# In[47]:


#load the model and the scaling object created during model building

model=pickle.load(open('heart_disease_prediction_model.pkl','rb'))
sc=pickle.load(open('heart_disease_prediction_scaler.pkl','rb'))


# In[48]:


def predict():
    # if user not fill any entry 
    # then print "empty input" 
    if (age_field.get() == "" or
        sex_field.get() == "" or
        cp_field.get() == "" or
        trestbps_field.get() == "" or
        chol_field.get() == "" or
        thalach_field.get() == "" or
        exang_field.get() == "" or
        oldpeak_field.get() == "" or
        slope_field.get() == "" or
        ca_field.get() == "" or
        thal_field.get() == ""): 
              
        messagebox.showerror("Error","All fields are required.")
        
    else:
        data=[[]]
        
        # getting all the values in a list
        
        data[0].append(float(age_field.get()))
        data[0].append(float(sex_field.get()))
        data[0].append(float(cp_field.get()))
        data[0].append(float(trestbps_field.get()))
        data[0].append(float(chol_field.get()))
        data[0].append(float(thalach_field.get()))
        data[0].append(float(exang_field.get()))
        data[0].append(float(oldpeak_field.get()))
        data[0].append(float(slope_field.get()))
        data[0].append(float(ca_field.get()))
        data[0].append(float(thal_field.get()))
        
        # prediction of the data after scaling it
        
        prediction=model.predict(sc.transform(data))
        
        # This returns the prediction to the user
        
        if prediction[0]==0: # prediction === 0 belongs to Not diseased category
            messagebox.showinfo("Welcome","Feel Free You don't have any risk of having heart disease.")
        else:
            messagebox.showinfo("Welcome","You either have heart disease or have risk of getting disease.")


# In[49]:


# Driver code 
if __name__ == "__main__": 
      
    # create a GUI window 
    root = Tk() 
  
    # set the background colour of GUI window 
    bg = ImageTk.PhotoImage(file="Output_image.jpeg")
    bg_image = Label(root,image=bg).place(x=0,y=0,relwidth=1,relheight=1)
    
    # set the title of GUI window 
    root.title("Heart Disease Prediction") 
  
    # set the configuration of GUI window 
    root.geometry("1000x550+100+50") 
    root.resizable(FALSE,FALSE)
    
    Frame_input = Frame(root)
    Frame_input.place(x=500,y=100,height=340,width=400)
    
    # create a Form label 
    heading = Label(Frame_input, text="Enter The Details Below",font=("Impact",20,"bold"),fg="green").place(x=90,y=5)
    
    age = Label(Frame_input, text="Age :").place(x=30,y=50)
    sex = Label(Frame_input, text="Gender :").place(x=30,y=100)
    cp = Label(Frame_input, text="Chest Pain :").place(x=30,y=150)
    trestbps = Label(Frame_input, text="Resting Bps :").place(x=30,y=200)
    chol = Label(Frame_input, text="Cholestrol :").place(x=30,y=250)
    thalach = Label(Frame_input, text="Thalach :").place(x=30,y=300)
    exang = Label(Frame_input, text="Exang :").place(x=200,y=50)
    oldpeak = Label(Frame_input, text="Oldpeak :").place(x=200,y=100)
    slope = Label(Frame_input, text="Slope :").place(x=200,y=150)
    ca = Label(Frame_input, text="Ca :").place(x=200,y=200)
    thal = Label(Frame_input, text="Thal :").place(x=200,y=250)
    
    # Entry will make a field for taking inputs
    age_field = Entry(Frame_input,width=5,bg="lightgray")
    age_field.place(x=130,y=50)
    sex_field = Entry(Frame_input,width=5,bg="lightgray")
    sex_field.place(x=130,y=100)
    cp_field = Entry(Frame_input,width=5,bg="lightgray")
    cp_field.place(x=130,y=150)
    trestbps_field = Entry(Frame_input,width=5,bg="lightgray")
    trestbps_field.place(x=130,y=200)
    chol_field = Entry(Frame_input,width=5,bg="lightgray")
    chol_field.place(x=130,y=250)
    thalach_field = Entry(Frame_input,width=5,bg="lightgray")
    thalach_field.place(x=130,y=300)
    exang_field = Entry(Frame_input,width=5,bg="lightgray")
    exang_field.place(x=300,y=50)
    oldpeak_field = Entry(Frame_input,width=5,bg="lightgray")
    oldpeak_field.place(x=300,y=100)
    slope_field = Entry(Frame_input,width=5,bg="lightgray")
    slope_field.place(x=300,y=150)
    ca_field = Entry(Frame_input,width=5,bg="lightgray")
    ca_field.place(x=300,y=200)
    thal_field = Entry(Frame_input,width=5,bg="lightgray")
    thal_field.place(x=300,y=250)
    
    # creating button to predict
    submit = Button(Frame_input, text="Predict", fg="Black",bg="blue", command=predict,width=7) 
    submit.place(x=250, y=300)
    
    # start the GUI 
    root.mainloop() 

