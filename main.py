import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load your model
model = load_model('model_name.h5')  # Update with your model's path

# Function to make predictions
def model_prediction(test_image):
    # Preprocess the image
    img = image.load_img(test_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image

    # Make predictions
    predictions = model.predict(img_array)
    
    # Return the index of the predicted class
    return np.argmax(predictions, axis=1)[0]

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Detection"])

#Main Page
if(app_mode=="Home"):
    st.header("CHICKEN DISEASE DETECTION SYSTEM")
    image_path = "home.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Chicken Disease Detection System! 🐓🐔🔍
    
    Our mission is to help in identifying Chicken diseases efficiently. Upload an image of a Chicken, and our system will analyze it to detect any signs of diseases. Together, let's protect our Chicken Farms and ensure a healthier Production!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Detection** page and upload an image of a suspected Chicken.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Detection** page in the sidebar to upload an image and experience the power of our Chicken Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is Collected from Various farm from different locations.
                we did data augmentation for increase the number of unhealthy chicken images.We trained our data in multiple models. Than,We choose densenet201 model for our capstone 

                #### Content
                1. train  images
                2. test images
                3. validation  images

                """)

# Prediction Page
if app_mode == "Disease Detection":
    st.header("Disease Detection")
    test_image = st.file_uploader("Choose an Image:", type=['jpg', 'png', 'jpeg'])
    
    if test_image is not None:
        # Display the uploaded image
        st.image(test_image, width=300)

        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)

            # Reading Labels
            class_names = ['Healthy_Chicken', 'UnHealthy_Chicken']
            st.success(f"Model is predicting it's a {class_names[result_index]}")

          
# #Prediction Page
# if app_mode == "Disease Recognition":
#     st.header("Disease Recognition")
#     test_image = st.file_uploader("Choose an Image:", type=['jpg', 'png', 'jpeg'])
    
#     if test_image is not None:
#         # Display the uploaded image
#         st.image(test_image, width=300)
        
#         # Preprocess the image
#         img = image.load_img(test_image, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         img_array /= 255.0  # Rescale the image

#         # Make predictions
#         predictions = model.predict(img_array)

#         # Debugging output
#         st.write("Predictions array:", predictions)
#         st.write("Predictions shape:", predictions.shape)

#         # Ensure the predictions are correct
#         if predictions.ndim != 2 or predictions.shape[1] != 2:
#             st.error("Unexpected prediction output shape.")
#         else:
#             predicted_class = np.argmax(predictions, axis=1)
#             st.write("Predicted class index:", predicted_class)

#             # Check if the predicted_class is valid
#             if len(predicted_class) == 0:
#                 st.error("No predicted classes found.")
#             else:
#                 class_names = ['Healthy_Chicken', 'UnHealthy_Chicken']
                
#                 if predicted_class[0] < len(class_names):
#                     st.write(f"Predicted Class: {class_names[predicted_class[0]]}")
#                 else:
#                     st.error("Predicted class index is out of range.")

#             # Display prediction probabilities
#             st.write("Prediction probabilities:")
#             for idx, class_name in enumerate(class_names):
#                 st.write(f"{class_name}: {predictions[0][idx]:.4f}")
#     # #Predict butt
#     # if(st.button("Predict")):
#     #     st.snow()
#     #     st.write("Our Prediction")
#     #     result_index = model_prediction(test_image)
#     #     #Reading Labels
#     #     class_name = ['Healthy_Chicken', 'UnHealthy_Chicken', ]
#     #     st.success("Model is Predicting it's a {}".format(class_name[result_index]))
