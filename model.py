import tensorflow as tf

try:   
    # Load your existing model (replace with your actual model loading code)
    model = tf.keras.models.load_model('liveness.model') 
    # Save the model in .h5 format
    model.save('liveness_model.h5') 
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    
