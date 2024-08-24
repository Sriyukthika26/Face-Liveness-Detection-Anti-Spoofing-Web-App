import tensorflow as tf
import tf2onnx

try:   
    # Loading existing model
    model = tf.keras.models.load_model('liveness.model') 
    # Save the model in .h5 format
    model.save('liveness_model.h5') 
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")


# Convert the model to ONNX
spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
output_path = "liveness_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
    
