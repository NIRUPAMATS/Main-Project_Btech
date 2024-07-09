import tensorflow as tf
import numpy as np
import os
import cv2
import time
from collections import Counter

def preprocess_input(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    # Setting default cam to webcam and necessary variables
    img = cv2.VideoCapture(0)
    exit = False

    # labelling data
    CODES = {
        0: "nothing"
    }
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(1, 27):
        CODES[i] = alpha[i - 1]

    CODES[27] = "del"
    CODES[28] = "space"

    model_path = os.path.join(r"C:\Users\Aiswarya\OneDrive\Desktop\S8\project\Last presentation", "best_model.tflite")
    print("Model Path:", os.path.abspath(model_path))
    
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize variables for word formation
    sequence = ''
    temp_sequence = []

    # Initialize the maximum text width and height
    max_text_width = 1100
    max_text_height = 200

    # Create a black screen for displaying formed words
    black_screen = np.zeros((max_text_height, max_text_width, 3), np.uint8)

    # Initialize the start time
    start_time = time.time()
    is_green = True

    while True:
        window_width = 1200
        window_height = 820
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', window_width, window_height)


        font = cv2.FONT_HERSHEY_SIMPLEX
        ret, frame = img.read()
        frame = cv2.flip(frame, 1)

        # Define frame to be used
        if is_green:
            frame = cv2.rectangle(frame, (60, 100), (310, 350), (0, 255, 0), 3)  # Green rectangle
        else:
            frame = cv2.rectangle(frame, (60, 100), (310, 350), (0, 0, 255), 3)  # Red rectangle
        frame2 = frame[100:350, 60:310]
        image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        image = preprocess_input(image)  # Add this line to preprocess the image

        # Set the input tensor for the TensorFlow Lite model
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        pred = interpreter.get_tensor(output_details[0]['index'])

        # predicting the letter
        move_code = CODES[np.argmax(pred[0])]

        ret, frame = img.read()

        frame = cv2.flip(frame, 1)
        frame = cv2.rectangle(frame, (60, 100), (310, 350), (0, 0, 255), 3)

        # displaying our prediction
        cv2.putText(frame, "Letter : {}".format(move_code), (63, 320),
                    font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        

        # Check if 3 seconds have elapsed
        elapsed_time = time.time() - start_time
        if elapsed_time < 3:
            if move_code == 'nothing':
                temp_sequence.append('*')
            elif move_code == 'space':
                temp_sequence.append(' ')
            elif move_code == 'del':
                temp_sequence.append('-')
            else:
                temp_sequence.append(move_code)

        if elapsed_time >= 3:
            # Calculate the most frequent alphabet within the last 3 seconds
            most_frequent = Counter(temp_sequence).most_common(1)
            if most_frequent:
                temp_sequence = []
                start_time = time.time()  # Reset the start time
                is_green = not is_green

                # Form words by concatenating predicted letters
                if most_frequent[0][0] != '*':
                    if most_frequent[0][0] == ' ':
                        sequence += ' '
                    elif most_frequent[0][0] == '-':
                        sequence = sequence[:-1]
                    else:
                        sequence += most_frequent[0][0]

     
            # Display the formed words on the black screen
            black_screen[:] = (0, 0, 0)  # Fill the black screen with black color
            cv2.putText(black_screen, '%s' % (sequence.upper()), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Word Formation', black_screen)

        if cv2.waitKey(1) & 0xff == ord('q'):
            exit = True
            break
        if exit:
            break
        cv2.imshow('Frame', frame)

    img.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()