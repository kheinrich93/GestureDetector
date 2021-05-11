import os.path
import numpy as np
import cv2


def read_image(image):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    # img = np.expand_dims(img, axis=-1)
    return img

if __name__ == '__main__':
    # Set relative directories
    project_dir = os.path.split(os.getcwd())[0]
    res_dir = os.path.join(project_dir, 'res')

    print ("test")
    img_src = "img"
    if img_src == "img":
        img_dir = os.path.join(res_dir, 'test_ok.jpg')
        cap = read_image(img_dir)
    else:
        cap = cv2.VideoCapture(0)
        # Display the resulting frame or image
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # cv2.imshow('frame', gray)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

    # semantic hand segmentation

    # gesture classification

    # software reaction



    print("test")


