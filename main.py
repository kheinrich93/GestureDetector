import src.utils as utils
import src.img_utils as img_utils
import dataloader


dir = utils.get_dirs()

'''
img_src = "img"
if img_src == "img":
    img_dir = os.path.join(dir["my_data_te"], 'test_ok.jpg')
    cap = img_utils.read_image(img_dir)
    cap = img_utils.resize_image_to_percent(cap, scale_percent=50)
else:
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        gray = img_utils.to_grayscale(frame)
'''

[images, labels] = dataloader.load_images(dir['asl_tr'])

# gesture classification
# DNN with ASL-dataset

# load training data set

# labels and images

# dataloader

# training


# software reaction
# spotify API
