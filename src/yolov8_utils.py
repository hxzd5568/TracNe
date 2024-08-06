import numpy as np
from PIL import Image


def get_input(path=None):
    # path = '/home/user/tvm/models/ils2012/ILSVRC2012_val_00049674.JPEG'
    if path is None:
        path = "/home/user/tvm/models/ils2012/49/ILSVRC2012_val_00041346.jpeg"
    img = Image.open(path)
    # save original image size for future
    img_width, img_height = img.size
    # convert image to RGB,
    img = img.convert("RGB")
    # resize to 640x640
    img = img.resize((640, 640))
    # convert the image to tensor
    # of [1,3,640,640] as required for
    # the model input
    input = np.array(img)
    input = input.transpose(2, 0, 1)
    input = input.reshape(1, 3, 640, 640).astype("uint8")
    return input


def get_yolo_result(outputs: [np.array, np.array]):
    output0 = outputs[0]
    output1 = outputs[1]

    # !!! control batch element
    output0 = output0[0].transpose()
    output1 = output1[0]

    boxes = output0[:, 0:84]
    masks = output0[:, 84:]

    output1 = output1.reshape(32, 160 * 160)

    masks = masks @ output1

    boxes = np.hstack((boxes, masks))

    yolo_classes = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def intersection(box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        return (x2 - x1) * (y2 - y1)

    def union(box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        return box1_area + box2_area - intersection(box1, box2)

    def iou(box1, box2):
        return intersection(box1, box2) / union(box1, box2)

    def get_mask(row, box):
        mask = row.reshape(160, 160)
        return mask

    # parse and filter detected objects
    objects = []
    img_width = 640
    img_height = 640
    for row in boxes:
        prob = row[4:84].max()
        if prob < 0.5:
            continue
        xc, yc, w, h = row[:4]
        class_id = row[4:84].argmax()
        x1 = (xc - w / 2) / 640 * img_width
        y1 = (yc - h / 2) / 640 * img_height
        x2 = (xc + w / 2) / 640 * img_width
        y2 = (yc + h / 2) / 640 * img_height
        label = yolo_classes[class_id]
        mask = get_mask(row[84:25684], (x1, y1, x2, y2))
        objects.append([x1, y1, x2, y2, label, prob, mask])

    # apply non-maximum suppression to filter duplicated
    # boxes
    objects.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(objects) > 0:
        result.append(objects[0])
        objects = [object for object in objects if iou(object, objects[0]) < 0.7]

    return result
