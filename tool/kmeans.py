import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans_kernel(boxes, k):
    np.random.seed()
    kpoint = list(boxes[np.random.choice(boxes.shape[0], 1)])
    dests = np.zeros(boxes.shape[0])
    for kit in range(k-1):
        for i, box in enumerate(boxes):
            ious = iou(box, np.array(kpoint))
            dests[i] = ious.max()
        kpoint.append(boxes[dests.argmin()])

    return np.array(kpoint)

def kmeans(boxes, k):
    kpoint = kmeans_kernel(boxes, k)

    row = boxes.shape[0]
    last_dest = np.zeros(row)
    dests = np.zeros(row)

    while True:
        for i, box in enumerate(boxes):
            ious = iou(box, kpoint)
            dests[i] = ious.argmax()

        dests = np.array(dests)
        if (dests == last_dest).all():
            break

        last_dest = dests.copy()
        for i, wh in enumerate(kpoint):
            nk = np.median(boxes[dests == i], 0)
            kpoint[i] = nk

    return kpoint
