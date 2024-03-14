import os
import cv2
import numpy as np
from imageai.Detection import ObjectDetection


# Название модели
MODEL_NAME = "../yolov3.pt"

# Название видеофайла, либо 0 для видеокамеры
# VIDEO_SOURCE = 0

# Парковочные места
parked_car_boxes = None

# Загружаем видеофайл, для которого хотим запустить распознавание
# video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# Сколько кадров подряд с пустым местом мы уже видели.
free_space_frames = 0

# Корневая директория
execution_path = os.getcwd()

# Загрузка обученной модели
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
# /Users/nelson/PycharmProjects/MEGASTREAM/streaming/stream/yolov3.pt
# detector.setModelPath(os.path.join(execution_path, MODEL_NAME))
detector.setModelPath("/Users/nelson/PycharmProjects/MEGASTREAM/streaming/stream/yolov3.pt")
detector.loadModel()

# Распознавание только машин
custom = detector.CustomObjects(car=True)


def get_crosses_boxes(parking_boxes: list, car_boxes: list) -> list:
    """
    Проверит все парковочные места и вернет словарь, где будут данные по
    всем парковчным местам
    :param parking_boxes: Список координатов парковочных мест
    :param car_boxes: Список координатов обнаруженных на кадре машин
    :return: Список показателей по заполненности по каждому парковочному места
    """
    output = []
    for parked_car_box in parking_boxes:
        crosses = []
        for car_box in car_boxes:
            crosses.append(iou(parked_car_box, car_box))
        output.append(crosses)
    return output


def iou(boxa, boxb):
    xA = max(boxa[0], boxb[0])
    yA = max(boxa[1], boxb[1])
    xB = min(boxa[2], boxb[2])
    yB = min(boxa[3], boxb[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxa[2] - boxa[0] + 1) * (boxa[3] - boxa[1] + 1)
    boxBArea = (boxb[2] - boxb[0] + 1) * (boxb[3] - boxb[1] + 1)
    output = interArea / float(boxAArea + boxBArea - interArea)
    return output


def get_car_boxes(capture: np.array) -> list:
    """
    Функци обнаруживает на кадре машины и возвращает
    координаты всех машин на фотографии
    :param capture: кадр видео
    :return: список координотов машин на фото
    """
    output = []
    returned_image, detections = detector.detectObjectsFromImage(
        input_image=capture,
        output_type="array",
        minimum_percentage_probability=30,
        custom_objects=custom
    )
    for detection in detections:
        output.append(detection["box_points"])

    return output


def detect(image):
    # Счетчик для пропуска кадров для ускорения
    global parked_car_boxes, free_space_frames
    cap = 0

    # if cap < 5:
    #     cap += 1
    #     continue
    # else:
    #     cap = 0

    if parked_car_boxes is None:
        # Программа на первом кадре запоминает парковочные места по обнаруженным машинам
        parked_car_boxes = get_car_boxes(image)
    else:
        # Если уже есть парковочные места, то запоминаем машины на кадре
        car_boxes = get_car_boxes(image)
        # Сравниваем пересечения машин с парковочными местами
        overlaps = get_crosses_boxes(parked_car_boxes, car_boxes)
        # Предполагаем, что свободных мест нет, пока не найдём хотя бы одно.
        free_space = False
        # Проходимся в цикле по каждому известному парковочному месту.
        for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):
            # Ищем максимальное значение пересечения с любой обнаруженной
            # на кадре машиной (неважно, какой).
            max_IoU_overlap = np.max(overlap_areas)
            # Получаем верхнюю левую и нижнюю правую координаты парковочного места.
            x1, y1, x2, y2 = parking_area
            # Проверяем, свободно ли место, проверив значение IoU.
            if max_IoU_overlap < 0.15:
                # Место свободно! Рисуем зелёную рамку вокруг него.
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Отмечаем, что мы нашли как минимум оно свободное место.
                free_space = True
            else:
                # Место всё ещё занято — рисуем красную рамку.
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # Записываем значение IoU внутри рамки.
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))
        # Если хотя бы одно место было свободным, начинаем считать кадры.
        if free_space:
            free_space_frames += 1
        else:
            # Если всё занято, обнуляем счётчик.
            free_space_frames = 0
        # Показываем кадр на экране.
    return image
