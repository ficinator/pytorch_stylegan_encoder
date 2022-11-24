from typing import Union, List

import dlib
import numpy as np


PREDICTOR_PATH = 'shape_predictor_5_face_landmarks.dat'
IMAGE_SIZE = 256
PADDING = .5


class FaceAlignment:
    """ Face detection and alignment.

    Inspired by https://github.com/davisking/dlib/blob/master/python_examples/face_alignment.py.

    Download the model from http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 and extract with bzip2 -d.
    """

    def __init__(
        self,
        predictor_path: str = PREDICTOR_PATH,
        image_size: int = IMAGE_SIZE,
        padding: float = PADDING,
    ) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_path)
        self.image_size: int = image_size
        self.padding: float = padding

    def detect_faces(self, image: np.ndarray):
        return self.detector(image, 1)

    def _get_full_object_detections(self, image: np.ndarray, detections: Union[list, None] = None) -> list:
        detections = detections or self.detect_faces(image)
        faces = dlib.full_object_detections()
        for detection in detections:
            faces.append(self.shape_predictor(image, detection))
        return faces

    def get_aligned_face_images(self, image: np.ndarray, detections: Union[list, None] = None) -> List[np.ndarray]:
        faces = self._get_full_object_detections(image, detections)
        return dlib.get_face_chips(image, faces, size=self.image_size, padding=self.padding)

    def get_aligned_face_image(self, image: np.ndarray, detections: Union[list, None] = None) -> np.ndarray:
        faces = self._get_full_object_detections(image, detections)
        n_faces: int = len(faces)
        if n_faces != 1:
            raise Exception(f'Exactly one face expected, detected {n_faces}')

        return dlib.get_face_chip(image, faces[0], size=self.image_size, padding=self.padding)
