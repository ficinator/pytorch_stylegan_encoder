from typing import Union, List, Tuple, Iterator, Iterable

import dlib
import numpy as np
import scipy
from PIL import Image

# PREDICTOR_PATH = 'shape_predictor_5_face_landmarks.dat'
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
IMAGE_SIZE = 256
PADDING = .5

FaceLandmarks = List[Tuple[int, int]]


class FaceAlignment:
    """ Face detection and alignment.

    Inspired by
    * https://github.com/davisking/dlib/blob/master/python_examples/face_alignment.py.
    * https://github.com/Puzer/stylegan-encoder/blob/master/align_images.py

    Download the models from
    * http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
    * http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

    Extract them with bzip2 -d <model_path>.
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

    @property
    def image_shape(self) -> Tuple[int, int]:
        return self.image_size, self.image_size

    def detect_faces(self, image: np.ndarray):
        return self.detector(image, 1)

    def _get_faces_landmarks(
            self,
            image: np.ndarray,
            detections: Union[list, None] = None,
    ) -> Iterable[FaceLandmarks]:
        detections = detections or self.detect_faces(image)
        # faces = dlib.full_object_detections()
        for detection in detections:
            yield [(p.x, p.y) for p in self.shape_predictor(image, detection).parts()]

    # def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True):
    def _align_face(
            self,
            image: np.ndarray,
            face_landmarks: FaceLandmarks,
            enable_padding: bool = True,
    ) -> np.ndarray:
        lm = np.asarray(face_landmarks)
        lm_eye_left = lm[36:42]  # left-clockwise
        lm_eye_right = lm[42:48]  # left-clockwise
        lm_mouth_outer = lm[48:60]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        image = Image.fromarray(image)

        # Shrink.
        shrink = int(np.floor(qsize / self.image_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(image.size[0]) / shrink)), int(np.rint(float(image.size[1]) / shrink)))
            image = image.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, image.size[0]),
                min(crop[3] + border, image.size[1]))
        if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
            image = image.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - image.size[0] + border, 0),
               max(pad[3] - image.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            image = np.pad(np.float32(image), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = image.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(mask * 3.0 + 1.0, 0.0,
                                                                                               1.0)
            image += (np.median(image, axis=(0, 1)) - image) * np.clip(mask, 0.0, 1.0)
            image = Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        image = image.transform(self.image_shape, Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)

        return np.asarray(image)

    def get_aligned_face_images(self, image: np.ndarray, detections: Union[list, None] = None) -> Iterable[np.ndarray]:
        for face_landmarks in self._get_faces_landmarks(image, detections):
            yield self._align_face(image, face_landmarks)

    def get_aligned_face_image(self, image: np.ndarray, detections: Union[list, None] = None) -> np.ndarray:
        faces_landmarks = self._get_faces_landmarks(image, detections)
        try:
            return self._align_face(image, next(iter(faces_landmarks)))
        except StopIteration:
            raise Exception(f'No face detected')
