import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum
import json
import time
import os

pyautogui.FAILSAFE = False

class DetectionMode(Enum):
    MEDIAPIPE_ONLY = "mediapipe"
    ARUCO_ONLY = "aruco"
    HYBRID = "hybrid"

@dataclass
class Config:
    """Класс конфигурации приложения"""
    # Общие настройки
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    fps: int = 30
    
    # Настройки детекции
    detection_mode: DetectionMode = DetectionMode.ARUCO_ONLY
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 0  # 0 - легкая модель
    
    # Настройки производительности
    skip_frames: int = 2  # Обрабатывать каждый N-й кадр для MediaPipe
    
    # Настройки ArUco 4x4_50
    aruco_marker_size_mm: int = 50  # Размер маркера в миллиметрах для печати
    aruco_detection_threshold: float = 0.1  # Порог детекции
    
    # Настройки управления курсором
    cursor_smoothing: float = 0.3
    cursor_sensitivity: float = 1.5
    
    # Настройки отображения
    show_fps: bool = True
    show_skeleton: bool = True
    show_aruco_info: bool = True
    
    def save(self, filename: str = "config.json"):
        data = {k: v.value if isinstance(v, Enum) else v 
                for k, v in self.__dict__.items()}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    @classmethod
    def load(cls, filename: str = "config.json"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            config = cls()
            for k, v in data.items():
                if k == 'detection_mode':
                    setattr(config, k, DetectionMode(v))
                else:
                    setattr(config, k, v)
            return config
        except FileNotFoundError:
            return cls()

class BodyLandmarks:
    """Класс для хранения всех меток тела"""
    def __init__(self):
        self.landmarks = {
            'head': None,
            'left_shoulder': None,
            'right_shoulder': None,
            'pelvis': None,
            'left_elbow': None,
            'left_hand': None,
            'right_elbow': None,
            'right_hand': None,
            'left_knee': None,
            'left_foot': None,
            'right_knee': None,
            'right_foot': None
        }
        
        # Соответствие ArUco 4x4_50 маркеров (ID от 0 до 11)
        self.aruco_mapping = {
            0: 'head',
            1: 'left_shoulder',
            2: 'right_shoulder',
            3: 'pelvis',
            4: 'left_elbow',
            5: 'left_hand',
            6: 'right_elbow',
            7: 'right_hand',
            8: 'left_knee',
            9: 'left_foot',
            10: 'right_knee',
            11: 'right_foot'
        }

class ArUco4x4Detector:
    """Специализированный детектор для ArUco 4x4_50"""
    def __init__(self, config: Config):
        self.config = config
        
        # Используем именно DICT_4X4_50
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Оптимизированные параметры для 4x4_50
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Настройки для лучшего распознавания 4x4 маркеров
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant = 7
        
        # Настройки для маленьких маркеров 4x4
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        self.aruco_params.minCornerDistanceRate = 0.05
        
        # Улучшение детекции углов
        self.aruco_params.minDistanceToBorder = 3
        self.aruco_params.minMarkerDistanceRate = 0.05
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1
        
        # Специфично для 4x4
        self.aruco_params.markerBorderBits = 1
        self.aruco_params.perspectiveRemovePixelPerCell = 4  # Меньше для 4x4
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        
        # Обработка ошибок
        self.aruco_params.maxErroneousBitsInBorderRate = 0.35
        self.aruco_params.minOtsuStdDev = 5.0
        self.aruco_params.errorCorrectionRate = 0.6
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Для отслеживания стабильности маркеров
        self.marker_history = {}
        self.history_size = 5
    
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """Предобработка изображения для улучшения детекции"""
        # Конвертируем в градации серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Применяем размытие для уменьшения шума
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Улучшаем контраст
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        return gray
    
    def detect(self, frame: np.ndarray) -> Optional[BodyLandmarks]:
        """Детекция ArUco 4x4_50 маркеров"""
        # Предобработка
        gray = self.preprocess_image(frame)
        
        # Детекция маркеров
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            body = BodyLandmarks()
            
            for i, marker_id in enumerate(ids.flatten()):
                if 0 <= marker_id <= 11:  # Проверяем, что ID в нужном диапазоне
                    # Вычисляем центр маркера с субпиксельной точностью
                    corner = corners[i][0]
                    center_x = float(np.mean(corner[:, 0]))
                    center_y = float(np.mean(corner[:, 1]))
                    
                    # Стабилизация позиции
                    if marker_id not in self.marker_history:
                        self.marker_history[marker_id] = []
                    
                    self.marker_history[marker_id].append((center_x, center_y))
                    if len(self.marker_history[marker_id]) > self.history_size:
                        self.marker_history[marker_id].pop(0)
                    
                    # Усредняем позицию для стабильности
                    avg_x = np.mean([p[0] for p in self.marker_history[marker_id]])
                    avg_y = np.mean([p[1] for p in self.marker_history[marker_id]])
                    
                    body_part = self.aruco_mapping.get(marker_id)
                    if body_part:
                        body.landmarks[body_part] = (int(avg_x), int(avg_y), 1.0)
            
            return body
        
        return None
    
    def draw(self, frame: np.ndarray, corners, ids):
        """Отрисовка обнаруженных маркеров"""
        if ids is not None:
            # Рисуем контуры маркеров
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            if self.config.show_aruco_info:
                for i, marker_id in enumerate(ids.flatten()):
                    if 0 <= marker_id <= 11:
                        corner = corners[i][0]
                        center_x = int(np.mean(corner[:, 0]))
                        center_y = int(np.mean(corner[:, 1]))
                        
                        # Получаем название части тела
                        body_part = self.aruco_mapping.get(marker_id, "Unknown")
                        
                        # Рисуем информацию
                        cv2.putText(frame, f"{body_part}", 
                                  (center_x - 30, center_y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def generate_markers(self, output_dir: str = "aruco_4x4_50_markers"):
        """Генерация ArUco 4x4_50 маркеров для печати"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Размер маркера в пикселях (для печати с разрешением 300 DPI)
        dpi = 300
        marker_size_px = int(self.config.aruco_marker_size_mm / 25.4 * dpi)
        
        print(f"Генерация маркеров размером {self.config.aruco_marker_size_mm}мм ({marker_size_px}px при {dpi}DPI)")
        
        for marker_id in range(12):  # ID от 0 до 11
            # Генерируем маркер
            marker_img = cv2.aruco.generateImageMarker(
                self.aruco_dict, marker_id, marker_size_px
            )
            
            # Добавляем белую рамку (20% от размера маркера)
            border_size = int(marker_size_px * 0.2)
            marker_with_border = cv2.copyMakeBorder(
                marker_img, 
                border_size, border_size * 2,  # Больше места снизу для текста
                border_size, border_size,
                cv2.BORDER_CONSTANT, value=255
            )
            
            # Добавляем информацию
            body_part = self.aruco_mapping.get(marker_id, "Unknown")
            text = f"ID: {marker_id} - {body_part}"
            font_scale = marker_size_px / 200
            thickness = max(1, int(marker_size_px / 100))
            
            # Позиция текста
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (marker_with_border.shape[1] - text_size[0]) // 2
            text_y = marker_with_border.shape[0] - border_size // 2
            
            cv2.putText(marker_with_border, text,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness)
            
            # Сохраняем
            filename = os.path.join(output_dir, f"aruco_4x4_50_id{marker_id:02d}_{body_part}.png")
            cv2.imwrite(filename, marker_with_border)
            print(f"✓ Маркер {marker_id} ({body_part}) сохранен")
        
        # Создаем общий лист для печати
        self.create_marker_sheet(output_dir)
    
    def create_marker_sheet(self, output_dir: str):
        """Создание листа A4 с маркерами для печати"""
        # A4 размеры при 300 DPI
        dpi = 300
        a4_width_px = int(210 / 25.4 * dpi)  # 210mm
        a4_height_px = int(297 / 25.4 * dpi)  # 297mm
        
        # Создаем белый лист
        sheet = np.ones((a4_height_px, a4_width_px), dtype=np.uint8) * 255
        
        # Размер маркера с рамкой
        marker_size_px = int(self.config.aruco_marker_size_mm / 25.4 * dpi)
        border_size = int(marker_size_px * 0.2)
        total_size = marker_size_px + 2 * border_size
        
        # Расположение маркеров на листе
        margin = int(20 / 25.4 * dpi)  # 20mm margin
        x_spacing = int(10 / 25.4 * dpi)  # 10mm spacing
        y_spacing = int(15 / 25.4 * dpi)  # 15mm spacing
        
        markers_per_row = (a4_width_px - 2 * margin) // (total_size + x_spacing)
        
        for i in range(12):
            row = i // markers_per_row
            col = i % markers_per_row
            
            x = margin + col * (total_size + x_spacing)
            y = margin + row * (total_size + y_spacing + int(10 / 25.4 * dpi))  # Extra space for text
            
            if y + total_size > a4_height_px - margin:
                break  # Не помещается на лист
            
            # Генерируем маркер
            marker_img = cv2.aruco.generateImageMarker(
                self.aruco_dict, i, marker_size_px
            )
            
            # Вставляем маркер на лист
            sheet[y:y+marker_size_px, x:x+marker_size_px] = marker_img
            
            # Добавляем рамку
            cv2.rectangle(sheet, 
                         (x - border_size, y - border_size),
                         (x + marker_size_px + border_size, y + marker_size_px + border_size),
                         0, 2)
            
            # Добавляем подпись
            body_part = self.aruco_mapping.get(i, "Unknown")
            text = f"ID:{i} - {body_part}"
            cv2.putText(sheet, text,
                       (x, y + marker_size_px + border_size + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        
        # Сохраняем лист
        sheet_filename = os.path.join(output_dir, "aruco_4x4_50_print_sheet.png")
        cv2.imwrite(sheet_filename, sheet)
        print(f"\n✓ Лист для печати сохранен: {sheet_filename}")
        print(f"  Размер листа: A4 ({dpi}DPI)")
        print(f"  Размер маркеров: {self.config.aruco_marker_size_mm}мм")

class LightweightMediaPipeDetector:
    """Облегченный детектор MediaPipe"""
    def __init__(self, config: Config):
        self.config = config
        self.mp_pose = mp.solutions.pose
        
        # Минимальные настройки для производительности
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Самая легкая модель
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
        
        self.frame_counter = 0
        self.last_body = None
        
        # Только необходимые точки MediaPipe
        self.mediapipe_mapping = {
            0: 'head',  # nose
            11: 'left_shoulder',
            12: 'right_shoulder',
            23: 'pelvis',  # left hip
            13: 'left_elbow',
            15: 'left_hand',  # left wrist
            14: 'right_elbow',
            16: 'right_hand',  # right wrist
            25: 'left_knee',
            27: 'left_foot',  # left ankle
            26: 'right_knee',
            28: 'right_foot'  # right ankle
        }
    
    def detect(self, frame: np.ndarray) -> Optional[BodyLandmarks]:
        """Детекция с пропуском кадров"""
        self.frame_counter += 1
        
        # Пропускаем кадры для экономии ресурсов
        if self.frame_counter % self.config.skip_frames != 0:
            return self.last_body
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            body = BodyLandmarks()
            h, w = frame.shape[:2]
            
            for mp_idx, body_part in self.mediapipe_mapping.items():
                if mp_idx < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[mp_idx]
                    if landmark.visibility > 0.5:  # Только видимые точки
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        body.landmarks[body_part] = (x, y, landmark.visibility)
            
            self.last_body = body
            return body
        
        return self.last_body

class SimpleCursorController:
    """Простой контроллер курсора"""
    def __init__(self, config: Config):
        self.config = config
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothed_x = self.screen_width // 2
        self.smoothed_y = self.screen_height // 2
        self.active_hand = None
    
    def update(self, body: BodyLandmarks, frame_shape: Tuple[int, int]):
        """Обновление позиции курсора"""
        if not body:
            return
        
        h, w = frame_shape[:2]
        
        # Определяем активную руку
        hand_position = None
        if body.landmarks.get('right_hand'):
            hand_position = body.landmarks['right_hand']
            self.active_hand = 'right'
        elif body.landmarks.get('left_hand'):
            hand_position = body.landmarks['left_hand']
            self.active_hand = 'left'
        
        if not hand_position:
            return
        
        # Преобразуем координаты
        target_x = int((hand_position[0] / w) * self.screen_width)
        target_y = int((hand_position[1] / h) * self.screen_height)
        
        # Применяем сглаживание
        self.smoothed_x += (target_x - self.smoothed_x) * self.config.cursor_smoothing
        self.smoothed_y += (target_y - self.smoothed_y) * self.config.cursor_smoothing
        
        # Перемещаем курсор
        try:
            pyautogui.moveTo(int(self.smoothed_x), int(self.smoothed_y), duration=0)
        except:
            pass

class MotionCaptureApp:
    """Основное приложение"""
    def __init__(self, config: Config):
        self.config = config
        self.aruco_detector = ArUco4x4Detector(config)
        self.mediapipe_detector = LightweightMediaPipeDetector(config)
        self.cursor_controller = SimpleCursorController(config)
        
        self.cap = None
        self.running = False
        self.fps_history = []
        self.last_time = time.time()
    
    def start(self):
        """Запуск приложения"""
        # Генерируем маркеры при первом запуске
        if not os.path.exists("aruco_4x4_50_markers"):
            print("Генерация ArUco 4x4_50 маркеров...")
            self.aruco_detector.generate_markers()
        
        self.cap = cv2.VideoCapture(self.config.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Попробуем установить формат для лучшей производительности
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        if not self.cap.isOpened():
            print("Ошибка: не удалось открыть камеру")
            return
        
        self.running = True
        self.run()
    
    def calculate_fps(self):
        """Расчет FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
    
    def run(self):
        """Основной цикл"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Детекция в зависимости от режима
            body = None
            detected_corners = None
            detected_ids = None
            
            if self.config.detection_mode == DetectionMode.ARUCO_ONLY:
                # Только ArUco
                gray = self.aruco_detector.preprocess_image(frame)
                detected_corners, detected_ids, _ = self.aruco_detector.detector.detectMarkers(gray)
                body = self.aruco_detector.detect(frame)
                
            elif self.config.detection_mode == DetectionMode.MEDIAPIPE_ONLY:
                # Только MediaPipe
                body = self.mediapipe_detector.detect(frame)
                
            elif self.config.detection_mode == DetectionMode.HYBRID:
                # Гибридный режим
                gray = self.aruco_detector.preprocess_image(frame)
                detected_corners, detected_ids, _ = self.aruco_detector.detector.detectMarkers(gray)
                aruco_body = self.aruco_detector.detect(frame)
                
                # MediaPipe только если не все маркеры найдены
                if not aruco_body or len([v for v in aruco_body.landmarks.values() if v]) < 6:
                    mp_body = self.mediapipe_detector.detect(frame)
                    
                    if aruco_body and mp_body:
                        # Объединяем результаты
                        body = aruco_body
                        for part, coords in mp_body.landmarks.items():
                            if not body.landmarks.get(part) and coords:
                                body.landmarks[part] = coords
                    elif mp_body:
                        body = mp_body
                else:
                    body = aruco_body
            
            # Отрисовка
            if detected_corners is not None and detected_ids is not None:
                self.aruco_detector.draw(frame, detected_corners, detected_ids)
            
            if body and self.config.show_skeleton:
                self.draw_skeleton(frame, body)
            
            # Управление курсором
            if body:
                self.cursor_controller.update(body, frame.shape)
            
            # Отображение информации
            self.draw_info(frame, body)
            
            cv2.imshow('Motion Capture - ArUco 4x4_50', frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                modes = list(DetectionMode)
                current_idx = modes.index(self.config.detection_mode)
                self.config.detection_mode = modes[(current_idx + 1) % len(modes)]
                print(f"Режим: {self.config.detection_mode.value}")
            elif key == ord('g'):
                print("Генерация маркеров...")
                self.aruco_detector.generate_markers()
            elif key == ord('s'):
                self.config.save()
                print("Настройки сохранены")
            elif key == ord('t'):
                # Тестовый режим - показать все возможные маркеры
                self.test_aruco_detection(frame)
        
        self.stop()
    
    def draw_skeleton(self, frame: np.ndarray, body: BodyLandmarks):
        """Отрисовка скелета"""
        connections = [
            ('head', 'left_shoulder'),
            ('head', 'right_shoulder'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_hand'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_hand'),
            ('left_shoulder', 'pelvis'),
            ('right_shoulder', 'pelvis'),
            ('pelvis', 'left_knee'),
            ('left_knee', 'left_foot'),
            ('pelvis', 'right_knee'),
            ('right_knee', 'right_foot')
        ]
        
        # Рисуем соединения
        for start, end in connections:
            if body.landmarks.get(start) and body.landmarks.get(end):
                cv2.line(frame, 
                        body.landmarks[start][:2], 
                        body.landmarks[end][:2], 
                        (0, 255, 0), 2)
        
        # Рисуем точки
        for name, point in body.landmarks.items():
            if point:
                cv2.circle(frame, point[:2], 5, (0, 0, 255), -1)
    
    def draw_info(self, frame: np.ndarray, body: BodyLandmarks):
        """Отображение информации"""
        # FPS
        if self.config.show_fps:
            fps = self.calculate_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Режим
        cv2.putText(frame, f"Mode: {self.config.detection_mode.value}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Количество обнаруженных точек
        if body:
            detected_count = sum(1 for v in body.landmarks.values() if v)
            cv2.putText(frame, f"Detected: {detected_count}/12", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Активная рука
        if self.cursor_controller.active_hand:
            cv2.putText(frame, f"Hand: {self.cursor_controller.active_hand}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Инструкции
        instructions = "Q-quit | M-mode | G-generate | S-save | T-test"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def test_aruco_detection(self, frame):
        """Тестовый режим для проверки всех типов ArUco"""
        test_window = "ArUco Detection Test"
        
        # Тестируем разные словари
        test_dicts = [
            (cv2.aruco.DICT_4X4_50, "4X4_50"),
            (cv2.aruco.DICT_4X4_100, "4X4_100"),
            (cv2.aruco.DICT_5X5_50, "5X5_50"),
            (cv2.aruco.DICT_6X6_50, "6X6_50")
        ]
        
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for dict_type, dict_name in test_dicts:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            detector = cv2.aruco.ArucoDetector(aruco_dict, self.aruco_detector.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None:
                results.append(f"{dict_name}: {len(ids)} markers found (IDs: {ids.flatten().tolist()})")
            else:
                results.append(f"{dict_name}: No markers found")
        
        # Показываем результаты
        test_frame = np.ones((300, 600, 3), dtype=np.uint8) * 255
        for i, result in enumerate(results):
            cv2.putText(test_frame, result, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.imshow(test_window, test_frame)
        cv2.waitKey(2000)
        cv2.destroyWindow(test_window)
    
    def stop(self):
        """Остановка приложения"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Точка входа"""
    config = Config.load()
    app = MotionCaptureApp(config)
    
    print("\n=== Motion Capture с ArUco 4x4_50 ===")
    print("\nМаркеры ArUco 4x4_50:")
    print("  ID 0  - Голова")
    print("  ID 1  - Левое плечо")
    print("  ID 2  - Правое плечо")
    print("  ID 3  - Таз")
    print("  ID 4  - Левый локоть")
    print("  ID 5  - Левая ладонь")
    print("  ID 6  - Правый локоть")
    print("  ID 7  - Правая ладонь")
    print("  ID 8  - Левое колено")
    print("  ID 9  - Левая ступня")
    print("  ID 10 - Правое колено")
    print("  ID 11 - Правая ступня")
    print("\nУправление:")
    print("  Q - Выход")
    print("  M - Переключение режима")
    print("  G - Генерация маркеров")
    print("  S - Сохранение настроек")
    print("  T - Тест детекции ArUco")
    print("=====================================\n")
    
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nОстановка...")
        app.stop()

if __name__ == "__main__":
    main()