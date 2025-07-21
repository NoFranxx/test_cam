import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from enum import Enum
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionMode(Enum):
    MEDIAPIPE_ONLY = "mediapipe"
    ARUCO_ONLY = "aruco"
    HYBRID = "hybrid"

@dataclass
class AppConfig:
    """Конфигурация приложения"""
    # Общие настройки
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    fps: int = 30
    
    # Настройки детекции
    detection_mode: DetectionMode = DetectionMode.HYBRID
    
    # MediaPipe настройки
    mp_detection_confidence: float = 0.7
    mp_tracking_confidence: float = 0.5
    mp_model_complexity: int = 1
    
    # ArUco настройки
    aruco_dict_type: str = "DICT_4X4_50"  # Изменено на строку
    aruco_marker_size: float = 0.05  # размер маркера в метрах
    
    # Настройки управления курсором
    cursor_smoothing: float = 0.3
    cursor_sensitivity: float = 2.0
    screen_width: int = pyautogui.size()[0]
    screen_height: int = pyautogui.size()[1]
    
    # Настройки стабилизации
    use_kalman_filter: bool = True
    detection_threshold: int = 3  # минимальное количество кадров для подтверждения детекции
    
    def save(self, filename: str = "config.json"):
        """Сохранение конфигурации в файл"""
        config_dict = {
            k: v.value if isinstance(v, Enum) else v 
            for k, v in self.__dict__.items()
        }
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, filename: str = "config.json") -> 'AppConfig':
        """Загрузка конфигурации из файла"""
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            config_dict['detection_mode'] = DetectionMode(config_dict.get('detection_mode', 'hybrid'))
            return cls(**config_dict)
        except FileNotFoundError:
            logger.warning(f"Файл конфигурации {filename} не найден. Используются настройки по умолчанию.")
            return cls()

class KalmanFilter:
    """Фильтр Калмана для сглаживания координат"""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.initialized = False
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Обновление фильтра и получение сглаженных координат"""
        measurement = np.array([[x], [y]], dtype=np.float32)
        
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
        
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        
        return float(prediction[0]), float(prediction[1])

class BodyTracker:
    """Класс для отслеживания тела с использованием MediaPipe и ArUco"""
    
    # Соответствие меток позициям тела
    BODY_LANDMARKS = {
        'head': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'pelvis': 23,  # Используем среднее между бедрами
        'left_elbow': 13,
        'left_wrist': 15,
        'right_elbow': 14,
        'right_wrist': 16,
        'left_knee': 25,
        'left_ankle': 27,
        'right_knee': 26,
        'right_ankle': 28
    }
    
    # Соответствие ArUco маркеров позициям тела
    ARUCO_MARKERS = {
        0: 'head',
        1: 'left_shoulder',
        2: 'right_shoulder',
        3: 'pelvis',
        4: 'left_elbow',
        5: 'left_wrist',
        6: 'right_elbow',
        7: 'right_wrist',
        8: 'left_knee',
        9: 'left_ankle',
        10: 'right_knee',
        11: 'right_ankle'
    }
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # Инициализация MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.mp_model_complexity,
            min_detection_confidence=config.mp_detection_confidence,
            min_tracking_confidence=config.mp_tracking_confidence
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.mp_detection_confidence,
            min_tracking_confidence=config.mp_tracking_confidence
        )
        
        # Инициализация ArUco с новым API
        try:
            # Получаем словарь ArUco
            aruco_dict_id = getattr(cv2.aruco, config.aruco_dict_type)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
            
            # Создаем детектор с новым API
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except AttributeError:
            # Fallback для старых версий OpenCV
            logger.warning("Используется старый API ArUco")
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None
        
        # Калибровка камеры (упрощенная)
        self.camera_matrix = np.array([[800, 0, config.camera_width/2],
                                      [0, 800, config.camera_height/2],
                                      [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((4,1))
        
        # Фильтры Калмана для сглаживания
        self.kalman_filters = {}
        
        # История детекций для стабилизации
        self.detection_history = {}
        
        # Управление курсором
        self.cursor_position = None
        self.cursor_kalman = KalmanFilter() if config.use_kalman_filter else None
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Обработка кадра в зависимости от режима"""
        results = {
            'pose_landmarks': None,
            'hand_landmarks': None,
            'aruco_markers': {},
            'cursor_control': None
        }
        
        if self.config.detection_mode in [DetectionMode.MEDIAPIPE_ONLY, DetectionMode.HYBRID]:
            results.update(self._process_mediapipe(frame))
        
        if self.config.detection_mode in [DetectionMode.ARUCO_ONLY, DetectionMode.HYBRID]:
            results.update(self._process_aruco(frame))
        
        # Определение управления курсором
        if results['hand_landmarks']:
            results['cursor_control'] = self._calculate_cursor_position(
                results['hand_landmarks'], frame.shape
            )
        
        return results
    
    def _process_mediapipe(self, frame: np.ndarray) -> Dict:
        """Обработка MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)
        
        results = {}
        
        if pose_results.pose_landmarks:
            results['pose_landmarks'] = pose_results.pose_landmarks
        
        if hands_results.multi_hand_landmarks:
            results['hand_landmarks'] = hands_results.multi_hand_landmarks[0]  # Берем первую руку
        
        return results
    
    def _process_aruco(self, frame: np.ndarray) -> Dict:
        """Обработка ArUco маркеров"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Используем новый или старый API в зависимости от версии
        if self.aruco_detector:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        aruco_markers = {}
        
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.ARUCO_MARKERS:
                    # Оценка позы маркера
                    try:
                        # Новый API
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners[i], self.config.aruco_marker_size, 
                            self.camera_matrix, self.dist_coeffs
                        )
                    except:
                        # Альтернативный метод для новых версий
                        rvec = np.zeros((1, 1, 3))
                        tvec = np.zeros((1, 1, 3))
                    
                    # Получение центра маркера
                    center = np.mean(corners[i][0], axis=0)
                    
                    # Применение фильтра Калмана если включено
                    if self.config.use_kalman_filter:
                        filter_key = f"aruco_{marker_id}"
                        if filter_key not in self.kalman_filters:
                            self.kalman_filters[filter_key] = KalmanFilter()
                        center = self.kalman_filters[filter_key].update(center[0], center[1])
                    
                    aruco_markers[self.ARUCO_MARKERS[marker_id]] = {
                        'position': center,
                        'corners': corners[i][0],
                        'rvec': rvec,
                        'tvec': tvec
                    }
        
        return {'aruco_markers': aruco_markers}
    
    def _calculate_cursor_position(self, hand_landmarks, frame_shape) -> Optional[Tuple[int, int]]:
        """Расчет позиции курсора на основе положения руки"""
        # Используем указательный палец для управления
        index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Преобразование нормализованных координат в экранные
        x = index_finger.x * frame_shape[1]
        y = index_finger.y * frame_shape[0]
        
        # Применение фильтра Калмана
        if self.cursor_kalman:
            x, y = self.cursor_kalman.update(x, y)
        
        # Масштабирование на размер экрана
        screen_x = int(x * self.config.screen_width / frame_shape[1] * self.config.cursor_sensitivity)
        screen_y = int(y * self.config.screen_height / frame_shape[0] * self.config.cursor_sensitivity)
        
        # Ограничение координат экраном
        screen_x = max(0, min(screen_x, self.config.screen_width - 1))
        screen_y = max(0, min(screen_y, self.config.screen_height - 1))
        
        return (screen_x, screen_y)
    
    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Отрисовка результатов на кадре"""
        # Отрисовка MediaPipe
        if results.get('pose_landmarks'):
            self.mp_drawing.draw_landmarks(
                frame, results['pose_landmarks'], self.mp_pose.POSE_CONNECTIONS
            )
        
        if results.get('hand_landmarks'):
            self.mp_drawing.draw_landmarks(
                frame, results['hand_landmarks'], self.mp_hands.HAND_CONNECTIONS
            )
        
        # Отрисовка ArUco
        for body_part, marker_data in results.get('aruco_markers', {}).items():
            corners = marker_data['corners']
            cv2.polylines(frame, [corners.astype(int)], True, (0, 255, 0), 2)
            
            # Подпись части тела
            center = tuple(np.mean(corners, axis=0).astype(int))
            cv2.putText(frame, body_part, center, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 0, 0), 2)
        
        # Отображение информации о курсоре
        if results.get('cursor_control'):
            x, y = results['cursor_control']
            cv2.putText(frame, f"Cursor: ({x}, {y})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Отображение режима
        mode_text = f"Mode: {self.config.detection_mode.value}"
        cv2.putText(frame, mode_text, (10, frame.shape[0] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame

class CursorController(threading.Thread):
    """Поток для управления курсором"""
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.cursor_queue = queue.Queue()
        self.running = True
        self.daemon = True
        
    def run(self):
        """Основной цикл управления курсором"""
        last_position = None
        
        while self.running:
            try:
                position = self.cursor_queue.get(timeout=0.1)
                
                if position and last_position:
                    # Плавное перемещение курсора
                    smooth_x = last_position[0] + (position[0] - last_position[0]) * self.config.cursor_smoothing
                    smooth_y = last_position[1] + (position[1] - last_position[1]) * self.config.cursor_smoothing
                    
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                    last_position = (smooth_x, smooth_y)
                elif position:
                    pyautogui.moveTo(position[0], position[1], duration=0)
                    last_position = position
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка управления курсором: {e}")
    
    def update_position(self, position: Optional[Tuple[int, int]]):
        """Обновление позиции курсора"""
        if position:
            try:
                self.cursor_queue.put_nowait(position)
            except queue.Full:
                # Очищаем очередь если она переполнена
                try:
                    self.cursor_queue.get_nowait()
                    self.cursor_queue.put_nowait(position)
                except queue.Empty:
                    pass
    
    def stop(self):
        """Остановка потока"""
        self.running = False

class Application:
    """Основное приложение"""
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        self.tracker = BodyTracker(self.config)
        self.cursor_controller = CursorController(self.config)
        self.running = False
        
        # Статистика производительности
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Настройка горячих клавиш
        self.key_bindings = {
            ord('q'): self.quit,
            ord('m'): lambda: self.switch_mode(DetectionMode.MEDIAPIPE_ONLY),
            ord('a'): lambda: self.switch_mode(DetectionMode.ARUCO_ONLY),
            ord('h'): lambda: self.switch_mode(DetectionMode.HYBRID),
            ord('s'): self.save_config,
            ord('l'): self.load_config,
            ord('c'): self.toggle_cursor_control,
            ord('k'): self.toggle_kalman_filter,
            ord('r'): self.reset_tracking,
            ord('+'): lambda: self.adjust_sensitivity(0.1),
            ord('-'): lambda: self.adjust_sensitivity(-0.1),
            ord('i'): self.show_info,
            27: self.quit  # ESC key
        }
        
        self.cursor_control_enabled = True
        self.show_debug_info = True
        
    def initialize_camera(self) -> Optional[cv2.VideoCapture]:
        """Инициализация камеры"""
        cap = cv2.VideoCapture(self.config.camera_index)
        
        if not cap.isOpened():
            logger.error(f"Не удалось открыть камеру {self.config.camera_index}")
            return None
        
        # Установка параметров камеры
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # Проверка фактических параметров
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Камера инициализирована: {actual_width}x{actual_height} @ {actual_fps} FPS")
        
        return cap
    
    def run(self):
        """Основной цикл приложения"""
        cap = self.initialize_camera()
        if not cap:
            return
        
        self.running = True
        self.cursor_controller.start()
        
        # Создание окна
        cv2.namedWindow('Body Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Body Tracking', 1280, 720)
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Не удалось получить кадр с камеры")
                    break
                
                # Обработка кадра
                results = self.tracker.process_frame(frame)
                
                # Управление курсором
                if self.cursor_control_enabled and results.get('cursor_control'):
                    self.cursor_controller.update_position(results['cursor_control'])
                
                # Отрисовка результатов
                frame = self.tracker.draw_results(frame, results)
                
                # Добавление отладочной информации
                if self.show_debug_info:
                    frame = self.draw_debug_info(frame, results)
                
                # Отображение кадра
                cv2.imshow('Body Tracking', frame)
                
                # Обновление FPS
                self.update_fps()
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key in self.key_bindings:
                    self.key_bindings[key]()
                
        except Exception as e:
            logger.error(f"Ошибка в основном цикле: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.cleanup(cap)
    
    def draw_debug_info(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Отрисовка отладочной информации"""
        info_y = 60
        line_height = 25
        
        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += line_height
        
        # Количество обнаруженных объектов
        pose_count = 1 if results.get('pose_landmarks') else 0
        hand_count = 1 if results.get('hand_landmarks') else 0
        aruco_count = len(results.get('aruco_markers', {}))
        
        cv2.putText(frame, f"Pose: {pose_count}, Hands: {hand_count}, ArUco: {aruco_count}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += line_height
        
        # Статус курсора
        cursor_status = "ON" if self.cursor_control_enabled else "OFF"
        cv2.putText(frame, f"Cursor Control: {cursor_status}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += line_height
        
        # Фильтр Калмана
        kalman_status = "ON" if self.config.use_kalman_filter else "OFF"
        cv2.putText(frame, f"Kalman Filter: {kalman_status}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += line_height
        
        # Чувствительность курсора
        cv2.putText(frame, f"Sensitivity: {self.config.cursor_sensitivity:.1f}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Инструкции
        instructions = [
            "Q/ESC - Quit | M - MediaPipe | A - ArUco | H - Hybrid",
            "C - Toggle Cursor | K - Toggle Kalman | I - Toggle Info",
            "+/- Adjust Sensitivity | S - Save Config | L - Load Config | R - Reset"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, frame.shape[0] - 80 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def update_fps(self):
        """Обновление счетчика FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def switch_mode(self, mode: DetectionMode):
        """Переключение режима детекции"""
        self.config.detection_mode = mode
        logger.info(f"Режим переключен на: {mode.value}")
    
    def toggle_cursor_control(self):
        """Включение/выключение управления курсором"""
        self.cursor_control_enabled = not self.cursor_control_enabled
        logger.info(f"Управление курсором: {'включено' if self.cursor_control_enabled else 'выключено'}")
    
    def toggle_kalman_filter(self):
        """Включение/выключение фильтра Калмана"""
        self.config.use_kalman_filter = not self.config.use_kalman_filter
        # Очищаем существующие фильтры
        self.tracker.kalman_filters.clear()
        self.tracker.cursor_kalman = KalmanFilter() if self.config.use_kalman_filter else None
        logger.info(f"Фильтр Калмана: {'включен' if self.config.use_kalman_filter else 'выключен'}")
    
    def adjust_sensitivity(self, delta: float):
        """Регулировка чувствительности курсора"""
        self.config.cursor_sensitivity = max(0.1, min(5.0, self.config.cursor_sensitivity + delta))
        logger.info(f"Чувствительность курсора: {self.config.cursor_sensitivity:.1f}")
    
    def show_info(self):
        """Переключение отображения отладочной информации"""
        self.show_debug_info = not self.show_debug_info
    
    def reset_tracking(self):
        """Сброс отслеживания"""
        self.tracker.kalman_filters.clear()
        self.tracker.detection_history.clear()
        if self.config.use_kalman_filter:
            self.tracker.cursor_kalman = KalmanFilter()
        logger.info("Отслеживание сброшено")
    
    def save_config(self):
        """Сохранение конфигурации"""
        self.config.save()
        logger.info("Конфигурация сохранена")
    
    def load_config(self):
        """Загрузка конфигурации"""
        self.config = AppConfig.load()
        self.tracker.config = self.config
        self.cursor_controller.config = self.config
        logger.info("Конфигурация загружена")
    
    def quit(self):
        """Выход из приложения"""
        self.running = False
    
    def cleanup(self, cap: cv2.VideoCapture):
        """Очистка ресурсов"""
        logger.info("Завершение работы...")
        
        self.cursor_controller.stop()
        self.cursor_controller.join(timeout=1.0)
        
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info("Приложение завершено")

def print_aruco_markers():
    """Функция для генерации и сохранения ArUco маркеров"""
    import os
    
    # Создаем папку для маркеров
    os.makedirs("aruco_markers", exist_ok=True)
    
    # Генерируем маркеры
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    marker_names = {
        0: 'head',
        1: 'left_shoulder',
        2: 'right_shoulder',
        3: 'pelvis',
        4: 'left_elbow',
        5: 'left_wrist',
        6: 'right_elbow',
        7: 'right_wrist',
        8: 'left_knee',
        9: 'left_ankle',
        10: 'right_knee',
        11: 'right_ankle'
    }
    
    for marker_id, body_part in marker_names.items():
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
        filename = f"aruco_markers/marker_{marker_id}_{body_part}.png"
        cv2.imwrite(filename, marker_image)
        logger.info(f"Сохранен маркер: {filename}")

def main():
    """Точка входа в приложение"""
    import sys
    

    if len(sys.argv) > 1 and sys.argv[1] == "--generate-markers":
        print_aruco_markers()
        return
    

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01
    
    # Загрузка или создание конфигурации
    config = AppConfig.load()
    
    # Создание и запуск приложения
    app = Application(config)
    
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Прерывание пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
