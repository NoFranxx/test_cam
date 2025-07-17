import cv2
import mediapipe as mp
import numpy as np
import openpyxl
from openpyxl import Workbook
import time
from datetime import datetime
import pyautogui
import warnings
import os
import logging
import psutil
import multiprocessing
import threading
import queue

# Подавление предупреждений
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)

# GPU настройки
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Максимальная производительность
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['TF_NUM_INTEROP_THREADS'] = str(multiprocessing.cpu_count())
os.environ['TF_NUM_INTRAOP_THREADS'] = str(multiprocessing.cpu_count())

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Проверка GPU
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU обнаружен: {len(gpus)} устройств")
    else:
        print("GPU не обнаружен, используется CPU")
except:
    print("TensorFlow не установлен для GPU")

# Установка высокого приоритета
try:
    import sys
    if sys.platform == 'win32':
        import win32api
        import win32process
        import win32con
        handle = win32api.GetCurrentProcess()
        win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
except:
    pass

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

class GPUOptimizedController:
    def __init__(self):
        # MediaPipe с GPU оптимизацией
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        
        # Holistic для тела
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=False
        )
        
        # Отдельный детектор рук для точности пальцев
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Параметры управления мышью
        self.mouse_control_active = False
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothing_factor = 0.7
        self.prev_x, self.prev_y = 0, 0
        self.mouse_sensitivity = 2.0
        
        # Расширенная зона управления
        self.control_zone = {
            'x_min': 0.05,
            'x_max': 0.95,
            'y_min': 0.05,
            'y_max': 0.95
        }
        
        # Excel настройки
        self.setup_excel()
        self.last_save_time = time.time()
        self.save_interval = 0.5
        self.recording = False
        
        # Статистика производительности
        self.stats = {
            'pointing': 0,
            'clicks': 0,
            'fps': 0,
            'gpu_usage': 0,
            'frame_time': 0
        }
        
        # Параметры клика
        self.click_threshold = 0.06
        self.was_clicking = False
        
        # FPS счетчики
        self.fps_time = time.time()
        self.fps_counter = 0
        self.frame_times = []
        
        # Многопоточность
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Цвета для точек
        self.colors = {
            'right_index': (0, 255, 0),      # Зеленый для правого указательного
            'right_thumb': (0, 200, 0),      # Темно-зеленый для правого большого
            'right_palm': (0, 150, 0),       # Еще темнее для правой ладони
            'left_index': (255, 0, 0),       # Красный для левого указательного
            'left_thumb': (200, 0, 0),       # Темно-красный для левого большого
            'left_palm': (150, 0, 0),        # Еще темнее для левой ладони
            'body': (255, 255, 0),           # Желтый для тела
            'connection': (100, 100, 100)    # Серый для соединений
        }
        
        # Отладочный режим
        self.debug_mode = False
        
    def setup_excel(self):
        """Excel с расширенными данными"""
        self.wb = Workbook()
        self.ws = self.wb.active
        self.ws.title = "GPU Tracking Data"
        
        headers = ["Timestamp", "Gesture", "Mouse_Control", "Mouse_X", "Mouse_Y", 
                  "FPS", "Frame_Time_ms", "GPU_Usage"]
        
        # Точки для обеих рук
        for hand in ['Right', 'Left']:
            headers.extend([
                f"{hand}_Index_X", f"{hand}_Index_Y", f"{hand}_Index_Z",
                f"{hand}_Thumb_X", f"{hand}_Thumb_Y", f"{hand}_Thumb_Z",
                f"{hand}_Palm_X", f"{hand}_Palm_Y", f"{hand}_Palm_Z"
            ])
        
        # Основные точки тела
        body_points = ["L_Elbow", "R_Elbow", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Foot", "R_Foot"]
        for point in body_points:
            headers.extend([f"{point}_X", f"{point}_Y", f"{point}_Z"])
        
        self.ws.append(headers)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"gpu_tracking_{timestamp}.xlsx"
        
    def detect_pointing_gesture_precise(self, hand_landmarks, is_right_hand):
        """Точное определение жеста указывания"""
        if not hand_landmarks:
            return False
            
        landmarks = hand_landmarks.landmark
        
        # Индексы точек MediaPipe Hands
        INDEX_TIP = 8
        INDEX_PIP = 6
        INDEX_MCP = 5
        MIDDLE_TIP = 12
        MIDDLE_PIP = 10
        RING_TIP = 16
        RING_PIP = 14
        PINKY_TIP = 20
        PINKY_PIP = 18
        
        # Проверка выпрямленного указательного пальца
        index_extended = landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y < landmarks[INDEX_MCP].y
        
        # Проверка согнутых остальных пальцев
        middle_folded = landmarks[MIDDLE_TIP].y > landmarks[MIDDLE_PIP].y
        ring_folded = landmarks[RING_TIP].y > landmarks[RING_PIP].y
        pinky_folded = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y
        
        is_pointing = index_extended and middle_folded and ring_folded and pinky_folded
        
        if is_pointing:
            self.stats['pointing'] += 1
            
        return is_pointing
        
    def process_frame_gpu(self, frame):
        """Обработка кадра с GPU ускорением"""
        start_time = time.time()
        
        # Конвертация в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Параллельная обработка
        holistic_results = self.holistic.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)
        
        # Время обработки
        process_time = (time.time() - start_time) * 1000
        self.frame_times.append(process_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        self.stats['frame_time'] = np.mean(self.frame_times)
        
        return holistic_results, hands_results
        
    def draw_precise_hands(self, image, hands_results, holistic_results):
        """Точная отрисовка рук с правильным определением"""
        h, w, _ = image.shape
        
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(
                hands_results.multi_hand_landmarks, 
                hands_results.multi_handedness
            )):
                # Определяем какая рука (с учетом НЕ зеркального изображения)
                hand_label = handedness.classification[0].label
                is_right = (hand_label == "Right")
                
                # Выбираем цвета
                if is_right:
                    index_color = self.colors['right_index']
                    thumb_color = self.colors['right_thumb']
                    palm_color = self.colors['right_palm']
                    label = "RIGHT"
                else:
                    index_color = self.colors['left_index']
                    thumb_color = self.colors['left_thumb']
                    palm_color = self.colors['left_palm']
                    label = "LEFT"
                
                landmarks = hand_landmarks.landmark
                
                # Указательный палец (точка 8)
                index_tip = landmarks[8]
                index_x = int(index_tip.x * w)
                index_y = int(index_tip.y * h)
                cv2.circle(image, (index_x, index_y), 15, index_color, -1)
                cv2.circle(image, (index_x, index_y), 17, (255, 255, 255), 2)
                cv2.putText(image, f"{label[0]}I", (index_x - 15, index_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, index_color, 2)
                
                # Большой палец (точка 4)
                thumb_tip = landmarks[4]
                thumb_x = int(thumb_tip.x * w)
                thumb_y = int(thumb_tip.y * h)
                cv2.circle(image, (thumb_x, thumb_y), 15, thumb_color, -1)
                cv2.circle(image, (thumb_x, thumb_y), 17, (255, 255, 255), 2)
                cv2.putText(image, f"{label[0]}T", (thumb_x - 15, thumb_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, thumb_color, 2)
                
                # Центр ладони (точка 0 - запястье)
                palm = landmarks[0]
                palm_x = int(palm.x * w)
                palm_y = int(palm.y * h)
                cv2.circle(image, (palm_x, palm_y), 20, palm_color, -1)
                cv2.circle(image, (palm_x, palm_y), 22, (255, 255, 255), 2)
                cv2.putText(image, f"{label[0]}P", (palm_x - 15, palm_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, palm_color, 2)
                
                # Линия между большим и указательным для визуализации клика
                cv2.line(image, (index_x, index_y), (thumb_x, thumb_y), (200, 200, 200), 2)
                
                # Расстояние для клика
                distance = np.sqrt((index_tip.x - thumb_tip.x)**2 + 
                                 (index_tip.y - thumb_tip.y)**2 + 
                                 (index_tip.z - thumb_tip.z)**2)
                
                # Визуализация расстояния
                mid_x = (index_x + thumb_x) // 2
                mid_y = (index_y + thumb_y) // 2
                color = (0, 255, 0) if distance < self.click_threshold else (200, 200, 200)
                cv2.putText(image, f"{distance:.3f}", (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Отладочная информация
                if self.debug_mode:
                    info_y = 150 + idx * 100
                    cv2.putText(image, f"{label} HAND", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f"Confidence: {handedness.classification[0].score:.2f}", 
                               (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
        # Отрисовка основных точек тела из holistic
        if holistic_results.pose_landmarks:
            self.draw_body_points(image, holistic_results.pose_landmarks)
            
    def draw_body_points(self, image, pose_landmarks):
        """Отрисовка ключевых точек тела"""
        h, w, _ = image.shape
        landmarks = pose_landmarks.landmark
        
        # Точки тела
        body_points = {
            'L_Elbow': 13, 'R_Elbow': 14,
            'L_Hip': 23, 'R_Hip': 24,
            'L_Knee': 25, 'R_Knee': 26,
            'L_Foot': 31, 'R_Foot': 32
        }
        
        for name, idx in body_points.items():
            if idx < len(landmarks):
                point = landmarks[idx]
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(image, (x, y), 10, self.colors['body'], -1)
                cv2.putText(image, name[:2], (x - 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['body'], 1)
                           
    def get_right_hand_data(self, hands_results):
        """Получение данных правой руки с правильным определением"""
        if not hands_results.multi_hand_landmarks or not hands_results.multi_handedness:
            return None, None
            
        for hand_landmarks, handedness in zip(
            hands_results.multi_hand_landmarks,
            hands_results.multi_handedness
        ):
            # Проверяем что это именно правая рука
            if handedness.classification[0].label == "Right":
                return hand_landmarks, True
                
        return None, None
        
    def map_hand_to_screen_inverted(self, hand_x, hand_y):
        """Преобразование с инверсией по X"""
        if not (self.control_zone['x_min'] <= hand_x <= self.control_zone['x_max'] and
                self.control_zone['y_min'] <= hand_y <= self.control_zone['y_max']):
            return None, None
            
        norm_x = (hand_x - self.control_zone['x_min']) / (self.control_zone['x_max'] - self.control_zone['x_min'])
        norm_y = (hand_y - self.control_zone['y_min']) / (self.control_zone['y_max'] - self.control_zone['y_min'])
        
        # Инверсия по X для зеркального управления
        norm_x = 1 - norm_x
        
        # Применяем кривую ускорения
        norm_x = self.apply_acceleration_curve(norm_x)
        norm_y = self.apply_acceleration_curve(norm_y)
        
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        
        return screen_x, screen_y
        
    def apply_acceleration_curve(self, value):
        """Кривая ускорения"""
        if value < 0.5:
            return 2 * value * value
        else:
            return 1 - 2 * (1 - value) * (1 - value)
            
    def smooth_mouse_movement(self, target_x, target_y):
        """Сглаживание движения"""
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = target_x, target_y
            return target_x, target_y
            
        smooth_x = int(self.prev_x * self.smoothing_factor + target_x * (1 - self.smoothing_factor))
        smooth_y = int(self.prev_y * self.smoothing_factor + target_y * (1 - self.smoothing_factor))
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        
        return smooth_x, smooth_y
        
    def draw_interface(self, image):
        """Интерфейс с производительностью"""
        h, w, _ = image.shape
        
        # Зона управления
        if self.mouse_control_active:
            cv2.rectangle(image, (20, 20), (w-20, h-20), (0, 255, 0), 3)
            cv2.putText(image, "MOUSE CONTROL ACTIVE", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Панель производительности
        panel_x = w - 250
        panel_y = 20
        
        # Фон панели
        cv2.rectangle(image, (panel_x - 10, panel_y - 10), 
                     (w - 10, panel_y + 150), (0, 0, 0), -1)
        cv2.rectangle(image, (panel_x - 10, panel_y - 10), 
                     (w - 10, panel_y + 150), (100, 100, 100), 1)
        
        # FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_time >= 1.0:
            self.stats['fps'] = self.fps_counter
            self.fps_counter = 0
            self.fps_time = current_time
            
        cv2.putText(image, f"FPS: {self.stats['fps']}", (panel_x, panel_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Время кадра
        cv2.putText(image, f"Frame: {self.stats['frame_time']:.1f}ms", (panel_x, panel_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Использование ресурсов
        cpu_percent = psutil.cpu_percent(interval=0)
        memory_percent = psutil.virtual_memory().percent
        
        cv2.putText(image, f"CPU: {cpu_percent:.1f}%", (panel_x, panel_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
        cv2.putText(image, f"RAM: {memory_percent:.1f}%", (panel_x, panel_y + 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
        
        # GPU (если есть nvidia-smi)
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_usage = int(result.stdout.strip())
                self.stats['gpu_usage'] = gpu_usage
                cv2.putText(image, f"GPU: {gpu_usage}%", (panel_x, panel_y + 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        except:
            pass
        
        # Статистика
        cv2.putText(image, f"Clicks: {self.stats['clicks']}", (30, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Инструкции
        cv2.putText(image, "F-Fullscreen | D-Debug | Q-Quit", (30, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Запись
        if self.recording:
            cv2.circle(image, (30, 100), 15, (0, 0, 255), -1)
            cv2.putText(image, "REC", (50, 107),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
    def run(self):
        """Основной цикл с GPU оптимизацией"""
        # Настройка камеры для максимальной производительности
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        window_name = 'GPU Optimized Tracking'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\n=== GPU ОПТИМИЗИРОВАННОЕ ОТСЛЕЖИВАНИЕ ===")
        print(f"CPU: {multiprocessing.cpu_count()} ядер")
        print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
            
        print("\nГорячие клавиши:")
        print("F - полноэкранный режим")
        print("D - режим отладки")
        print("R - запись")
        print("Q - выход")
        print("-" * 50)
        
        fullscreen = False
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # НЕ зеркалируем для правильного определения рук
                # frame = cv2.flip(frame, 1)
                
                # GPU обработка
                holistic_results, hands_results = self.process_frame_gpu(frame)
                
                # Отрисовка точных рук
                self.draw_precise_hands(frame, hands_results, holistic_results)
                
                # Управление мышью правой рукой
                right_hand, is_right = self.get_right_hand_data(hands_results)
                
                if right_hand and is_right:
                    # Проверка жеста
                    is_pointing = self.detect_pointing_gesture_precise(right_hand, True)
                    
                    if is_pointing != self.mouse_control_active:
                        self.mouse_control_active = is_pointing
                        if is_pointing:
                            print("☝ Управление правой рукой АКТИВИРОВАНО")
                        else:
                            print("✊ Управление ДЕАКТИВИРОВАНО")
                            self.prev_x, self.prev_y = 0, 0
                    
                    if self.mouse_control_active:
                        # Позиция указательного пальца
                        index_tip = right_hand.landmark[8]
                        
                        # Преобразование с инверсией
                        screen_x, screen_y = self.map_hand_to_screen_inverted(
                            index_tip.x, index_tip.y
                        )
                        
                        if screen_x is not None:
                            smooth_x, smooth_y = self.smooth_mouse_movement(screen_x, screen_y)
                            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                            
                            # Проверка клика
                            thumb_tip = right_hand.landmark[4]
                            distance = np.sqrt(
                                (index_tip.x - thumb_tip.x)**2 +
                                (index_tip.y - thumb_tip.y)**2 +
                                (index_tip.z - thumb_tip.z)**2
                            )
                            
                            is_clicking = distance < self.click_threshold
                            
                            if is_clicking and not self.was_clicking:
                                pyautogui.click()
                                self.stats['clicks'] += 1
                                print(f"👆 Клик! (расстояние: {distance:.3f})")
                                
                                # Визуальный эффект
                                h, w = frame.shape[:2]
                                click_x = int(index_tip.x * w)
                                click_y = int(index_tip.y * h)
                                cv2.circle(frame, (click_x, click_y), 60, (0, 255, 0), 5)
                                
                            self.was_clicking = is_clicking
                else:
                    self.mouse_control_active = False
                    self.was_clicking = False
                
                # Интерфейс
                self.draw_interface(frame)
                
                # Сохранение данных
                if self.recording:
                    current_time = time.time()
                    if (current_time - self.last_save_time) >= self.save_interval:
                        self.save_tracking_data(hands_results, holistic_results)
                        self.last_save_time = current_time
                
                # Показ
                cv2.imshow(window_name, frame)
                
                # Клавиши
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                            cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                            cv2.WINDOW_NORMAL)
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Режим отладки: {'ВКЛ' if self.debug_mode else 'ВЫКЛ'}")
                elif key == ord('r'):
                    self.recording = not self.recording
                    if self.recording:
                        print("⏺ Запись начата")
                    else:
                        print("⏹ Запись остановлена")
                        self.wb.save(self.filename)
                    
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            if self.recording:
                self.wb.save(self.filename)
                print(f"\n💾 Данные сохранены: {self.filename}")
                
            print(f"\nСтатистика:")
            print(f"- Жестов: {self.stats['pointing']}")
            print(f"- Кликов: {self.stats['clicks']}")
            print(f"- Средний FPS: {self.stats['fps']}")
            print(f"- Среднее время кадра: {self.stats['frame_time']:.1f}ms")
            
    def save_tracking_data(self, hands_results, holistic_results):
        """Сохранение данных"""
        row_data = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "Pointing" if self.mouse_control_active else "None",
            self.mouse_control_active,
            self.prev_x if self.mouse_control_active else "",
            self.prev_y if self.mouse_control_active else "",
            self.stats['fps'],
            round(self.stats['frame_time'], 1),
            self.stats.get('gpu_usage', 0)
        ]
        
        # Сохранение данных рук
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            hands_data = {'Right': None, 'Left': None}
            
            for hand_landmarks, handedness in zip(
                hands_results.multi_hand_landmarks,
                hands_results.multi_handedness
            ):
                label = handedness.classification[0].label
                if label in hands_data:
                    hands_data[label] = hand_landmarks
            
            # Добавляем данные для каждой руки
            for hand_type in ['Right', 'Left']:
                if hands_data[hand_type]:
                    landmarks = hands_data[hand_type].landmark
                    # Указательный палец
                    row_data.extend([
                        round(landmarks[8].x, 6),
                        round(landmarks[8].y, 6),
                        round(landmarks[8].z, 6)
                    ])
                    # Большой палец
                    row_data.extend([
                        round(landmarks[4].x, 6),
                        round(landmarks[4].y, 6),
                        round(landmarks[4].z, 6)
                    ])
                    # Ладонь
                    row_data.extend([
                        round(landmarks[0].x, 6),
                        round(landmarks[0].y, 6),
                        round(landmarks[0].z, 6)
                    ])
                else:
                    row_data.extend([""] * 9)
        else:
            row_data.extend([""] * 18)
            
        # Данные тела
        if holistic_results.pose_landmarks:
            landmarks = holistic_results.pose_landmarks.landmark
            body_indices = [13, 14, 23, 24, 25, 26, 31, 32]
            for idx in body_indices:
                if idx < len(landmarks):
                    row_data.extend([
                        round(landmarks[idx].x, 6),
                        round(landmarks[idx].y, 6),
                        round(landmarks[idx].z, 6)
                    ])
                else:
                    row_data.extend(["", "", ""])
        else:
            row_data.extend([""] * 24)
            
        self.ws.append(row_data)

if __name__ == "__main__":
    try:
        controller = GPUOptimizedController()
        controller.run()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана")
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()