from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from htrtdetr.annotator import AnnotationApp
from htrtdetr.annotation.project import ProjectSession

folder = Path(__file__).resolve().parent / 'workspace-preview'
folder.mkdir(exist_ok=True)
for name in ('experiment_A.avi', 'experiment_B.avi'):
    writer = cv2.VideoWriter(str(folder / name), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))
    for index in range(30):
        frame = np.full((480, 640, 3), 40, np.uint8)
        cv2.ellipse(frame, (200 + index, 200), (35, 17), 25, 0, 360, (170, 185, 195), -1)
        writer.write(frame)
    writer.release()
app = AnnotationApp()
dpg.create_viewport(title='YOAKE Workspace visual verification', width=1480, height=960)
dpg.setup_dearpygui()
dpg.show_viewport()
app._build_main_window()
project = ProjectSession('Behavior experiment', ['fly', 'worm'], ['walking', 'resting'], directory=folder)
project.settings['onboarding_done'] = True
app._adopt_project(project)
app._add_videos([folder / 'experiment_A.avi', folder / 'experiment_B.avi'])
app._create_object([160, 170, 245, 235])
for _ in range(8):
    dpg.render_dearpygui_frame()
app._on_viewport_resize()
for _ in range(8):
    dpg.render_dearpygui_frame()
dpg.output_frame_buffer(file=str(folder / 'canvas.png'))
for _ in range(8):
    dpg.render_dearpygui_frame()
app.show_video_browser()
for _ in range(8):
    dpg.render_dearpygui_frame()
dpg.output_frame_buffer(file=str(folder / 'browser.png'))
for _ in range(8):
    dpg.render_dearpygui_frame()
app.cap.release()
app._stop_sam()
dpg.destroy_context()
