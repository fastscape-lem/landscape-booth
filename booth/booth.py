import io
import os
import time
from threading import Thread

import cv2
import numpy as np
import pooch
import matplotlib.pyplot as plt
from skimage import filters
from ipycanvas import MultiCanvas, hold_canvas
from ipywidgets import (
    Button,
    FloatSlider,
    IntSlider,
    HTML,
    Label,
    HBox,
    VBox,
    Layout,
    Text,
)
from ipywebrtc import CameraStream, ImageRecorder

from .toposim import TopographySimulator
from .particles import Particles


file_path = pooch.retrieve(
    url="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    known_hash="sha256:0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0",
)
face_cascade = cv2.CascadeClassifier(file_path)


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


class LEMBooth:

    def __init__(
        self,
        scale=3,
        particles=True,
        particles_color="black",
        single_flow=False,
        snapshot=False,
        cmap=plt.cm.terrain,
        grid_size=240,
    ):
        self.scale = scale
        self.particles = particles
        self.particles_color = particles_color
        self.single_flow = single_flow
        self.snapshot = snapshot
        self.grid_size = grid_size
        self.record = False
        self.save = False

        self.toposim = TopographySimulator(
            shape=(grid_size, grid_size), cmap=cmap
        )
        self.u_rate = np.zeros(self.toposim.shape)
        
        if self.particles:
            self.particles = Particles(self.toposim, scale)

        self.setup_canvas()
        self.setup_play_widgets()
        if self.particles:
            self.setup_particles_widgets()
        self.setup_toposim_widgets()
        self.setup_save_widgets()
        self.setup_layout()

        self.process = None
        self._running = False
        self._step = 0
        self._snapshot_step = 0
        self._save_step = 0
        
        self.camera = CameraStream(
            constraints={
                'facing_mode': 'user',
                'audio': False,
                'video': { 'width': 640, 'height': 480 },
            },
        )
        self.image_recorder = ImageRecorder(stream=self.camera)
        self.image_recorder.recording = False

    def setup_canvas(self):
        # canvas 0: topography
        # canvas 1: particles

        self.canvas = MultiCanvas(
            ncanvases=2,
            width=self.scale * self.toposim.shape[0],
            height=self.scale * self.toposim.shape[1],
        )
        self.canvas.on_client_ready(self.redraw)
        
        # prevent error when saving snapshot to file
        # for the 1st time (no image data)
        self.canvas.sync_image_data = True
        time.sleep(0.1)
        self.canvas.sync_image_data = False

    def setup_play_widgets(self):
        self.play_widgets = {
            'start': Button(description="Start", icon='play'),
            'stop': Button(description="Stop/Reset", icon='stop',
                           disabled=True)
        }

        self.play_widgets['start'].on_click(self.start)
        self.play_widgets['stop'].on_click(self.stop)
        
        if self.snapshot:
            snapshot_desc = "Take picture"
            snapshot_icon = "fa-camera"
        else:
            snapshot_desc = "Record"
            snapshot_icon = "fa-video"
        
        self.play_widgets['snapshot'] = Button(
            description=snapshot_desc, disabled=True, icon=snapshot_icon
        )
        self.play_widgets['snapshot'].on_click(self.click_snapshot)

    def setup_particles_widgets(self):
        self.particles_labels = {
            'size': Label(value='Number of particles'),
            'speed': Label(value='Particle "speed"')
        }

        self.particles_widgets = {
            'size': IntSlider(
                value=13000, min=500, max=15000, step=500
            ),
            'speed': FloatSlider(
                value=0.9, min=0.1, max=1., step=0.1
            )
        }

        self.particles_widgets['size'].observe(
            self.on_change_size, names='value'
        )
        self.particles_widgets['speed'].observe(
            self.on_change_speed, names='value'
        )

    def on_change_size(self, change):
        self.particles.n_particles = change.new
        self.initialize()

    def on_change_speed(self, change):
        self.particles.speed_factor = change.new
    
    def click_snapshot(self, b):
        if self.snapshot:
            self.record = True
        else:
            # toggle
            self.record = not self.record
            
            if self.record:
                self.play_widgets['snapshot'].description = "Stop record"
                self.play_widgets['snapshot'].icon = "fa-video-slash"
            else:
                self.play_widgets['snapshot'].description = "Record"
                self.play_widgets['snapshot'].icon = "fa-video"

    def setup_toposim_widgets(self):
        self.toposim_labels = {
            'kf': Label(value='River incision coefficient'),
            'g': Label(value='River transport coefficient'),
            'kd': Label(value='Hillslope diffusivity'),
            'p': Label(value='Flow partition exponent'),
            'u': Label(value='Uplift rate scale')
        }

        self.toposim_widgets = {
            'kf': FloatSlider(
                value=2e-4, min=5e-5, max=3e-4, step=1e-5,
                readout_format='.1e'
            ),
            'g': FloatSlider(
                value=0.7, min=0.5, max=1.5, step=0.1,
                readout_format='.1f', disabled=self.single_flow
            ),
            'kd': FloatSlider(
                value=0.07, min=0., max=0.1, step=0.01,
            ),
            'p': FloatSlider(
                value=6., min=0., max=10., step=0.2,
                readout_format='.1f', disabled=self.single_flow
            ),
            'u': FloatSlider(
                value=5., min=0., max=10, step=0.5,
                readout_format='.1f'
            )
        }
    
    def setup_save_widgets(self):
        self.save_text = Text(placeholder="filename without extension")
        self.save_button = Button(
            description="Snapshot to file", disabled=True, icon='fa-file-image'
        )
        self.save_button.on_click(self.on_save)
        
    def on_save(self, b):
        self.save = True

    def setup_layout(self):
        labels_boxes = []

        play_box = HBox(tuple(self.play_widgets.values()))
        labels_boxes.append(play_box)

        if self.particles:
            particles_hboxes = []
            for k in self.particles_widgets:
                self.particles_labels[k].layout = Layout(width='200px')
                self.particles_widgets[k].layout = Layout(width='200px')

                particles_hboxes.append(
                    HBox([self.particles_labels[k], self.particles_widgets[k]])
                )

            particles_label = HTML(value='<b>Particles parameters</b>')
            particles_box = VBox(particles_hboxes)
            particles_box.layout = Layout(grid_gap='6px')
            labels_boxes += [particles_label, particles_box]

        toposim_hboxes = []
        for k in self.toposim_widgets:
            self.toposim_labels[k].layout = Layout(width='200px')
            self.toposim_widgets[k].layout = Layout(width='200px')

            toposim_hboxes.append(
                HBox([self.toposim_labels[k], self.toposim_widgets[k]])
            )

        toposim_label = HTML(
            value='<b>Landscape evolution model parameters</b>'
        )
        toposim_box = VBox(toposim_hboxes)
        toposim_box.layout = Layout(grid_gap='6px')
        labels_boxes += [toposim_label, toposim_box]
        
        save_label = HTML(
            value='<b>Take snapshot and save to file</b>'
        )
        save_box = HBox([self.save_text, self.save_button])
        labels_boxes += [save_label, save_box]

        control_box = VBox(labels_boxes)
        control_box.layout = Layout(grid_gap='10px')

        self.main_box = HBox((self.canvas, control_box))
        self.main_box.layout = Layout(grid_gap='30px')

    def set_erosion_params(self):
        if self.single_flow:
            p = 10
            g = 0
        else:
            p = self.toposim_widgets['p'].value
            g = self.toposim_widgets['g'].value

        self.toposim.set_erosion_params(
            kf=self.toposim_widgets['kf'].value,
            g=g,
            kd=self.toposim_widgets['kd'].value,
            p=p,
            u=self.toposim_widgets['u'].value * self.u_rate
        )
    
    def capture_and_process_image(self):
        self.image_recorder.recording = False
        self.image_recorder.recording = True
        raw_img = self.image_recorder.image.value
        if not len(raw_img) or raw_img is None:
            return False
        
        self._snapshot_step = self._step

        arr = np.asarray(self.image_recorder.image.value, np.uint8)
        img_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        # crop and resize from webcam 640x480 (4:3)
        # to LEM grid grid_size x grid_size (1:1)
        width = int(self.grid_size * 4 / 3)
        height = self.grid_size
        offset = (width - height) // 2
        img_resized = cv2.resize(img_np, [width, height])[:, offset:offset+height]

        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Process only one face
        if len(faces) != 1:
            return False

        # Pad rectangle
        pad_xy = 20, 35
        rect = faces[0]
        rect[0] -= pad_xy[0]
        rect[1] -= pad_xy[1]
        rect[2] += pad_xy[0]
        rect[3] += pad_xy[1]

        # two passes grabcut for background delineation
        mask = np.zeros(img_resized.shape[:2], dtype=np.uint8)
        mask2 = np.zeros(img_resized.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_resized, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)
            cv2.grabCut(img_resized, mask, None, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)
        except Exception:
            # TODO: https://stackoverflow.com/questions/7546083/grabcut-bgdmodel-fgdmodel-empty-assertion-error
            return False

        mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')
        masked = gray * mask
        
        self.image_to_uplift_rate(masked)
        
        return True
    
    def image_to_uplift_rate(self, im):
        sobel = filters.sobel(im)
        if self.snapshot:
            self.u_rate = im / 255.0 * sobel
        else:
            self.u_rate = 0.2 * self.u_rate + 0.8 * im / 255.0 * sobel
    
    def capture_and_process_image_retry(self, n=1):
        for _ in range(n):
            success = self.capture_and_process_image()
            if success:
                return True
            else:
                time.sleep(0.05)
        
        return False
    
    def hold_uplift_or_relax(self):
        if self._step == self._snapshot_step + 40:
            self.u_rate[:] = 0.0
    
    def save_canvas(self):
        if not os.path.exists('snapshots'):
            os.makedirs('snapshots')
        path = uniquify(f"snapshots/{self.save_text.value}.png")
        self.canvas.to_file(path)
    
    def initialize(self):
        self.toposim.initialize()
        
        if self.particles:
            self.particles.initialize()

        self.redraw()

    def run(self):
        while self._running:
            if self.save:
                if self._save_step == self._step:
                    self.save_canvas()
                    self.save = False
                    self.canvas.sync_image_data = False
                else:
                    self.canvas.sync_image_data = True
                    self._save_step = self._step + 1

            if self.record:
                success = self.capture_and_process_image_retry()
                self.image_recorder.recording = False
                if self.snapshot:
                    #if not success:
                    #    # avoid capturing old image
                    #    self.u_rate[:] = 0.0
                    self.record = False
            else:
                self.hold_uplift_or_relax()

            self.set_erosion_params()
            self.toposim.run_step()
            
            if self.particles:
                self.particles.run_step()

            self.redraw()
            self._step += 1

        self.play_widgets['stop'].description = "Reset"
        self.play_widgets['stop'].icon = "retweet"

    def toggle_disabled(self):
        for w in self.play_widgets.values():
            w.disabled = not w.disabled
            
        self.save_button.disabled = not self.save_button.disabled

        if self.particles:
            w = self.particles_widgets['size']
            w.disabled = not w.disabled

    def start(self, b):
        self.process = Thread(target=self.run)
        self._running = True
        self.process.start()
        self.toggle_disabled()

    def stop(self, b):
        self._running = False
        self.process.join()
        self.reset()
        self.toggle_disabled()
        self.record = False
        self.image_recorder.recording = False

    def reset_particle(self, b):
        self.particles.reset()
        
    def reset(self):
        self._step = 0
        self._snapshot_step = 0
        self._save_step = 0
        self.u_rate[:] = 0.0

        self.toposim.reset()
        
        if self.particles:
            self.particles.reset()

        self.redraw()

        self.play_widgets['stop'].description = "Stop/Reset"
        self.play_widgets['stop'].icon = "stop"

    def redraw(self):
        self.draw_topography()
        
        if self.particles:
            self.draw_particles()
        
    def draw_topography(self):
        with hold_canvas(self.canvas[0]):
            self.canvas[0].save()
            self.canvas[0].scale(self.scale)
            self.canvas[0].clear()
            self.canvas[0].put_image_data(
                self.toposim.shaded_topography, 0, 0
            )
            self.canvas[0].restore()

    def draw_particles(self):
        x, y = self.particles.positions

        with hold_canvas(self.canvas[1]):
            self.canvas[1].clear()
            self.canvas[1].global_alpha = 0.25
            self.canvas[1].fill_style = self.particles_color
            self.canvas[1].fill_rects(x, y, self.particles.sizes)

    def show(self):
        self.initialize()

        return self.main_box
    
    def __del__(self):
        self.camera.close()
        self.image_recorder.close()
        super().__del__()