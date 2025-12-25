from dataclasses import dataclass, field
import glfw
from numba import njit, prange
import numpy as np
import moderngl


@dataclass
class Camera: # Simple camera class to manage position, rotation, fov, and distance
    position_center: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    rotation: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    fov: float = 90.0
    distance: float = 1.0     # distance from the camera to the view center

    def zoom(self, zoom_amount: float) -> None: # Multiplicative change in fov as well as clamping
        self.fov *= zoom_amount
        if self.fov < 10.0:
            self.fov = 10.0
        if self.fov > 180.0:
            self.fov = 180.0

    # Returns the current position of the camera [x, y, z]
    def get_position(self): # Calculate the camera position based on the orientation, distance and center
        
        pitch = np.radians(self.rotation[0])
        yaw = np.radians(self.rotation[1])   

        x = self.distance * np.cos(pitch) * np.sin(yaw)
        y = self.distance * np.sin(pitch)
        z = -self.distance * np.cos(pitch) * np.cos(yaw)

        return self.position_center + np.array([x, y, z], dtype=np.float32)

@dataclass
class WindowState: # Tracks the window state for fullscreen toggling
    is_fullscreen: bool = True
    windowed_pos: tuple = (100, 100)
    windowed_size: tuple = (1280, 800)

@dataclass
class InputState: # Tracks the state of the input devices
    mouse_pos: np.ndarray = field(
        default_factory=lambda: np.array([0, 0], dtype=np.float32)
    )
    mouse_delta: np.ndarray = field(
        default_factory=lambda: np.array([0, 0], dtype=np.float32)
    )
    mouse_pressed: bool = False
    scroll_delta: float = 0.0


@dataclass
class State:
    cam: Camera = field(default_factory=Camera)
    window_state: WindowState = field(default_factory=WindowState)
    input_state: InputState = field(default_factory=InputState)

    def update(self, dt: float = 0.016) -> None: # Update the camera position and rotation
        self.cam.distance -= dt * 0.1 * self.input_state.scroll_delta
        if self.cam.distance < 0.0:
            self.cam.distance = 0.0
        self.cam.rotation[0] += dt * 0.1 * self.input_state.mouse_delta[1]
        self.cam.rotation[1] += dt * 0.1 * self.input_state.mouse_delta[0]
        self.cam.rotation[0] = np.clip(self.cam.rotation[0], -89.0, 89.0)

        # Reset input deltas after processing
        self.input_state.mouse_delta[:] = 0
        self.input_state.scroll_y_delta = 0

# returns the 3d vector of direction of a point
def lorenz_map(x: float, y: float, z: float, sigma: float = 10.0, 
               rho: float = 28.0, beta: float = 8.0/3.0) -> np.ndarray:
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=np.float32)

@njit(parallel=True)
def lorenz_system(points: np.ndarray, dt: float = 0.001, sigma: float = 10.0, 
                  rho: float = 28.0, beta: float = 8.0/3.0, steps: int = 50) -> np.ndarray:

    points_local = points.copy()

    for point in prange(points_local.shape[0]):
        for _ in range(steps):
            dx = sigma * (points_local[point, 1] - points_local[point, 0])
            dy = points_local[point, 0] * (rho - points_local[point, 2]) - points_local[point, 1]
            dz = points_local[point, 0] * points_local[point, 1] - beta * points_local[point, 2]
            points_local[point, 0] += dx * dt
            points_local[point, 1] += dy * dt
            points_local[point, 2] += dz * dt

    return points_local

    
# Initializes GLFW and sets callbacks
def glfw_init(title: str = "Lorenz Attractor"): 

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # Request OpenGL context compatible with ModernGL
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE) # Allow resizing

    # Fullscreen resolution
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    window = glfw.create_window(mode.size.width, mode.size.height, title, monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Set GLFW callbacks
    state = State()
    cam = state.cam
    window_state = state.window_state
    input_state = state.input_state
    def framebuffer_size_callback(window, width, height): # Sets the viewport to the current framebuffer size
        if width == 0 or height == 0:
            return

        ctx = moderngl.get_context()
        ctx.viewport = (0, 0, width, height)
    def key_callback(window, key, scancode, action, mods):
        """
        Handles key presses:
        - F11: toggle fullscreen
        - ESC: switch to windowed mode with a smaller default size
        """
        
        def toggle_fullscreen(window): # Toggles between fullscreen and windowed mode

            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)

            if window_state.is_fullscreen:
                # Save windowed size and position
                window_state.windowed_pos = glfw.get_window_pos(window)
                window_state.windowed_size = glfw.get_window_size(window)

                # Switch to windowed mode
                glfw.set_window_monitor(
                    window,
                    None,                        # windowed
                    window_state.windowed_pos[0],
                    window_state.windowed_pos[1],
                    window_state.windowed_size[0],
                    window_state.windowed_size[1],
                    0
                )
                window_state.is_fullscreen = False

            else:
                # Switch to fullscreen
                glfw.set_window_monitor(
                    window,
                    monitor,
                    0,
                    0,
                    mode.size.width,
                    mode.size.height,
                    mode.refresh_rate
                )
                window_state.is_fullscreen = True

        if key == glfw.KEY_F11 and action == glfw.PRESS:
            toggle_fullscreen(window)

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            if window_state.is_fullscreen:
                # Set default windowed position and size
                window_state.windowed_pos = (100, 100)
                window_state.windowed_size = (1280, 800)
                glfw.set_window_monitor(
                    window,
                    None,             # windowed mode
                    window_state.windowed_pos[0],
                    window_state.windowed_pos[1],
                    window_state.windowed_size[0],
                    window_state.windowed_size[1],
                    0
                )
                window_state.is_fullscreen = False
    def make_mouse_callback(cam):

        def mouse_callback(window, button, action, mods):
            if button == glfw.MOUSE_BUTTON_1 and action == glfw.PRESS:
                input_state.mouse_pressed = True
                print("Mouse button left pressed")
                # TODO: Implement mouse position tracking here
            elif button == glfw.MOUSE_BUTTON_1 and action == glfw.RELEASE:
                input_state.mouse_pressed = False
                print("Mouse button left released")

        return mouse_callback
    def make_scroll_callback(cam):

        def scroll_callback(window, xoffset, yoffset):
            if yoffset > 0:
                state.input_state.scroll_delta = yoffset
                print("Mouse scrolled up")
            elif yoffset < 0:
                state.input_state.scroll_delta = yoffset
                print("Mouse scrolled down")

        return scroll_callback
    def make_cursor_pos_callback(cam):
        def cursor_pos_callback(window, xpos, ypos):
            print(f"Mouse cursor at ({xpos}, {ypos})")
            # Update mouse position and delta
            inp = state.input_state
            inp.mouse_delta[0] += xpos - inp.mouse_pos[0]
            inp.mouse_delta[1] += ypos - inp.mouse_pos[1]
            inp.mouse_pos[:] = (xpos, ypos)
        return cursor_pos_callback
    
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, make_mouse_callback(cam))    # I use this format to keep cam within the scope of the callback
    glfw.set_scroll_callback(window, make_scroll_callback(cam))
    glfw.set_cursor_pos_callback(window, make_cursor_pos_callback(cam)) 

    # Create ModernGL context
    ctx = moderngl.create_context()

    # Set initial viewport
    width, height = glfw.get_framebuffer_size(window)
    ctx.viewport = (0, 0, width, height)

    return window, ctx, state

def main():

    # Starting parameters
    num_points = 1000
    buffer_len = 100

    # Initialize GLFW and create a window along with the global variables
    global is_fullscreen, windowed_pos, windowed_size
    is_fullscreen = True
    windowed_pos = (100, 100)
    windowed_size = (1280, 800)
    window, ctx, state = glfw_init()

    # Initialize the beginning state of the system
    tail_coords = np.zeros((buffer_len, num_points, 3), dtype=np.float32)
    tail_coords[0] = np.zeros((num_points, 3), dtype=np.float32)

    # tail_coords is a ring buffer of shape (buffer_length, num_points, 3)
    head = 0  # head points to where the NEXT value will be written
    current = tail_coords[0]
    
    # Main loop
    while not glfw.window_should_close(window):

        glfw.poll_events()
        state.update()

        # Just clear the screen for now
        ctx.clear(0.2, 0.2, 0.5, 1.0)

        glfw.swap_buffers(window)

    glfw.terminate()
    
if __name__ == "__main__":
    main()