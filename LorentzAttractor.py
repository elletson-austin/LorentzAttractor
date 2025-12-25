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


def get_view_matrix(self): # Returns the view matrix for the current camera state

    # Get camera position in world space and what we're looking at
    cam_pos = self.cam.get_position()      # Where camera is
    target = self.cam.position_center       # What camera looks at (Lorenz center)
    world_up = np.array([0, 1, 0], dtype=np.float32)  # World's up direction (Y-axis)
    
    # ===== Build camera's coordinate system (3 perpendicular vectors) =====
    
    # Forward: direction from camera to target (normalized)
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    # Right: perpendicular to both forward and world up (normalized)
    # Cross product: forward × world_up = right (camera's X-axis)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    
    # Up: perpendicular to both right and forward
    # Cross product: right × forward = up (camera's actual Y-axis)
    up = np.cross(right, forward)
    # Already normalized because right and forward are normalized and perpendicular
    
    # ===== Build 4x4 view matrix =====
    # Layout:
    # ┌                           ┐
    # │ Rx  Ry  Rz  Tx │  Right vector + X translation
    # │ Ux  Uy  Uz  Ty │  Up vector + Y translation
    # │ Fx  Fy  Fz  Tz │  Forward vector + Z translation
    # │ 0   0   0   1  │  Homogeneous coordinate
    # └                           ┘
    
    view = np.eye(4, dtype=np.float32)  # Start with identity matrix
    
    # Top-left 3x3: Rotation (camera basis vectors)
    view[0, :3] = right        # Row 0: Right vector (Rx, Ry, Rz)
    view[1, :3] = up           # Row 1: Up vector (Ux, Uy, Uz)
    view[2, :3] = -forward     # Row 2: Forward vector (Fx, Fy, Fz)
                                # Negative because OpenGL looks down -Z axis
    
    # Right column (first 3 rows): Translation
    # Move world so camera is at origin
    # Dot products project camera position onto each camera axis
    view[0, 3] = -np.dot(right, cam_pos)     # Tx: Translation along camera's X
    view[1, 3] = -np.dot(up, cam_pos)        # Ty: Translation along camera's Y
    view[2, 3] = np.dot(forward, cam_pos)    # Tz: Translation along camera's Z
    
    # Bottom row [0, 0, 0, 1] already set by np.eye()
    
    # OpenGL uses column-major order, so transpose before flattening
    return view.T.flatten()

def get_projection_matrix(self, aspect_ratio):
    """
    Returns 4x4 perspective projection matrix as flattened array for OpenGL.
    Creates the "things far away look smaller" effect.
    
    aspect_ratio: width / height (prevents stretching)
    """
    
    # Convert FOV to radians and set clipping planes
    fov_rad = np.radians(self.fov)  # Field of view in radians
    near = 0.1     # Near clipping plane (minimum render distance)
    far = 500.0    # Far clipping plane (maximum render distance)
    
    # Calculate focal length (controls zoom)
    f = 1.0 / np.tan(fov_rad / 2.0)
    
    # ===== Build 4x4 projection matrix =====
    # Layout:
    # ┌                                      ┐
    # │ f/aspect   0         0          0    │  Row 0: Scale X (adjusted for aspect)
    # │ 0          f         0          0    │  Row 1: Scale Y (based on FOV)
    # │ 0          0    (f+n)/(n-f)    -1    │  Row 2: Map Z depth & trigger perspective
    # │ 0          0    2*f*n/(n-f)     0    │  Row 3: Perspective divide value
    # └                                      ┘
    # Where: f = far plane, n = near plane
    
    proj = np.zeros((4, 4), dtype=np.float32)
    
    # Row 0, Col 0: X-coordinate scaling (adjusted for aspect ratio)
    # Wider screens (aspect > 1) need less X scaling to prevent horizontal stretching
    # Taller screens (aspect < 1) need more X scaling to prevent horizontal squishing
    proj[0, 0] = f / aspect_ratio
    
    # Row 1, Col 1: Y-coordinate scaling (based purely on FOV)
    # This creates the vertical field of view
    # Larger f (smaller FOV) = more zoom = larger scale factor
    proj[1, 1] = f
    
    # Row 2, Col 2: Z-coordinate depth mapping (non-linear)
    # Maps camera space [near, far] to clip space [-1, 1] for depth buffer
    # This preserves more precision for nearby objects (important for z-fighting)
    proj[2, 2] = (far + near) / (near - far)
    
    # Row 2, Col 3: Perspective divide trigger (THE KEY TO PERSPECTIVE!)
    # This copies -z into the w component
    # After matrix multiply: w' = -z
    # GPU then divides (x', y', z') by w' = -z
    # Result: things at z=10 shrink 10×, things at z=50 shrink 50×
    # We use -z (not distance) because perspective is about DEPTH, not radial distance
    proj[2, 3] = -1.0
    
    # Row 3, Col 2: Z-coordinate offset for depth mapping
    # Works with proj[2,2] to create the non-linear depth mapping
    # This value ends up in the z' component after multiplication
    proj[3, 2] = (2.0 * far * near) / (near - far)
    
    # All other values stay 0 (set by np.zeros)
    # Notably proj[3, 3] = 0 (not 1!) because we want w' = -z, not w' = -z + 1
    
    # OpenGL uses column-major order, so transpose before flattening
    # This converts our row-major numpy array to OpenGL's expected format
    return proj.T.flatten()

def create_compute_shader():
    COMPUTE_SHADER = """
    #version 430

    layout(local_size_x = 256) in;

    layout(std430, binding = 0) buffer PointsBuffer {
        vec4 points[];
    };

    uniform float dt;
    uniform float sigma;
    uniform float rho;
    uniform float beta;
    uniform int steps;

    void main() {
        uint idx = gl_GlobalInvocationID.x;
        if (idx >= points.length()) return;
        
        vec3 p = points[idx].xyz;
        
        // Integrate Lorenz system for 'steps' iterations
        for (int i = 0; i < steps; i++) {
            float dx = sigma * (p.y - p.x);
            float dy = p.x * (rho - p.z) - p.y;
            float dz = p.x * p.y - beta * p.z;
            
            p.x += dx * dt;
            p.y += dy * dt;
            p.z += dz * dt;
        }
        
        points[idx].xyz = p;
    }
    """
    return COMPUTE_SHADER
    
def create_vertex_shader():
    VERTEX_SHADER = """
    #version 330

    in vec3 in_position;

    uniform mat4 view;
    uniform mat4 projection;

    out vec3 frag_pos;

    void main() {
        frag_pos = in_position;
        gl_Position = projection * view * vec4(in_position, 1.0);
    }
    """
    return VERTEX_SHADER

def create_fragment_shader():
    
    FRAGMENT_SHADER = """
    #version 330

    in vec3 frag_pos;
    out vec4 fragColor;

    void main() {
        // Color based on position
        vec3 color = vec3(
            0.5 + 0.5 * sin(frag_pos.z * 0.05),
            0.5 + 0.5 * cos(frag_pos.x * 0.05),
            0.8
        );
        fragColor = vec4(color, 1.0);
    }
    """
    return FRAGMENT_SHADER


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

def create_inital_points(num_points: int) -> np.ndarray:
    initial_points = np.random.randn(num_points, 4).astype(np.float32)
    initial_points[:, :3] *= 2.0  # Small spread
    initial_points[:, :3] += [1.0, 1.0, 1.0]  # Center near (1,1,1)
    initial_points[:, 3] = 1.0  # w component
    return initial_points

def setup_compute_program(ctx):
    compute_program = ctx.compute_shader(create_compute_shader())
    compute_program['dt'] = 0.001
    compute_program['sigma'] = 10.0
    compute_program['rho'] = 28.0
    compute_program['beta'] = 8.0 / 3.0
    compute_program['steps'] = 100
    return compute_program

def setup_render_program():
    render_program = ctx.program(
        vertex_shader=create_vertex_shader(),
        fragment_shader=create_fragment_shader()
    )
    return render_program
def main():

    # Starting parameters
    buffer_len = 1000

    window, ctx, state = glfw_init()

    
    # Initialize points near the attractor starting region
    num_points = 10000
    initial_points = create_inital_points(num_points)
    
    # Create buffer (used by both compute and render)
    points_buffer = ctx.buffer(initial_points)

    # Create compute shader program
    compute_program = setup_compute_program(ctx)
    render_program = setup_render_program(ctx)
    # Create render program
    render_program = ctx.program(
        vertex_shader=create_vertex_shader(),
        fragment_shader=create_fragment_shader()
    )
    # Create VAO (Vertex Array Object)
    vao = ctx.vertex_array(
        render_program,
        [(points_buffer, '3f 1f', 'in_position')]  # 3 floats for xyz, 1 for w
    )
    
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