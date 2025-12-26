from dataclasses import dataclass, field
import glfw
from numba import njit, prange
import numpy as np
import moderngl
import time

@dataclass
class Camera:
    position_center: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 20.0, 25.0], dtype=np.float32)  
    )
    rotation: np.ndarray = field(
        default_factory=lambda: np.array([20.0, 45.0, 0.0], dtype=np.float32)  
    )
    fov: float = 80.0  
    distance: float = 50.0  

    # Returns the current position of the camera [x, y, z]
    def get_position(self) -> np.ndarray: # Calculate the camera position based on the orientation, distance and center
        
        pitch = np.radians(self.rotation[0])
        yaw = np.radians(self.rotation[1])   

        x = self.distance * np.cos(pitch) * np.sin(yaw)
        y = self.distance * np.sin(pitch)
        z = self.distance * np.cos(pitch) * np.cos(yaw)

        return self.position_center + np.array([x, y, z], dtype=np.float32)

    def get_view_matrix(self) -> np.ndarray:
        cam_pos = self.get_position()
        target = self.position_center
        world_up = np.array([0, 1, 0], dtype=np.float32)
        
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        view = np.eye(4, dtype=np.float32)
        
        # Rotation part
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        
        # Translation part - ALL NEGATIVE
        view[0, 3] = -np.dot(right, cam_pos)
        view[1, 3] = -np.dot(up, cam_pos)
        view[2, 3] = -np.dot(-forward, cam_pos)  # Double negative = positive np.dot(forward, cam_pos)
        
        # Bottom row should be [0, 0, 0, 1] - already set by np.eye()
        
        return view.T.flatten()

    def get_projection_matrix(self, width, height) -> np.ndarray: #Returns 4x4 perspective projection matrix as flattened array for OpenGL.
        
        aspect_ratio = width / height
        fov_rad = np.radians(self.fov)  # Field of view in radians
        near = 0.01     # Near clipping plane (minimum render distance)
        far = 500.0    # Far clipping plane (maximum render distance)
        focal_len = 1.0 / np.tan(fov_rad / 2.0)  # Calculate focal length (controls zoom)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = focal_len / aspect_ratio
        proj[1, 1] = focal_len
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        
        return proj.T.flatten()

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
    keys_held: set = field(default_factory=set)
    mouse_pressed: bool = False
    scroll_delta: float = 0.0

@dataclass
class State:
    cam: Camera = field(default_factory=Camera)
    window_state: WindowState = field(default_factory=WindowState)
    input_state: InputState = field(default_factory=InputState)
    ctx: moderngl.Context = field(default=None)

    def update(self, dt: float = 0.016) -> None:

        keys = self.input_state.keys_held

        if abs(self.input_state.scroll_delta) > 0:
            self.cam.distance -= self.input_state.scroll_delta * 1.0
            self.cam.distance = np.clip(self.cam.distance, 1.0, 300.0)

        if self.input_state.mouse_pressed:
            self.cam.rotation[0] += self.input_state.mouse_delta[1] * 0.2 # Pitch
            self.cam.rotation[1] += self.input_state.mouse_delta[0] * 0.2 # Yaw
            self.cam.rotation[0] = np.clip(self.cam.rotation[0], -89.0, 89.0) # Prevent flipping
            self.cam.rotation[1] = self.cam.rotation[1] % 360.0 # Wrap around

        if glfw.KEY_W in keys:
            forward = self.cam.position_center - self.cam.get_position()
            forward /= np.linalg.norm(forward)
            self.cam.position_center += forward * dt * 20.0

        if glfw.KEY_S in keys:
            forward = self.cam.position_center - self.cam.get_position()
            forward /= np.linalg.norm(forward)
            self.cam.position_center -= forward * dt * 20.0

        if glfw.KEY_A in keys:
            forward = self.cam.position_center - self.cam.get_position()
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
            right /= np.linalg.norm(right)
            self.cam.position_center -= right * dt * 20.0

        if glfw.KEY_D in keys:
            forward = self.cam.position_center - self.cam.get_position()
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
            right /= np.linalg.norm(right)
            self.cam.position_center += right * dt * 20.0

        # Reset deltas
        self.input_state.mouse_delta[:] = 0
        self.input_state.scroll_delta = 0

def create_compute_shader() -> str:
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
    
def create_vertex_shader() -> str:
    VERTEX_SHADER = """
    #version 330

    in vec4 in_position;

    uniform mat4 view;
    uniform mat4 projection;

    out vec3 frag_pos;

    void main() {
        frag_pos = in_position.xyz;
        gl_Position = projection * view * vec4(in_position.xyz, 1.0);
        gl_PointSize = 0.5; 
    }
    """
    return VERTEX_SHADER

def create_fragment_shader() -> str:
    FRAGMENT_SHADER = """
    #version 330

    in vec3 frag_pos;
    out vec4 fragColor;

    void main() {
        // BRIGHT RED - impossible to miss
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
    """
    return FRAGMENT_SHADER


def glfw_init(state_global: State,title: str = "Lorenz Attractor") -> None:  # Initializes GLFW and sets callbacks

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE) # Allow resizing

    state = state_global
    
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    window = glfw.create_window(mode.size.width, mode.size.height, title, monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    state.ctx = moderngl.create_context()         # Create ModernGL context
    state.ctx.enable(moderngl.DEPTH_TEST)
    state.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    # Set GLFW callbacks
    def framebuffer_size_callback(window, width, height) -> None: # Sets the viewport to the current framebuffer size
        if width == 0 or height == 0:
            return
        state.ctx = moderngl.get_context()
        state.ctx.viewport = (0, 0, width, height)

    def key_callback(window, key, scancode, action, mods) -> None:
        """
        Handles key presses:
        - F11: toggle fullscreen
        - ESC: switch to windowed mode with a smaller default size
        """

        # Track all key presses/releases
        if action == glfw.PRESS:
            state.input_state.keys_held.add(key)
        elif action == glfw.RELEASE:
            state.input_state.keys_held.discard(key)
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            if state.window_state.is_fullscreen:
                # Set default windowed position and size
                state.window_state.windowed_pos = (100, 100)
                state.window_state.windowed_size = (1280, 800)
                glfw.set_window_monitor(
                    window,
                    None,             # windowed mode
                    state.window_state.windowed_pos[0],
                    state.window_state.windowed_pos[1],
                    state.window_state.windowed_size[0],
                    state.window_state.windowed_size[1],
                    0
                )
                state.window_state.is_fullscreen = False
    
    def make_mouse_callback(cam) -> None:
        def mouse_callback(window, button, action, mods):
            if button == glfw.MOUSE_BUTTON_1 and action == glfw.PRESS:
                state.input_state.mouse_pressed = True
            elif button == glfw.MOUSE_BUTTON_1 and action == glfw.RELEASE:
                state.input_state.mouse_pressed = False
        return mouse_callback
    
    def make_scroll_callback(cam) -> None:
        def scroll_callback(window, xoffset, yoffset):
            if yoffset > 0:
                state.input_state.scroll_delta = yoffset
            elif yoffset < 0:
                state.input_state.scroll_delta = yoffset
        return scroll_callback

    def make_cursor_pos_callback(cam) -> None:
        def cursor_pos_callback(window, xpos, ypos):
            state.input_state.mouse_delta[0] += xpos - state.input_state.mouse_pos[0] # Update mouse position and delta
            state.input_state.mouse_delta[1] += ypos - state.input_state.mouse_pos[1] # Update mouse position and delta
            state.input_state.mouse_pos[:] = (xpos, ypos)
        return cursor_pos_callback
    
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, make_mouse_callback(state.cam))    # I use this format to keep cam within the scope of the callback
    glfw.set_scroll_callback(window, make_scroll_callback(state.cam))
    glfw.set_cursor_pos_callback(window, make_cursor_pos_callback(state.cam)) 

    # Set initial viewport
    width, height = glfw.get_framebuffer_size(window)
    state.ctx.viewport = (0, 0, width, height)

    return window

def create_inital_points(num_points: int) -> np.ndarray:
    initial_points = np.random.randn(num_points, 4).astype(np.float32)
    initial_points[:, :3] *= 2.0  # Small spread
    initial_points[:, :3] += [1.0, 1.0, 1.0]  # Center near (1,1,1)
    initial_points[:, 3] = 1.0  # w component
    return initial_points

def setup_compute_program(ctx) -> moderngl.ComputeShader:
    compute_program = ctx.compute_shader(create_compute_shader())
    compute_program['dt'] = 0.0001
    compute_program['sigma'] = 10.0
    compute_program['rho'] = 28.0
    compute_program['beta'] = 8.0 / 3.0
    compute_program['steps'] = 50
    return compute_program

def main() -> None:

    state = State()
    window = glfw_init(state)

    # Initialize points near the attractor starting region
    num_points = 1000000
    initial_points = create_inital_points(num_points)

    points_buffer = state.ctx.buffer(initial_points) # Create buffer (used by both compute and render)

    compute_program = setup_compute_program(state.ctx)

    render_program = state.ctx.program(
        vertex_shader=create_vertex_shader(),
        fragment_shader=create_fragment_shader()
    )
    # Create VAO (Vertex Array Object)
    vao = state.ctx.vertex_array(
        render_program,
        [(points_buffer, '4f', 'in_position')]  # 4 floats matching vec4
    )
    frame_count = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()
        state.update()

        points_buffer.bind_to_storage_buffer(0) # Bind buffer for compute shader
        
        # Run compute shader (update Lorenz system)
        compute_program.run(group_x=(num_points + 255) // 256)
        
        state.ctx.clear(0.0, 0.0, 0.2, 1.0)  # Clear screen
        
        # Get matrices from camera
        width, height = glfw.get_framebuffer_size(window)
        view_matrix = state.cam.get_view_matrix()
        proj_matrix = state.cam.get_projection_matrix(width, height)
        
        # Set uniforms for rendering
        render_program['view'].write(view_matrix)
        render_program['projection'].write(proj_matrix)
        
        # Render points
        vao.render(moderngl.POINTS)

        glfw.swap_buffers(window)
        frame_count += 1
        time.sleep(0.01)  # Small delay to limit frame rate

    glfw.terminate()
    
if __name__ == "__main__":
    main()