from dataclasses import dataclass, field
import glfw
from numba import njit, prange
import numpy as np
import moderngl
import time

@dataclass
class Camera:
    position_center: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 20.0, 25.0], dtype=np.float32)  # ✓ Lorenz center
    )
    rotation: np.ndarray = field(
        default_factory=lambda: np.array([20.0, 45.0, 0.0], dtype=np.float32)  # ✓ Good angle
    )
    fov: float = 130.0  # ✓ Narrower FOV
    distance: float = 300.0  # ✓ Farther back

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
        z = self.distance * np.cos(pitch) * np.cos(yaw)

        return self.position_center + np.array([x, y, z], dtype=np.float32)

    def get_view_matrix(self):
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
        focal_len = 1.0 / np.tan(fov_rad / 2.0)  # Calculate focal length (controls zoom)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = focal_len / aspect_ratio
        proj[1, 1] = focal_len
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        
        # Notably proj[3, 3] = 0 (not 1!) because we want w' = -z, not w' = -z + 1
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
    mouse_pressed: bool = False
    scroll_delta: float = 0.0

@dataclass
class State:
    cam: Camera = field(default_factory=Camera)
    window_state: WindowState = field(default_factory=WindowState)
    input_state: InputState = field(default_factory=InputState)
    ctx: moderngl.Context = field(default=None)

    def update(self, dt: float = 0.016) -> None:
        if abs(self.input_state.scroll_delta) > 0:
            self.cam.distance -= self.input_state.scroll_delta * 10.0
            self.cam.distance = np.clip(self.cam.distance, 10.0, 300.0)
        
        # ONLY rotate when mouse button is held
        if self.input_state.mouse_pressed:
            self.cam.rotation[0] += self.input_state.mouse_delta[1] * 0.2
            self.cam.rotation[1] += self.input_state.mouse_delta[0] * 0.2
            self.cam.rotation[0] = np.clip(self.cam.rotation[0], -89.0, 89.0)

        # Reset deltas
        self.input_state.mouse_delta[:] = 0
        self.input_state.scroll_delta = 0

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

def create_fragment_shader():
    
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


# Initializes GLFW and sets callbacks
def glfw_init(state_global: State,title: str = "Lorenz Attractor"): 

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # Request OpenGL context compatible with ModernGL
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE) # Allow resizing

    state = state_global
    cam = state.cam
    window_state = state.window_state
    input_state = state.input_state
    
    # Fullscreen resolution
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    window = glfw.create_window(mode.size.width, mode.size.height, title, monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    state.ctx = moderngl.create_context() # Create ModernGL context
    state.ctx.enable(moderngl.DEPTH_TEST)
    state.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    # Set GLFW callbacks
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
            #print(f"Mouse cursor at ({xpos}, {ypos})")
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

def setup_compute_program(ctx):
    compute_program = ctx.compute_shader(create_compute_shader())
    compute_program['dt'] = 0.0001
    compute_program['sigma'] = 11.0
    compute_program['rho'] = 29.0
    compute_program['beta'] = 8.0 / 5.0
    compute_program['steps'] = 50
    return compute_program

def main():

    # Starting parameters
    state = State()
    window = glfw_init(state)

    # Initialize points near the attractor starting region
    num_points = 100000
    initial_points = create_inital_points(num_points)
    
    print("=== INITIAL SETUP ===")
    print(f"Initial points range:")
    print(f"  X: [{initial_points[:,0].min():.2f}, {initial_points[:,0].max():.2f}]")
    print(f"  Y: [{initial_points[:,1].min():.2f}, {initial_points[:,1].max():.2f}]")
    print(f"  Z: [{initial_points[:,2].min():.2f}, {initial_points[:,2].max():.2f}]")

    # Create buffer (used by both compute and render)
    points_buffer = state.ctx.buffer(initial_points)

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

        # Bind buffer for compute shader
        points_buffer.bind_to_storage_buffer(0)
        
        # Run compute shader (update Lorenz system)
        compute_program.run(group_x=(num_points + 255) // 256)
        if frame_count == 1:
            data = np.frombuffer(points_buffer.read(), dtype=np.float32).reshape(-1, 4)
            print(f"\n=== FRAME 1 DEBUG ===")
            print(f"Points after compute:")
            print(f"  X: [{data[:,0].min():.2f}, {data[:,0].max():.2f}]")
            print(f"  Y: [{data[:,1].min():.2f}, {data[:,1].max():.2f}]")
            print(f"  Z: [{data[:,2].min():.2f}, {data[:,2].max():.2f}]")
            
            cam_pos = state.cam.get_position()
            print(f"\nCamera:")
            print(f"  Position: {cam_pos}")
            print(f"  Target: {state.cam.position_center}")
            print(f"  Distance to target: {np.linalg.norm(cam_pos - state.cam.position_center):.2f}")
            print(f"  Rotation: {state.cam.rotation}")
            print(f"  FOV: {state.cam.fov}")
            
            # Test: manually transform one point
            test_point = data[0, :3]
            print(f"\nTest point: {test_point}")
            
            # Get matrices
            width, height = glfw.get_framebuffer_size(window)
            view = state.cam.get_view_matrix().reshape(4, 4)
            proj = state.cam.get_projection_matrix(width/height).reshape(4, 4)
            
            # Transform test point
            p_view = view @ np.append(test_point, 1.0)
            p_clip = proj @ p_view
            p_ndc = p_clip[:3] / p_clip[3]
            
            print(f"  After view: {p_view[:3]} (w={p_view[3]:.2f})")
            print(f"  After proj: {p_clip[:3]} (w={p_clip[3]:.2f})")
            print(f"  NDC: {p_ndc}")
            print(f"  In frustum: {-1 <= p_ndc[0] <= 1 and -1 <= p_ndc[1] <= 1 and -1 <= p_ndc[2] <= 1}")
            
            # Check how many points in frustum
            all_view = (view @ np.column_stack([data[:,:3], np.ones(num_points)]).T).T
            all_clip = (proj @ all_view.T).T
            all_ndc = all_clip[:,:3] / all_clip[:,3:4]
            in_frustum = np.sum(
                (all_ndc[:,0] >= -1) & (all_ndc[:,0] <= 1) &
                (all_ndc[:,1] >= -1) & (all_ndc[:,1] <= 1) &
                (all_ndc[:,2] >= -1) & (all_ndc[:,2] <= 1)
            )
            print(f"\nPoints in view frustum: {in_frustum}/{num_points}")
        
        # Clear screen
        state.ctx.clear(0.1, 0.1, 0.15, .005)
        
        # Get matrices from camera
        width, height = glfw.get_framebuffer_size(window)
        aspect = width / height
        view_matrix = state.cam.get_view_matrix()
        proj_matrix = state.cam.get_projection_matrix(aspect)
        
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