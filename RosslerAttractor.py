from dataclasses import dataclass, field
import glfw
import numpy as np
import moderngl
import time

@dataclass
class Camera:
    position_center: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 10.0], dtype=np.float32)  
    )
    rotation: np.ndarray = field(
        default_factory=lambda: np.array([30.0, 45.0, 0.0], dtype=np.float32)  
    )
    fov: float = 60.0  
    distance: float = 40.0  

    def get_position(self):
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
        
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        
        view[0, 3] = -np.dot(right, cam_pos)
        view[1, 3] = -np.dot(up, cam_pos)
        view[2, 3] = -np.dot(-forward, cam_pos)
        
        return view.T.flatten()

    def get_projection_matrix(self, aspect_ratio):
        fov_rad = np.radians(self.fov)
        near = 0.01
        far = 500.0
        focal_len = 1.0 / np.tan(fov_rad / 2.0)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = focal_len / aspect_ratio
        proj[1, 1] = focal_len
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        
        return proj.T.flatten()

@dataclass
class WindowState:
    is_fullscreen: bool = True
    windowed_pos: tuple = (100, 100)
    windowed_size: tuple = (1280, 800)

@dataclass
class InputState:
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
            self.cam.distance = np.clip(self.cam.distance, 5.0, 300.0)

        if self.input_state.mouse_pressed:
            self.cam.rotation[0] += self.input_state.mouse_delta[1] * 0.2
            self.cam.rotation[1] += self.input_state.mouse_delta[0] * 0.2
            self.cam.rotation[0] = np.clip(self.cam.rotation[0], -89.0, 89.0)
        
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
        
        self.input_state.mouse_delta[:] = 0
        self.input_state.scroll_delta = 0

def create_compute_shader():
    COMPUTE_SHADER = """
    #version 430

    layout(local_size_x = 256) in;

    layout(std430, binding = 0) buffer PointsBuffer {
        vec4 points[];
    };

    uniform float dt;
    uniform float a;
    uniform float b;
    uniform float c;
    uniform int steps;

    void main() {
        uint idx = gl_GlobalInvocationID.x;
        if (idx >= points.length()) return;
        
        vec3 p = points[idx].xyz;
        
        // Integrate Rössler system
        for (int i = 0; i < steps; i++) {
            float dx = -p.y - p.z;
            float dy = p.x + a * p.y;
            float dz = b + p.z * (p.x - c);
            
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
        // Color based on height (z-coordinate)
        float t = (frag_pos.z + 5.0) / 25.0;  // Normalize z to [0,1]
        vec3 color = mix(
            vec3(0.1, 0.3, 0.8),  // Blue at bottom
            vec3(0.9, 0.2, 0.3),  // Red at top
            t
        );
        fragColor = vec4(color, 1.0);
    }
    """
    return FRAGMENT_SHADER


def glfw_init(state_global: State, title: str = "Rössler Attractor"):
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

    state = state_global
    
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    window = glfw.create_window(mode.size.width, mode.size.height, title, monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    state.ctx = moderngl.create_context()
    state.ctx.enable(moderngl.DEPTH_TEST)
    state.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    def framebuffer_size_callback(window, width, height):
        if width == 0 or height == 0:
            return
        state.ctx = moderngl.get_context()
        state.ctx.viewport = (0, 0, width, height)
    
    def key_callback(window, key, scancode, action, mods):
        if action == glfw.PRESS:
            state.input_state.keys_held.add(key)
        elif action == glfw.RELEASE:
            state.input_state.keys_held.discard(key)
        
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            if state.window_state.is_fullscreen:
                state.window_state.windowed_pos = (100, 100)
                state.window_state.windowed_size = (1280, 800)
                glfw.set_window_monitor(
                    window,
                    None,
                    state.window_state.windowed_pos[0],
                    state.window_state.windowed_pos[1],
                    state.window_state.windowed_size[0],
                    state.window_state.windowed_size[1],
                    0
                )
                state.window_state.is_fullscreen = False
    
    def make_mouse_callback(cam):
        def mouse_callback(window, button, action, mods):
            if button == glfw.MOUSE_BUTTON_1 and action == glfw.PRESS:
                state.input_state.mouse_pressed = True
            elif button == glfw.MOUSE_BUTTON_1 and action == glfw.RELEASE:
                state.input_state.mouse_pressed = False
        return mouse_callback
    
    def make_scroll_callback(cam):
        def scroll_callback(window, xoffset, yoffset):
            if yoffset > 0:
                state.input_state.scroll_delta = yoffset
            elif yoffset < 0:
                state.input_state.scroll_delta = yoffset
        return scroll_callback
    
    def make_cursor_pos_callback(cam):
        def cursor_pos_callback(window, xpos, ypos):
            state.input_state.mouse_delta[0] += xpos - state.input_state.mouse_pos[0]
            state.input_state.mouse_delta[1] += ypos - state.input_state.mouse_pos[1]
            state.input_state.mouse_pos[:] = (xpos, ypos)
        return cursor_pos_callback
    
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, make_mouse_callback(state.cam))
    glfw.set_scroll_callback(window, make_scroll_callback(state.cam))
    glfw.set_cursor_pos_callback(window, make_cursor_pos_callback(state.cam))

    width, height = glfw.get_framebuffer_size(window)
    state.ctx.viewport = (0, 0, width, height)

    return window

def create_initial_points(num_points: int) -> np.ndarray:
    initial_points = np.zeros((num_points, 4), dtype=np.float32)
    
    # Random positions in a 3D box
    initial_points[:, 0] = np.random.uniform(-10, 10, num_points)  # X: [-10, 10]
    initial_points[:, 1] = np.random.uniform(-10, 10, num_points)  # Y: [-10, 10]
    initial_points[:, 2] = np.random.uniform(0, 20, num_points)    # Z: [0, 20]
    initial_points[:, 3] = 1.0  # w component
    
    return initial_points

def setup_compute_program(ctx):
    compute_program = ctx.compute_shader(create_compute_shader())
    compute_program['dt'] = 0.0005
    compute_program['a'] = 0.2
    compute_program['b'] = 0.2
    compute_program['c'] = 5.7
    compute_program['steps'] = 10
    return compute_program

def main():
    state = State()
    window = glfw_init(state)

    num_points = 1000000
    initial_points = create_initial_points(num_points)

    points_buffer = state.ctx.buffer(initial_points)
    compute_program = setup_compute_program(state.ctx)

    render_program = state.ctx.program(
        vertex_shader=create_vertex_shader(),
        fragment_shader=create_fragment_shader()
    )
    
    vao = state.ctx.vertex_array(
        render_program,
        [(points_buffer, '4f', 'in_position')]
    )
    
    frame_count = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()
        state.update()

        points_buffer.bind_to_storage_buffer(0)
        compute_program.run(group_x=(num_points + 255) // 256)
        
        state.ctx.clear(0.0, 0.0, 0.2, 1.0)
        
        width, height = glfw.get_framebuffer_size(window)
        aspect = width / height
        view_matrix = state.cam.get_view_matrix()
        proj_matrix = state.cam.get_projection_matrix(aspect)
        
        render_program['view'].write(view_matrix)
        render_program['projection'].write(proj_matrix)
        
        vao.render(moderngl.POINTS)

        glfw.swap_buffers(window)
        frame_count += 1
        time.sleep(0.01)

    glfw.terminate()
    
if __name__ == "__main__":
    main()