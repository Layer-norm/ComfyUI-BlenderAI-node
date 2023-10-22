import gpu
import ctypes
import numpy as np
import bgl as gl
import bpy
import time
# from OpenGL import GL as gl
from gpu_extras.batch import batch_for_shader
from ..utils import logger
try:
    import imgui
except ModuleNotFoundError:
    print("ERROR: imgui was not found")

from imgui.integrations.base import BaseOpenGLRenderer

class Renderer400(BaseOpenGLRenderer):
    """Integration of ImGui into Blender."""

    VERTEX_SHADER_SRC = """
    uniform mat4 ProjMtx;
    in vec2 Position;
    in vec2 UV;
    in vec4 Color;
    out vec2 Frag_UV;
    out vec4 Frag_Color;

    void main() {
        Frag_UV = UV;
        Frag_Color = Color;
        gl_Position = ProjMtx * vec4(Position.xy, 0, 1);
    }
    """

    FRAGMENT_SHADER_SRC = """
    uniform sampler2D Texture;
    in vec2 Frag_UV;
    in vec4 Frag_Color;
    out vec4 Out_Color;

    vec4 linear_to_srgb(vec4 linear) {
        return mix(
            1.055 * pow(linear, vec4(1.0 / 2.4)) - 0.055,
            12.92 * linear,
            step(linear, vec4(0.00031308))
        );
    }

    vec4 srgb_to_linear(vec4 srgb) {
        return mix(
            pow((srgb + 0.055) / 1.055, vec4(2.4)),
            srgb / 12.92,
            step(srgb, vec4(0.04045))
        );
    }

    void main() {
        Out_Color = Frag_Color * texture(Texture, Frag_UV.st);
        Out_Color.rgba = srgb_to_linear(Out_Color.rgba);
    }
    """
    instance = None

    def __init__(self):
        self._shader_handle = None
        self._vert_handle = None
        self._fragment_handle = None

        self._attrib_location_tex = None
        self._attrib_proj_mtx = None
        self._attrib_location_position = None
        self._attrib_location_uv = None
        self._attrib_location_color = None

        self._vbo_handle = None
        self._elements_handle = None
        self._vao_handle = None
        self._font_tex = None
        Renderer400.instance = self

        super().__init__()

    def refresh_font_texture(self):
        self.refresh_font_texture_ex(self)
        if self.refresh_font_texture_ex not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(self.refresh_font_texture_ex)

    @staticmethod
    @bpy.app.handlers.persistent
    def refresh_font_texture_ex(scene=None):
        # save texture state
        self = Renderer400.instance
        if not (img := bpy.data.images.get(".imgui_font", None)) or img.bindcode == 0:
            ts = time.time()
            width, height, pixels = self.io.fonts.get_tex_data_as_rgba32()
            if not img:
                img = bpy.data.images.new(".imgui_font", width, height, alpha=True, float_buffer=True)
            pixels = np.frombuffer(pixels, dtype=np.uint8) / np.float32(256)
            img.pixels.foreach_set(pixels)
            self.io.fonts.clear_tex_data()
            logger.debug(f"MLT Init -> {time.time() - ts:.2f}s")
        img.gl_load()
        self._font_tex = gpu.texture.from_image(img)
        self._font_texture = img.bindcode
        self.io.fonts.texture_id = self._font_texture

    def _create_device_objects(self):
        self._bl_shader = gpu.types.GPUShader(self.VERTEX_SHADER_SRC, self.FRAGMENT_SHADER_SRC)

    def render(self, draw_data):
        io = self.io
        shader = self._bl_shader

        display_width, display_height = io.display_size
        fb_width = int(display_width * io.display_fb_scale[0])
        fb_height = int(display_height * io.display_fb_scale[1])

        if fb_width == 0 or fb_height == 0:
            return

        draw_data.scale_clip_rects(*io.display_fb_scale)

        ortho_projection = (
            2.0 / display_width, 0.0, 0.0, 0.0,
            0.0, 2.0 / -display_height, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0
        )
        self.refresh_font_texture_ex()
        shader.bind()
        shader.uniform_float("ProjMtx", ortho_projection)
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set("NONE")
        gpu.state.face_culling_set("NONE")
        gpu.state.scissor_test_set(True)
        gpu.state.point_size_set(5.0)
        gpu.state.program_point_size_set(False)
        shader.uniform_sampler("Texture", self._font_tex)
        gpu.state.viewport_set(0, 0, int(fb_width), int(fb_height))
        # shader.uniform_int("Texture", 0)
        for commands in draw_data.commands_lists:
            size = commands.idx_buffer_size * imgui.INDEX_SIZE // 4
            address = commands.idx_buffer_data
            ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_int))
            idx_buffer_np = np.ctypeslib.as_array(ptr, shape=(size,))

            size = commands.vtx_buffer_size * imgui.VERTEX_SIZE // 4
            address = commands.vtx_buffer_data
            ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_float))
            vtx_buffer_np = np.ctypeslib.as_array(ptr, shape=(size,))
            vtx_buffer_shaped = vtx_buffer_np.reshape(-1, imgui.VERTEX_SIZE // 4)

            idx_buffer_offset = 0
            for command in commands.commands:
                # x, y, z, w = command.clip_rect
                # gl.glScissor(int(x), int(fb_height - w), int(z - x), int(w - y))

                vertices = vtx_buffer_shaped[:, :2]
                uvs = vtx_buffer_shaped[:, 2:4]
                colors = vtx_buffer_shaped.view(np.uint8)[:, 4 * 4:]
                colors = colors.astype('f') / 255.0

                indices = idx_buffer_np[idx_buffer_offset:idx_buffer_offset + command.elem_count]
                # logger.error(command.texture_id)
                # shader.uniform_sampler("Texture", self._font_tex)
                # gl.glBindTexture(gl.GL_TEXTURE_2D, command.texture_id)

                # print(dir(command))
                batch = batch_for_shader(shader, 'TRIS', {
                    "Position": vertices,
                    "UV": uvs,
                    "Color": colors,
                }, indices=indices)
                batch.draw(shader)

                idx_buffer_offset += command.elem_count

    def _invalidate_device_objects(self):
        if self._font_texture > -1:
            gl.glDeleteTextures([self._font_texture])
        self.io.fonts.texture_id = 0
        self._font_texture = 0
