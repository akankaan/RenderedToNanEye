import pyvista as pv
from PIL import Image

plotter = pv.Plotter(window_size=(320, 320))

cube1 = pv.Cube(center=(1.2, 0, 0.3), x_length=0.6, y_length=0.6, z_length=0.6)  # Left cube
cube2 = pv.Cube(center=(2.4, 0, 0.3), x_length=0.6, y_length=0.6, z_length=0.6)  # Right cube (gap between them)
floor = pv.Plane(center=(0, 0, 0), i_size=15, j_size=15)  # Floor plane

# Enable shadows and configure a directional light so the cubes cast shadows on the floor.
plotter.enable_shadows()
light = pv.Light(position=(5, 5, 10), focal_point=(1.5, 0, 0.3))
# Tone down the brightness by lowering intensity and using softer light colors
light.intensity = 0.35
light.diffuse_color = (0.8, 0.8, 0.8)
light.ambient_color = (0.15, 0.15, 0.15)
plotter.add_light(light)

# Render in grayscale by using gray colors and converting the final screenshot
plotter.add_mesh(cube1, style="surface", color="sandybrown")
plotter.add_mesh(cube2, style="surface", color="sandybrown")
plotter.add_mesh(floor, color="lightblue")

plotter.camera_position = [
    (2.4, 2, 0.1),   # camera position 
    (1.5, 0, 0.3),   # focal point (directly below in z, same x/y for straight-down view)
    (0, 1, 0),     # up direction (along y-axis to keep orientation level)
]

plotter.show()

screenshot_path = '/Users/KaanAkan/Codework/Rendered-to-NanEye/frame_test2.png'
plotter.screenshot(screenshot_path)

# Convert the rendered image to grayscale (L mode) and save.
gray_path = '/Users/KaanAkan/Codework/Rendered-to-NanEye/frame_test2_gray.png'
Image.open(screenshot_path).convert('L').save(gray_path)
print(f"Saved grayscale image: {gray_path}")

