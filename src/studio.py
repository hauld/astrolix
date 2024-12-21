from moviepy import *
from manim import *
from animate_flap import BirdAnimation
import os
import time

# Render the scene using the imported class
def render_bird_animation(media_dir="media"):
    file_path = os.path.join(media_dir, 'videos', '480p15', 'BirdAnimation_ManimCE_v0.18.1.gif')
    if not os.path.exists(file_path):
        # Ensure the format and quality settings match the command-line options
        config["quality"] = "low_quality"
        config["format"] = "gif"
        config["transparent"] = True
        config["media_dir"] = media_dir
        scene = BirdAnimation()  # Instantiate the imported scene class
        scene.render()  # Render the scene
        return file_path
    else:
        return file_path
# Render scene
def render_scene(media_dir, backdrop, clip, duration=10, position="center"):
    # Load the background (image or video)
    background_path = backdrop  # Or use "background_video.mp4" for video
    background = ImageClip(background_path, duration=duration) # Set the duration to match the animation

    # Load the bird animation video
    animation_clip = VideoFileClip(clip, has_mask=True) # Use the appropriate file and time range

    # Resize the bird animation to fit the background (optional)
    animation_clip = animation_clip.resized(width=500)  # Resize to fit on the background

    # Repeat the GIF x times
    times = int(duration / animation_clip.duration)
    repeated_animation = concatenate_videoclips([animation_clip] * times)

    # Overlay the animation on the background
    final_video = CompositeVideoClip([background, repeated_animation.with_position(position)], use_bgclip=True  )

    # Export the final video
    video_path = f"{media_dir}/final_video_{time.time()}.mp4"
    final_video.write_videofile(video_path, fps=24, codec='libx264', preset='slow', threads=4)
    return video_path

if __name__ == "__main__":
    clip_path = render_bird_animation()  # Call the render function
    backdrop_path="images/backdrop_image_1734197177.597732.jpg"
    render_scene(backdrop_path, clip_path)