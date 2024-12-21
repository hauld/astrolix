from manim import *

class BirdAnimation(Scene):
    def construct(self):
        # Create the body of the bird
        body = Ellipse(width=4, height=3.6, color=BLACK, fill_color=BLACK, fill_opacity=1).move_to(ORIGIN)

        # Create the eyes
        left_eye_white = Circle(radius=0.24, color=WHITE, fill_color=WHITE, fill_opacity=1).shift(LEFT * 0.8 + UP * 0.5)
        right_eye_white = Circle(radius=0.24, color=WHITE, fill_color=WHITE, fill_opacity=1).shift(RIGHT * 0.8 + UP * 0.5)

        # Create the pupils
        left_pupil = Circle(radius=0.12, color=BLUE, fill_color=BLUE, fill_opacity=1).shift(LEFT * 0.8 + UP * 0.5)
        right_pupil = Circle(radius=0.12, color=BLUE, fill_color=BLUE, fill_opacity=1).shift(RIGHT * 0.8 + UP * 0.5)

        # Create the beak
        beak = Polygon(
            LEFT * 0.4 + DOWN * 0.1, 
            RIGHT * 0.4 + DOWN * 0.1, 
            DOWN * 0.6,
            color=ORANGE, fill_color=ORANGE, fill_opacity=1
        )

        # Create the wings
        left_wing = Ellipse(width=2.4, height=1.2, color=BLACK, fill_color=BLACK, fill_opacity=1).shift(LEFT * 2 + DOWN * 0.4)
        right_wing = Ellipse(width=2.4, height=1.2, color=BLACK, fill_color=BLACK, fill_opacity=1).shift(RIGHT * 2 + DOWN * 0.4)

        # Create the legs
        left_leg = Line(DOWN * 0.8, DOWN * 1.2).shift(LEFT * 0.5 + DOWN * 0.9).set_color(ORANGE)
        right_leg = Line(DOWN * 0.8, DOWN * 1.2).shift(RIGHT * 0.5 + DOWN * 0.9).set_color(ORANGE)

        # Add all components to the scene
        self.add(body, left_wing, right_wing, left_eye_white, right_eye_white, left_pupil, right_pupil, beak, left_leg, right_leg)

        # Wing Animation (frequent flapping)
        flap_animation = AnimationGroup(
            left_wing.animate.shift(LEFT * 0.3).rotate(PI / 8),
            right_wing.animate.shift(RIGHT * 0.3).rotate(-PI / 8),
            run_time=0.2,
            rate_func=there_and_back
        )

        # Eye Blinking Animation
        blink_animation = AnimationGroup(
            left_pupil.animate.scale(0),
            right_pupil.animate.scale(0),
            run_time=0.2
        )
        restore_eyes = AnimationGroup(
            left_pupil.animate.scale(1),
            right_pupil.animate.scale(1),
            run_time=0.2
        )

        # Animate the bird flapping and blinking
        for _ in range(4):
            self.play(flap_animation)
            self.wait(0.1)
            #self.play(blink_animation, restore_eyes)
            #self.wait(0.1)

# Ensure the format and quality settings match the command-line options
config["quality"] = "low_quality"
config["format"] = "gif"
config["transparent"] = True
# Render the scene using the imported class
def render_bird_animation():
    scene = BirdAnimation()  # Instantiate the imported scene class
    scene.render()  # Render the scene

if __name__ == "__main__":
    render_bird_animation()  # Call the render function