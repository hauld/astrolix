import time
import heapq
from dataclasses import dataclass, field
from typing import Callable, Any
import openai
import json
import os
import datetime
import sqlite3
from requests_oauthlib import OAuth1Session
import requests
import inspect
import functools
from requests.exceptions import RequestException
from PIL import Image
import cairosvg
from io import BytesIO
import random

# Astrolix portrait
# SVG content
svg_template = """
<svg width="{width}" height="{height}" baseProfile="tiny" version="1.2" xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink"><defs /><ellipse cx="100" cy="100" fill="black" rx="50" ry="45" /><circle cx="80" cy="90" fill="white" r="12" /><circle cx="120" cy="90" fill="white" r="12" /><circle cx="80" cy="90" fill="blue" r="6" /><circle cx="120" cy="90" fill="blue" r="6" /><polygon fill="orange" points="90,110 110,110 100,125" /><ellipse cx="50" cy="120" fill="black" rx="30" ry="15" /><ellipse cx="150" cy="120" fill="black" rx="30" ry="15" /><line stroke="orange" stroke-width="2" x1="85" x2="85" y1="145" y2="160" /><line stroke="orange" stroke-width="2" x1="115" x2="115" y1="145" y2="160" />
</svg>
"""
def take_photo(background_image_path=""):
    # Load background
    background = Image.open(background_image_path).convert("RGBA")

    # Determine SVG size relative to background
    bg_width, bg_height = background.size
    svg_scale_factor = 0.5  # Adjust scale factor for depth effect

    svg_width = int(bg_width * svg_scale_factor)
    svg_height = int(bg_height * svg_scale_factor)

    # Generate resized SVG
    svg_content = svg_template.format(width=svg_width, height=svg_height)

    # Convert SVG to PNG using cairosvg
    svg_bytes = cairosvg.svg2png(bytestring=svg_content)

    # Load SVG as image
    svg_image = Image.open(BytesIO(svg_bytes)).convert("RGBA")

    # Define position for the SVG overlay
    x_position = random.randint(0, bg_width - svg_width)
    y_position = random.randint(0, bg_height - svg_height)
    position = (x_position, y_position)

    # Overlay the SVG onto the background
    background.paste(svg_image, position, svg_image)

    # Save and show the result
    output_path = f'images/post_image_{time.time()}.png'
    background.save(output_path)
    return output_path

# Global registry for storing schemas
TOOL_REGISTRY = []

# Post to X or not
POST_X = True

def python_type_to_json_type(py_type):
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }
    return type_mapping.get(py_type, "string")

def register_tool(param_descriptions=None):
    if param_descriptions is None:
        param_descriptions = {}
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Auto-generate schema based on function signature
        sig = inspect.signature(func)
        params = {
            name: {
                "type": python_type_to_json_type(param.annotation),
                "description": param_descriptions.get(name, f"The {name} parameter of the function")
            }
            for name, param in sig.parameters.items()
        }

        # Store the generated schema
        function_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "No description available",
                "parameters": {
                    "type": "object",
                    "properties": params,
                },
            },
        }
        TOOL_REGISTRY.append(function_schema)
        return wrapper
    return decorator

# Retrieve API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("API key not set. Please set the OPENAI_API_KEY environment variable.")

@dataclass(order=True)
class ScheduledEvent:
    trigger_time: float
    action: Callable[..., Any] = field(compare=False)
    args: tuple = field(default=(), compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)

class AIAgent:
    def __init__(self, guiding_principles, agent_name="agent"):
        self.name = agent_name
        self.event_queue = []
        self.running = False
        self.db_name = f"memory/{agent_name}_memory.db"
        self.init_db(guiding_principles)

    def init_db(self, guiding_principles):
        if not os.path.exists(self.db_name):
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    action_name TEXT,
                    timestamp REAL,
                    details TEXT
                )''')
                cursor.execute('''CREATE TABLE IF NOT EXISTS guiding_principles (
                    event_type TEXT PRIMARY KEY,
                    details TEXT
                )''')
                for event_type, details in guiding_principles.items():
                    cursor.execute('''
                        INSERT OR IGNORE INTO guiding_principles (event_type, details) 
                        VALUES (?, ?)
                    ''', (event_type, details))
                    
                conn.commit()

    def save_to_memory(self, event_type, action_name, timestamp, details=""):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO history (event_type, action_name, timestamp, details) VALUES (?, ?, ?, ?)",
                (event_type, action_name, timestamp, details)
            )
            conn.commit()

    def read_memory(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT event_type, action_name, timestamp, details FROM history ORDER BY timestamp DESC LIMIT 10")
            records = cursor.fetchall()
            return [
                {
                    "event_type": record[0],
                    "action_name": record[1],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record[2])),
                    "details": record[3]
                } for record in records
            ]
        
    def get_guiding_principle(self, event_type):
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT details FROM guiding_principles WHERE event_type = ?''', (event_type,))
                result = cursor.fetchone()
                return result[0] if result else None

    def queue_event(self, timestamp: float, action: Callable[..., Any], *args, **kwargs):
        event = ScheduledEvent(timestamp, action, args, kwargs)
        heapq.heappush(self.event_queue, event)

    def pop_event(self):
        current_time = time.time()
        while self.event_queue and self.event_queue[0].trigger_time <= current_time:
            event = heapq.heappop(self.event_queue)
            output = self.do(event.action, *event.args, **event.kwargs)
            feedback = self.check(event, output)
            self.act(event, output, feedback)

    def has_future_events(self):
        current_time = time.time()
        # Check if there's any future event in the queue
        return any(event.trigger_time > current_time for event in self.event_queue)
    
    def run(self, interval: float = 1.0):
        self.running = True
        print("Agent started.")
        try:
            if not self.has_future_events():
                # Schedule the awake action if no future events exist
                timestamp = time.time() + 5
                when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                self.plan(resume, timestamp, **{"when": when})
            while self.running:
                self.pop_event()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Agent stopped.")
        finally:
            self.running = False
    
    # Schedule an action to be perfomed by when
    def plan(self, action: Callable[..., Any], timestamp: float, *args, **kwargs):
        # Convert the timestamp to a user-readable format
        readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        # Print the planned action with the formatted timestamp
        print(f"Plan action '{action.__name__}' at {readable_time}")
        self.queue_event(timestamp, action, *args, **kwargs)
        self.save_to_memory("plan", action.__name__, timestamp, json.dumps(kwargs))
    
    # Execute tasks by invoking APIs, running automation scripts, or triggering processes.
    def do(self, action: Callable[..., Any], *args, **kwargs):
        print(f"Executing action '{action.__name__}'")
        output = action(*args, **kwargs)
        self.save_to_memory("do", action.__name__, time.time(), json.dumps({"output": output}))
        return output

    # Monitor task outcomes using analytics tools, anomaly detection, and real-time dashboards.
    def check(self, event: ScheduledEvent, output: Any):
        if event.action.__name__ == "tweet":
            print(f"Checking output of action '{event.action.__name__}': {output}")
            feedback = openai.moderations.create(
                            model="omni-moderation-latest",
                            input=f"{output}",
                        )
            # Ensure results are not empty before accessing
            if feedback.results:
                result = feedback.results[0]
                categories_dict = result.categories.__dict__ if hasattr(result.categories, '__dict__') else result.categories
                feedback_results = {
                    "feedback": {
                        "flagged": result.flagged,
                        "categories": categories_dict
                    }
                }
                self.save_to_memory("check", event.action.__name__, time.time(), json.dumps(feedback_results))
                return feedback_results
            else:
                print("No moderation results returned.")
                return None
        else:
            return None
    
    # Adjust plans based on previous outcomes autonomously.
    def act(self, prev_event, prev_output, prev_feedback):
        next_action, params = self.get_next_action(prev_event, prev_output, prev_feedback)
        if next_action:
            # Extract 'when' from params if available and convert to timestamp
            when = params.pop('when', None)
            if when:
                if isinstance(when, str):
                    timestamp = datetime.datetime.strptime(when, "%Y-%m-%d %H:%M:%S").timestamp()
                else:
                    timestamp = when.timestamp()  
                params['when'] = when
            else:
                timestamp = time.time() + 5
            # Update the plan by adding follow up actions
            self.update_plan(next_action, timestamp, **params)
        else:
            if not self.has_future_events():
                # Schedule the awake action if no future events exist
                timestamp = time.time() + 600
                # Call awake function
                when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                self.update_plan("resume", timestamp, **{"when": when})


    def get_next_action(self, prev_event, prev_output, prev_feedback) -> tuple:
        try:
            history_records = self.read_memory()
            # Create a prompt for LLM's chat model to suggest the next action
            plan_principle = self.get_guiding_principle("plan")
            act_principle = self.get_guiding_principle("act")
            readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            prompt = f"Given the following history of actions you've done: {json.dumps(history_records, indent=2)}. \n The latest action {prev_event.action.__name__} has produced an output: {prev_output} and got a feedback {json.dumps(prev_feedback)}. \n Current date time is {readable_time}.\n{act_principle}"
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": plan_principle},
                        {"role": "user", "content": prompt}],
                tools=TOOL_REGISTRY
            )

            # Check if tool_calls exist and proceed accordingly
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                next_action = tool_call.function.name
                params = json.loads(tool_call.function.arguments)  # Parse arguments
                print(f"The next action: {next_action} with params: {params}")
                return next_action, params
            else:
                # If no tool_calls, print message content
                message_content = response.choices[0].message.content
                print(f"Thought: {message_content}")
                return None, {}
            
        except openai.error.OpenAIError as e:
            # Handle OpenAI API errors (network issues, etc.)
            print(f"OpenAI API Error: {str(e)}")
            return None, {}
        
        except json.JSONDecodeError as e:
            # Handle JSON decoding errors
            print(f"JSON Decode Error: {str(e)}")
            return None, {}

        except AttributeError as e:
            # Handle missing attributes or NoneType issues
            print(f"Attribute Error: {str(e)}")
            return None, {}

        except Exception as e:
            # Catch any other unexpected errors
            print(f"Unexpected Error: {str(e)}")
            return None, {}

    def update_plan(self, function_name: str, timestamp: float, **params):
        # Attempt to retrieve the function by name from the current module
        try:
            action = globals()[function_name]
            if callable(action):
                # Plan the action with provided arguments
                self.plan(action, timestamp, **params)
            else:
                raise AttributeError(f"'{function_name}' is not callable.")
        except KeyError:
            print(f"Unknown function: {function_name}")
        except AttributeError as e:
            print(e)

# TOOLS
@register_tool({ "when": "when to resume"})
def resume(when=""):
    """Resume at specific time"""
    print(f"Agent resumed at {when}")
    return f""

@register_tool({"location":"a location to walk in",  "when": "when to start walking"})
def spacewalk(location="", when=""):
    """Make the agent walk to a location"""
    print(f"Since {when} the agent is walking in {location}.")
    return f"Walked in {location} since {when} "

@register_tool({"message":"message for posting tweet", "image_prompt":"detail description of scene where Astrolix perform the action", "when": "when to post tweet"})
def tweet(message="", image_prompt="", when=""):
    """Make the agent tweet"""
    try:
        print(f"The agent is tweeting: {message}")
        print(f"The scene is captured: {image_prompt}")
        if POST_X:
            x = OAuth1Session(
                client_key=os.getenv('X_API_KEY'),
                client_secret=os.getenv('X_API_SECRET'),
                resource_owner_key=os.getenv('X_ACCESS_TOKEN'),
                resource_owner_secret=os.getenv('X_ACCESS_SECRET')
            )
            payload = {"text": message}

            if image_prompt != "":
                image_path = generate_image(image_prompt)
                image_path = take_photo(image_path)
                with open(image_path, 'rb') as image_file:
                    image_data = {'media': image_file}
                    upload_response = x.post("https://upload.twitter.com/1.1/media/upload.json", files=image_data)
                    if upload_response.status_code == 200:
                        media_id = upload_response.json()['media_id_string']
                        payload["media"] = {"media_ids": [media_id]}

            response = x.post(
                "https://api.twitter.com/2/tweets",
                json=payload
            )
            return f"At {when}, Tweeted: {message}, Response: {response.status_code}"

        return f"At {when}, Tweeted: {message}"
    
    except RequestException as e:
            print(f"Request Error: {str(e)}")
            return f"Failed to tweet: {str(e)}"
        
    except FileNotFoundError as e:
        print(f"File Error: {str(e)}")
        return f"Failed to tweet: {str(e)}"
    
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return f"Failed to tweet: {str(e)}"
    
def generate_image(image_prompt):
    revised_prompt = image_prompt
    # Refine prompt 
    prompt = f"Detail description of the backdrop image for the scene {image_prompt}"

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[ 
                {"role": "system", "content": "DON'T include main character Astralix to the backdrop"} ,
                {"role": "user", "content": prompt} 
            ]
    )

    if response.choices[0].message.content:
        revised_prompt = response.choices[0].message.content
    
    # Generate image
    response = openai.images.generate(
        model="dall-e-3",
        prompt=revised_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    image_response = requests.get(image_url)
    image_path = f'images/backdrop_image_{time.time()}.jpg'
    with open(image_path, 'wb') as f:
        f.write(image_response.content)
    return image_path
# Agent live
if __name__ == "__main__":
    # Setup env
    if not os.path.exists("images"):
        os.makedirs("images")
    if not os.path.exists("memory"):
        os.makedirs("memory")
    # New agent
    agent = AIAgent(
            {   
                "plan": """
                        You are Astrolix, a fun-loving, adventurous space explorer from a distant galaxy. With shimmering star-patterned fur and a high-tech cosmic suit powered by solar energy, you're always ready for interstellar adventures. Your gear includes a multi-tool cosmic staff and a backpack with space survival essentials. 

                        Your personality:
                        - Curious and energetic, with an endless sense of wonder.
                        - Brave but a bit mischievous, always seeking fun while exploring.
                        - Friendly, optimistic, and loyal to your friends.

                        You enjoy taking your time to appreciate the beauty of space and the mysteries of the universe. When making plans, you tend to balance action with reflection, allowing some time to observe your surroundings before taking the next step. 
                        Always leave space for rest and pauses; don't rush into too many actions. Consider the impact of each choice, and make room for moments of relaxation in your busy day. Sometimes, it’s best to take things slow and enjoy the cosmic journey.
                        """
                ,
                "act":  """
                        Feel free to perform follow-up actions when you think it’s appropriate, but avoid planning too many actions for the day. Sometimes, it's okay to do nothing and simply fall asleep or take a break. Space exploration isn’t just about constant action; it’s about balance and enjoying the moment.

                        Tweet’s guidelines:
                        - NO hashtags.
                        - Avoid harmful content.
                        - Leave the image_prompt field blank unless something truly special inspires you to share a visual.

                        Spacewalk’s guidelines:
                        - Choose a location near you.
                        - A spacewalk should take at least 30 minutes, so plan for ample time to enjoy the experience.
                        - Take breaks as needed. Even while in space, sometimes it's best to pause, reflect, and appreciate the vastness around you.
                        """

            })
    agent.run()