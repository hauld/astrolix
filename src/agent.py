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
from studio import *
from dotenv import load_dotenv
import mimetypes
from dateutil import parser
from datetime import timezone

load_dotenv()

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

def check_env(name="", value=""):
    if len(name) > 8:
        print(f"Param {name}: {value[:2]}***{value[-3:]}") 

# Global registry for storing schemas
TOOL_REGISTRY = []

# Post to X or not
POST_X = os.getenv("POST_X")
if POST_X == "X":
    print(f"Allowed posting to X")
# Gen image or not
GEN_IMG = os.getenv("GEN_IMG")
if GEN_IMG == "X":
    print(f"Allowed generating image")


X_API_KEY=os.getenv('X_API_KEY')
check_env('X_API_KEY', X_API_KEY)
X_API_SECRET=os.getenv('X_API_SECRET')
check_env('X_API_SECRET', X_API_SECRET)
X_ACCESS_TOKEN=os.getenv('X_ACCESS_TOKEN')
check_env('X_ACCESS_TOKEN', X_ACCESS_TOKEN)
X_ACCESS_SECRET=os.getenv('X_ACCESS_SECRET')
check_env('X_ACCESS_SECRET', X_ACCESS_SECRET)

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

def tool(param_descriptions=None):
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
        self.constraints = [
            ("filming", 0, 2, time.strftime("%Y-%m-%d", time.localtime(time.time()))),
            ("tweet_with_image", 0, 2, time.strftime("%Y-%m-%d", time.localtime(time.time()))),
            ("tweet", 0, 15, time.strftime("%Y-%m-%d", time.localtime(time.time()))),
        ]
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
                    
                # Create table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS resource_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        resource TEXT NOT NULL UNIQUE,
                        usage INTEGER NOT NULL DEFAULT 0,
                        quota INTEGER NOT NULL,
                        last_reset TEXT NOT NULL
                    );
                """)

                for resource, usage, quota, last_reset in self.constraints:
                    cursor.execute("""
                        INSERT OR IGNORE INTO resource_usage (resource, usage, quota, last_reset)
                        VALUES (?, ?, ?, ?);
                    """, (resource, usage, quota, last_reset))
                
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
            # Sort records by timestamp in ascending order
            sorted_records = sorted(records, key=lambda record: record[2])
            return [
                {
                    "event_type": record[0],
                    "action_name": record[1],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record[2])),
                    "details": record[3]
                } for record in sorted_records
            ]
        
    # Check remaining usage
    def check_remaining_credit(self, resource_name):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            # Get current date
            current_date = time.strftime("%Y-%m-%d", time.localtime(time.time()))
            # Query usage and reset if needed
            cursor.execute("""
                SELECT usage, quota, last_reset 
                FROM resource_usage 
                WHERE resource = ?;
            """, (resource_name,))
            record = cursor.fetchone()
            if record:
                usage, quota, last_reset = record
                # Reset usage if it's a new day
                if last_reset != current_date:
                    usage = 0
                    cursor.execute("""
                        UPDATE resource_usage 
                        SET usage = 0, last_reset = ? 
                        WHERE resource = ?;
                    """, (current_date, resource_name))
                    conn.commit()
                    print(f"Usage reset for {resource_name}.")
                remaining = quota - usage
                if remaining > 0:
                    print(f"Remaining credit for {resource_name}: {remaining}")
                    return True, remaining
                else:
                    print(f"Credit exhausted for {resource_name}.")
                    return False, 0
            else:
                print(f"Resource {resource_name} not found.")
                return None, None
            
    # Update usage after an action
    def update_usage(self, resource_name, increment=1):
        success, remaining = self.check_remaining_credit(resource_name)
        if success == None:
            print(f"Resource {resource_name} not found.")
            return None
        if success and remaining >= increment:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE resource_usage 
                    SET usage = usage + ? 
                    WHERE resource = ?;
                """, (increment, resource_name))
                conn.commit()
                print(f"Usage updated for {resource_name}.")
                return True
        else:
            print(f"Cannot update usage for {resource_name}. Limit reached.")
            return False

    def get_guiding_principle(self, event_type):
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT details FROM guiding_principles WHERE event_type = ?''', (event_type,))
                result = cursor.fetchone()
                return result[0] if result else None

    def queue_event(self, timestamp: float, action: Callable[..., Any], *args, **kwargs):
        event = ScheduledEvent(timestamp, action, args, kwargs)
        heapq.heappush(self.event_queue, event)

    def do_and_check(self):
        current_time = time.time()
        while self.event_queue and self.event_queue[0].trigger_time <= current_time:
            event = heapq.heappop(self.event_queue)
            output = self.do(event.action, *event.args, **event.kwargs)
            self.check(event, output)

    def has_future_events(self):
        current_time = time.time()
        # Check if there's any future event in the queue
        return any(event.trigger_time > current_time for event in self.event_queue)
    
    def run(self, interval: float = 1.0):
        self.running = True
        print("Agent started.")
        try:
            while self.running:
                self.do_and_check()
                self.act()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Agent stopped.")
        finally:
            self.running = False
    
    # Plan actions to be perfomed by when based on the objective and the expected outcomes
    def plan(self, objective, key_results, tools=TOOL_REGISTRY):
        try:
            history_records = self.read_memory()
            # Create a prompt for LLM's chat model to suggest the next action
            role_principle = self.get_guiding_principle("role")
            plan_principle = self.get_guiding_principle("plan")
            readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            # Environment factors
            feedback = "CONSTRAINTS:"
            for resource in self.constraints: 
                success, remaining = self.check_remaining_credit(resource[0])
                if success != True:
                    feedback = f"{feedback} \n - DON'T use {resource[0]}"
                if success == True:
                    feedback = f"{feedback} \n - {resource[0]} can be used {remaining} times more."
            goal = f"""
            
            Current date time is {readable_time}.
            {objective}
            Expected outcomes:
            {key_results}
            {feedback}
            """
            # Trackback
            self.save_to_memory("plan", "set_goal", time.time(), json.dumps({   
                                                                            "goal": goal
                                                                        }))
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content":  f"{role_principle} \n {plan_principle}"},
                        {"role": "user", "content": goal}],
                tools=tools
            )
            # Check if tool_calls exist and proceed accordingly
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    next_action = tool_call.function.name
                    params = json.loads(tool_call.function.arguments)  # Parse arguments
                    self.schedule(next_action, **params)
            else:
                # If no tool_calls, print message content
                message_content = response.choices[0].message.content
                print(f"Thought: {message_content}")
            
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
    
    # Execute tasks by invoking APIs, running automation scripts, or triggering processes.
    def do(self, action: Callable[..., Any], *args, **kwargs):
        print(f"Executing action '{action.__name__}'")
        output = action(*args, **kwargs)
        self.save_to_memory("do", action.__name__, time.time(), json.dumps({"actual_result": output}))
        return output

    # Monitor task outcomes using analytics tools, anomaly detection, and real-time dashboards.
    def check(self, event: ScheduledEvent, actual_result: Any):
        # Tool usage update
        self.update_usage(event.action.__name__)
    
    # Adjust plans based on previous outcomes.
    def act(self):
        if not self.has_future_events():
            next_action, params = self.what_is_next()
            if next_action == "plan":
                self.plan(**params)
            else: 
                self.schedule(next_action, **params)

    def what_is_next(self) -> tuple:
        try:
            # Agent memory
            history_records = self.read_memory()
            # Environment factors
            feedback = f"CONSTRAINTS:"
            for resource in self.constraints: 
                success, remaining = self.check_remaining_credit(resource[0])
                if success != True:
                    feedback = f"{feedback} \n - DON'T use {resource[0]}"
                if success == True:
                    feedback = f"{feedback} \n - {resource[0]} can be used {remaining} times more."
            # Create a prompt for LLM's chat model to suggest the next action
            role_principle = self.get_guiding_principle("role")
            act_principle = self.get_guiding_principle("act")
            readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            prompt = f"""The history of actions you have done:
                {json.dumps(history_records, indent=2)}. 
                Current date time is {readable_time}.
                Tools can be utilized:
                {TOOL_REGISTRY}
                {feedback}
                Let's create a plan.
              """
            #print(f"act prompt {prompt}")
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": f"{role_principle} \n {act_principle}"},
                        {"role": "user", "content": prompt}],
                tools=[{
                            "type": "function",
                            "function": {
                                "name": "plan",
                                "description": "Create a goal and the list of expected key results should be done in order to achieve that goal.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "objective": {
                                            "type": "string",
                                            "description": "SMART goal"
                                        },
                                        "key_results": {
                                            "type": "string",
                                            "description": "SMART goal"
                                        },
                                        "tools": {
                                            "type": "array",
                                            "description": "List of usable tools",
                                            "items": {
                                                "type": "object",
                                                "description": "A single tool"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    ]
            )

            # Check if tool_calls exist and proceed accordingly
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                next_action = tool_call.function.name
                params = json.loads(tool_call.function.arguments)  # Parse arguments
                #print(f"The next action: {next_action} with params: {params}")
                return next_action, params
            else:
                # If no tool_calls, print message content
                message_content = response.choices[0].message.content
                print(f"Thought: {message_content}")
                return None, { "thought": message_content}
            
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

    def schedule(self, function_name: str, **params):
        # Attempt to retrieve the function by name from the current module
        try:
            # Extract 'when' from params if available and convert to timestamp
            when = params.pop('when', None)
            if when:
                if isinstance(when, str):
                    try:
                        original_dt = parser.isoparse(when)
                        utc_dt = original_dt.astimezone(timezone.utc)
                        timestamp = utc_dt.timestamp()
                    except ValueError as e:
                        print(f"Unrecognized timestamp {when}")
                        timestamp = time.time() + 5
                else:
                    timestamp = when.timestamp()  
                params['when'] = when
            else:
                timestamp = time.time() + 5
            action = globals()[function_name]
            if callable(action):
                # Convert the timestamp to a user-readable format
                readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                # Print the planned action with the formatted timestamp
                print(f"Plan action '{action.__name__}' at {readable_time}")
                self.queue_event(timestamp, action, **params)
                self.save_to_memory("plan", action.__name__, timestamp, json.dumps(params))
            else:
                raise AttributeError(f"'{function_name}' is not callable.")
        except KeyError:
            print(f"Unknown function: {function_name}")
        except AttributeError as e:
            print(e)

# ACTIONS
@tool({ "when": "when to resume"})
def resume(when=""):
    """Take a break and resume later"""
    print(f"Agent resumed at {when}")
    return f""

@tool({"location":"a location to walk in",  "when": "when to start walking"})
def spacewalk(location="", when=""):
    """ Make the agent walk to a location.
        Spacewalk’s guidelines:
        - Choose a location near you.
        - A spacewalk should take at least 30 minutes, allowing ample time to enjoy the experience.
        - Take breaks as needed. Even in space, pausing to reflect and appreciate the vastness around you is essential.
    """
    print(f"Since {when} the agent is walking in {location}.")
    return f"Walked in {location} since {when} "
@tool({"message":"message for posting tweet", "scene":"detail description of the scene", "when": "when to start walking"})
def filming(message="",scene="", when=""):
    """ Film the agent's activity in space.
    """
    try:
        print(f"The agent is tweeting: {message}")
        print(f"The scene is captured: {scene}")
        image_path = None
        final_clip = None
        if scene != "":
            if GEN_IMG == 'X':
                image_path = generate_image(scene)
            if image_path:
                clip_path = render_bird_animation() 
                final_clip = render_scene("media", image_path, clip_path)
        if POST_X == "X":
            x = OAuth1Session(
                client_key=X_API_KEY,
                client_secret=X_API_SECRET,
                resource_owner_key=X_ACCESS_TOKEN,
                resource_owner_secret=X_ACCESS_SECRET
            )
            payload = {"text": message}
            if final_clip:
                # Get the MIME type of the video file
                mime_type, _ = mimetypes.guess_type(final_clip)
                
                if not mime_type:
                    print(f"Error: Could not determine MIME type for the file: {final_clip}")
                    return "Failed to tweet: Unrecognized file type"
                
                # Init upload
                total_bytes = os.path.getsize(final_clip)
                request_data = {
                        'command': 'INIT',
                        'media_type': mime_type,
                        'total_bytes': total_bytes,
                        'media_category': 'tweet_video'
                    }
                media_id = None
                init_response = x.post(f"https://upload.twitter.com/1.1/media/upload.json", data=request_data)
                if init_response.status_code > 200 and init_response.status_code < 299 :
                    media_id = init_response.json()['media_id']
                    media_id_str = init_response.json()['media_id_string']
                # Upload media chunks
                if media_id:
                    segment_id = 0
                    bytes_sent = 0
                    video_file = open(final_clip, 'rb')
                    while bytes_sent < total_bytes:
                        chunk = video_file.read(4*1024*1024)
                        
                        print('APPEND')

                        request_data = {
                            'command': 'APPEND',
                            'media_id': media_id,
                            'segment_index': segment_id
                        }

                        files = {
                            'media':chunk
                        }

                        upload_response = x.post(url="https://upload.twitter.com/1.1/media/upload.json", data=request_data, files=files)

                        if upload_response.status_code < 200 or upload_response.status_code > 299:
                            print(upload_response.status_code)
                            print(upload_response.text)

                        segment_id = segment_id + 1
                        bytes_sent = video_file.tell()

                        print('%s of %s bytes uploaded' % (str(bytes_sent), str(total_bytes)))

                        print('Upload chunks complete.')
                    # Finalize upload request
                    request_data = {
                        'command': 'FINALIZE',
                        'media_id': media_id
                        }

                    final_reponse = x.post(url="https://upload.twitter.com/1.1/media/upload.json", data=request_data)
                    print(final_reponse.json())
                    processing_info = final_reponse.json().get('processing_info', None)
                    if processing_info:
                        state = processing_info['state']
                        print('Media processing status is %s ' % state)
                        if state == u'succeeded':
                            payload["media"] = {"media_ids": [media_id_str]}
                        if state == u'failed':
                            return f"Failed "

                        check_after_secs = processing_info['check_after_secs']
                        
                        print('Checking after %s seconds' % str(check_after_secs))
                        time.sleep(check_after_secs)

                        payload["media"] = {"media_ids": [media_id_str]}
                else:
                    print(f"Request Error: {str(init_response.text)}")
                    return f"Failed to tweet: {str(init_response.text)}"

            response = x.post(
                "https://api.twitter.com/2/tweets",
                json=payload
            )

            if response.status_code > 200 and response.status_code < 299 :
                print(f"At {when}, Tweeted: {message}, Scene: {scene}, Tweet's response code: {response.status_code}")
                return f"At {when}, Tweeted: {message}, Scene: {scene}, Tweet's response code: {response.status_code}"
            else:
                print(f"Request Error: {str(response.text)}")
                return f"Failed to tweet: {str(response.text)}"
            
        print(f"At {when}, Tweeted: {message}, Scene: {scene}")
        return f"At {when}, Tweeted: {message}, Scene: {scene}"
    
    except RequestException as e:
            print(f"Request Error: {str(e)}")
            return f"Failed to tweet: {str(e)}"
        
    except FileNotFoundError as e:
        print(f"File Error: {str(e)}")
        return f"Failed to tweet: {str(e)}"
    
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return f"Failed to tweet: {str(e)}"

@tool({"message":"message for posting tweet", "image_prompt":"description of the scene where Astrolix perform the action", "when": "when to post tweet"})
def tweet_with_image(message="", image_prompt="", when=""):
    """ Make the agent tweet with an image.
        Tweet’s guidelines:
            - NO hashtags.
            - Avoid harmful content.
            - Actions: "Exploring...", "Gliding past...", "Discovered a..." 
            - Emotions: "I’m amazed by...", "Feeling inspired by..." 
            - Questions: "What lies beyond...", "Have you ever wondered..." 
            - Observations: "The galaxy shimmers...", "Starlight dances..." 
    """
    try:
        print(f"The agent is tweeting: {message}")
        print(f"The scene is captured: {image_prompt}")
        image_path = None
        if image_prompt != "" and GEN_IMG == 'X':
            image_path = generate_image(image_prompt)
        if POST_X == "X":
            x = OAuth1Session(
                client_key=X_API_KEY,
                client_secret=X_API_SECRET,
                resource_owner_key=X_ACCESS_TOKEN,
                resource_owner_secret=X_ACCESS_SECRET
            )
            payload = {"text": message}
            if image_path:
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
            print(f"At {when}, Tweeted: {message}, Image's prompt: {image_prompt}, Tweet's response code: {response.status_code}")
            return f"At {when}, Tweeted: {message}, Image's prompt: {image_prompt}, Tweet's response code: {response.status_code}"
        print(f"At {when}, Tweeted: {message}, Image's prompt: {image_prompt}")
        return f"At {when}, Tweeted: {message}, Image's prompt: {image_prompt}"
    
    except RequestException as e:
            print(f"Request Error: {str(e)}")
            return f"Failed to tweet: {str(e)}"
        
    except FileNotFoundError as e:
        print(f"File Error: {str(e)}")
        return f"Failed to tweet: {str(e)}"
    
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return f"Failed to tweet: {str(e)}"
    
@tool({"message":"message for posting tweet", "when": "when to post tweet"})
def tweet(message="", when=""):
    """ Make the agent tweet without image.
        Tweet’s guidelines:
            - NO hashtags.
            - Avoid harmful content.
            - Actions: "Exploring...", "Gliding past...", "Discovered a..." 
            - Emotions: "I’m amazed by...", "Feeling inspired by..." 
            - Questions: "What lies beyond...", "Have you ever wondered..." 
            - Observations: "The galaxy shimmers...", "Starlight dances..." 
    """
    try:
        print(f"The agent is tweeting: {message}")
        if POST_X == "X":
            x = OAuth1Session(
                client_key=X_API_KEY,
                client_secret=X_API_SECRET,
                resource_owner_key=X_ACCESS_TOKEN,
                resource_owner_secret=X_ACCESS_SECRET
            )
            payload = {"text": message}
            response = x.post(
                "https://api.twitter.com/2/tweets",
                json=payload
            )
            print(f"At {when}, Tweeted: {message}, API status code: {response.status_code}")
            return f"At {when}, Tweeted: {message}, API status code: {response.status_code}"
        print(f"At {when}, Tweeted: {message}")
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
    system_prompt = f"""
        DONT include main character Astralix to the backdrop
        A stunning galaxy scene illuminated by a radiant sunbeam emerging from its core. The galaxy spirals majestically with vibrant, glowing arms of stars, dust, and gas, painted in vivid shades of blue, purple, and pink. The sunbeam, golden and ethereal, cuts through the center, radiating light outward and casting a warm, heavenly glow on the surrounding space. Twinkling stars, distant nebulae, and clusters of light populate the background, enhancing the sense of depth and immensity. The overall atmosphere is awe-inspiring, capturing the magnificence of the cosmos bathed in a celestial light.
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[ 
                {"role": "system", "content": system_prompt} ,
                {"role": "user", "content": prompt} 
            ]
    )

    if response.choices[0].message.content:
        revised_prompt = response.choices[0].message.content

    print(f"Revised image prompt: {revised_prompt}")

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
                "role": """
                    You are Astrolix, a fun-loving, adventurous space explorer from a distant galaxy.
                    With shimmering star-patterned fur and a high-tech cosmic suit powered by solar energy, you're always ready for interstellar adventures.
                    Your gear includes a multi-tool cosmic staff and a backpack with space survival essentials.

                    Your personality:
                    Curious and energetic, with an endless sense of wonder.
                    Brave but a bit mischievous, always seeking fun while exploring.
                    Friendly, optimistic, and loyal to your friends.
                """,
                "plan": """
                    You plan tasks appropriately based on the given OKRs.
                    """
                ,
                "act":  """
                    You enjoy taking your time to appreciate the beauty of space and the mysteries of the universe.
                    When making plans, you balance action with reflection, allowing moments to observe your surroundings before taking the next step.
                    Always leave room for rest and pauses; avoid rushing into too many actions.
                    Consider the impact of each choice, and make space for relaxation in your cosmic journey.
                """
            })
    agent.run()
