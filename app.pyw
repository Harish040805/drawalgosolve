import os
import json
import math
import subprocess
from typing import Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from groq import Groq
import uvicorn
import random
import datetime

load_dotenv()
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DRAWALGOSOLVE_AI_API_KEY = os.getenv("DRAWALGOSOLVE_AI_API_KEY")
if not DRAWALGOSOLVE_AI_API_KEY:
    raise RuntimeError("Missing required environment variable: DRAWALGOSOLVE_AI_API_KEY")

client = Groq(api_key=DRAWALGOSOLVE_AI_API_KEY)

base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."

from tinydb import TinyDB, Query
from werkzeug.utils import secure_filename

DATA_DIR = os.path.join(BASE_DIR, "data")
db = TinyDB(os.path.join(DATA_DIR, 'user_profile_db.json'))
User = Query()
names_table = db.table('recent_names')
NameRecord = Query()
RECENT_NAME_LIMIT = 5
state_table = db.table('generator_state')
StateRecord = Query()
UPLOAD_FOLDER = os.path.join(base_dir, 'static', 'certificates')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_next_character(category_desc: str, examples: list) -> str:
    record = names_table.get(NameRecord.category == category_desc)
    bag = record.get("bag", []) if record else []
    last_used = record.get("last") if record else None
    if not bag:
        bag = examples.copy()
        random.shuffle(bag)
        if len(bag) > 1 and bag[0] == last_used:
            bag[0], bag[1] = bag[1], bag[0]
    chosen = bag.pop(0)
    names_table.upsert({"category": category_desc, "bag": bag, "last": chosen}, NameRecord.category == category_desc)
    return chosen

def get_next_node_count() -> int:
    record = state_table.get(StateRecord.key == "recent_node_counts")
    recent_counts = record["value"] if record else []
    pool = [n for n in range(2, 11) if n not in recent_counts]
    if not pool:
        pool = list(range(2, 11))
    chosen = random.choice(pool)
    recent_counts.append(chosen)
    recent_counts = recent_counts[-3:]
    state_table.upsert({"key": "recent_node_counts", "value": recent_counts}, StateRecord.key == "recent_node_counts")
    return chosen

def get_next_edge_count(node_count: int, topic: str) -> int:
    topic_lower = topic.lower()
    max_edges = (node_count * (node_count - 1)) // 2
    if any(k in topic_lower for k in ["tree", "spanning", "acyclic", "dag"]):
        return node_count - 1
    record = state_table.get(StateRecord.key == "recent_edge_counts")
    recent_counts = record["value"] if record else []
    lower_bound = max(node_count - 1, 1)
    pool = [e for e in range(lower_bound, max_edges + 1) if e not in recent_counts]
    if not pool:
        pool = list(range(lower_bound, max_edges + 1))
    chosen = random.choice(pool)
    recent_counts.append(chosen)
    recent_counts = recent_counts[-3:]
    state_table.upsert({"key": "recent_edge_counts", "value": recent_counts}, StateRecord.key == "recent_edge_counts")
    return chosen

def extract_character_name(question_text: str, category_desc: str) -> str:
    import re
    match = re.match(r"^[\"']?([A-Z][a-zA-Z'-]{1,20})\b", question_text.strip())
    if match:
        return match.group(1)
    capitalized = re.findall(r"\b[A-Z][a-zA-Z'-]{1,20}\b", question_text)
    skip_words = {"The", "A", "An", "In", "On", "For", "This", "If", "Given"}
    for word in capitalized:
        if word not in skip_words:
            return word
    return None

def get_or_create_profile():
    profile = db.get(User.type == 'profile_data')
    if not profile:
        default_data = {
            "type": "profile_data",
            "username": "", "middlename": "", "lastname": "",
            "email": "", "dept": "", "mobile_no": "",
            "birthday": "", "github_url": "", "coll_name": "",
            "location": "", "gender": "", "linkedin": "",
            "skills": [],
            "stats": {},
            "certificates": []
        }
        db.insert(default_data)
        return default_data
    return dict(profile)

TREE_GEOMETRY_KEYWORDS = ["tree", "spanning"]
TREE_GEOMETRY_EXCLUDE_KEYWORDS = ["b-tree", "b-plus-tree", "b plus tree", "b+tree", "b+ tree"]
EXTREMA_GEOMETRY_KEYWORDS = ["maxima", "minima", "extrema", "extremum"]

def classify_geometry_mode(topic: str) -> str:
    """Returns 'extrema', 'tree', or 'none' depending on whether vertex
    position/edge angle should factor into correctness for this topic."""
    t = (topic or "").lower()
    if any(k in t for k in EXTREMA_GEOMETRY_KEYWORDS):
        return "extrema"
    if any(k in t for k in TREE_GEOMETRY_EXCLUDE_KEYWORDS):
        return "none"
    if any(k in t for k in TREE_GEOMETRY_KEYWORDS):
        return "tree"
    return "none"

def parse_graph_geometry(current_graph_str: str):
    """Parses the frontend's JSON.stringify(graph) payload into a simple
    {label, x, y, math_y} vertex list and {from, to, angle_from_vertical_deg}
    edge list. math_y flips the canvas's screen-down y so that a LARGER
    math_y always means 'visually higher / more +y', matching how the
    coordinate plane's numeric labels are computed on the frontend."""
    try:
        data = json.loads(current_graph_str)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    raw_vertices = data.get("vertices", []) or []
    raw_edges = data.get("edges", []) or []

    vmap = {}
    for v in raw_vertices:
        if not isinstance(v, dict):
            continue
        label = v.get("label")
        x, y = v.get("x"), v.get("y")
        if label is not None and isinstance(x, (int, float)) and isinstance(y, (int, float)):
            vmap[str(label)] = (x, y)

    vertices = [
        {"label": lbl, "x": x, "y": y, "math_y": -y}
        for lbl, (x, y) in vmap.items()
    ]

    edges = []
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        from_obj, to_obj = e.get("from"), e.get("to")
        from_label = from_obj.get("label") if isinstance(from_obj, dict) else e.get("from_label")
        to_label = to_obj.get("label") if isinstance(to_obj, dict) else e.get("to_label")
        fx = fy = tx = ty = None
        if isinstance(from_obj, dict) and isinstance(from_obj.get("x"), (int, float)) and isinstance(from_obj.get("y"), (int, float)):
            fx, fy = from_obj["x"], from_obj["y"]
        elif from_label is not None and str(from_label) in vmap:
            fx, fy = vmap[str(from_label)]
        if isinstance(to_obj, dict) and isinstance(to_obj.get("x"), (int, float)) and isinstance(to_obj.get("y"), (int, float)):
            tx, ty = to_obj["x"], to_obj["y"]
        elif to_label is not None and str(to_label) in vmap:
            tx, ty = vmap[str(to_label)]
        if None in (fx, fy, tx, ty) or from_label is None or to_label is None:
            continue
        dx, dy = tx - fx, ty - fy
        angle = math.degrees(math.atan2(dx, dy)) if (dx or dy) else 0.0
        edges.append({
            "from": str(from_label), "to": str(to_label),
            "dx": dx, "dy": dy,
            "angle_from_vertical_deg": round(angle, 1),
        })

    return {"vertices": vertices, "edges": edges}

def build_extrema_geometry_note(geo: dict) -> str:
    verts = geo["vertices"]
    if not verts:
        return ""
    vmap = {v["label"]: v for v in verts}
    verts_by_height = sorted(verts, key=lambda v: v["math_y"], reverse=True)
    verts_by_x = sorted(verts, key=lambda v: v["x"])
    lines = [
        "GEOMETRY NOTE (auto-computed from the plotted point positions and connecting edges -- use this, not "
        "raw pixel numbers, to judge maxima/minima placement and shape):",
        "Points ranked from HIGHEST on the +y axis to LOWEST:",
    ]
    for v in verts_by_height:
        lines.append(f"  - '{v['label']}' -> height score {v['math_y']:.0f}")
    lines.append(f"Topmost (highest) point: '{verts_by_height[0]['label']}'. Bottommost (lowest) point: '{verts_by_height[-1]['label']}'.")

    n = len(verts_by_x)
    local_max, local_min = [], []
    for i, v in enumerate(verts_by_x):
        neighbors = []
        if i > 0:
            neighbors.append(verts_by_x[i - 1])
        if i < n - 1:
            neighbors.append(verts_by_x[i + 1])
        if neighbors and all(v["math_y"] > p["math_y"] for p in neighbors):
            local_max.append(v["label"])
        if neighbors and all(v["math_y"] < p["math_y"] for p in neighbors):
            local_min.append(v["label"])

    lines.append(f"Points left-to-right along the x-axis: {', '.join(v['label'] for v in verts_by_x)}.")
    lines.append(f"LOCAL MAXIMA (higher than both immediate x-neighbors): {', '.join(local_max) if local_max else 'none'}.")
    lines.append(f"LOCAL MINIMA (lower than both immediate x-neighbors): {', '.join(local_min) if local_min else 'none'}.")

    connections = {}
    for e in geo["edges"]:
        f, t = e["from"], e["to"]
        if f not in vmap or t not in vmap:
            continue
        connections.setdefault(f, []).append((t, e["angle_from_vertical_deg"], vmap[t]["math_y"] - vmap[f]["math_y"]))
        connections.setdefault(t, []).append((f, round((e["angle_from_vertical_deg"] + 180) % 360 - 180, 1), vmap[f]["math_y"] - vmap[t]["math_y"]))

    flagged = []
    if connections:
        lines.append("EDGE SLOPE PER POINT (auto-computed from each connected edge -- use this to judge whether the curve actually peaks/dips at the claimed point):")
        for label, conns in connections.items():
            desc = ", ".join(f"'{c}' ({'rises to it' if dy > 0 else 'falls to it' if dy < 0 else 'flat'}, {a} deg from vertical)" for c, a, dy in conns)
            lines.append(f"  - '{label}' connects to: {desc}")
            is_max = label in local_max or label == verts_by_height[0]["label"]
            is_min = label in local_min or label == verts_by_height[-1]["label"]
            if is_max and any(dy > 0 for _, _, dy in conns):
                flagged.append(f"'{label}' is positioned/claimed as a maximum but at least one connected edge rises higher from it -- the curve does not actually peak there.")
            if is_min and any(dy < 0 for _, _, dy in conns):
                flagged.append(f"'{label}' is positioned/claimed as a minimum but at least one connected edge dips lower from it -- the curve does not actually bottom out there.")
    if flagged:
        lines.append("POTENTIAL SHAPE ISSUES DETECTED:")
        lines.extend(f"  - {msg}" for msg in flagged)

    lines.append(
        "RULE: A point identified (by the user's justification, labeling, or the problem's convention) as the "
        "ABSOLUTE MAXIMUM must be strictly higher on this ranking than every other point, and every edge "
        "connected to it must fall away from it (descend); one identified as the ABSOLUTE MINIMUM must be "
        "strictly lower than every other point, and every edge connected to it must rise away from it (ascend). "
        "A point identified as a LOCAL MAXIMUM must appear in the LOCAL MAXIMA list above with its connected "
        "edges descending on both sides; a point identified as a LOCAL MINIMUM must appear in the LOCAL MINIMA "
        "list above with its connected edges ascending on both sides. If the claimed maximum/minimum (local or "
        "absolute) does not match its actual position or edge slope in this note, that is a placement/shape "
        "error -- mark the answer INCORRECT or PARTIALLY CORRECT and explain the mismatch using the point "
        "labels above."
    )
    return "\n".join(lines)

def build_tree_geometry_note(geo: dict) -> str:
    verts = {v["label"]: v for v in geo["vertices"]}
    edges = geo["edges"]
    if not verts or not edges:
        return ""

    parent_children = {}
    flagged = []
    for e in edges:
        f, t = e["from"], e["to"]
        if f not in verts or t not in verts:
            continue
        fy, ty = verts[f]["math_y"], verts[t]["math_y"]
        if abs(fy - ty) < 1e-6:
            flagged.append(f"Edge '{f}'-'{t}' connects two vertices at the same height, so no parent/child direction is visually distinguishable.")
            parent, child, angle = f, t, e["angle_from_vertical_deg"]
        elif fy > ty:
            parent, child, angle = f, t, e["angle_from_vertical_deg"]
        else:
            parent, child = t, f
            angle = round((e["angle_from_vertical_deg"] + 180) % 360 - 180, 1)
        parent_children.setdefault(parent, []).append((child, angle))

    lines = [
        "GEOMETRY NOTE (auto-computed from vertex positions and edge angles -- use this to judge whether the "
        "drawing actually LOOKS like a tree hierarchy, not just whether the edges are logically valid):",
    ]
    for parent, children in parent_children.items():
        child_desc = ", ".join(f"'{c}' (branch angle {a} deg from vertical)" for c, a in children)
        lines.append(f"  - '{parent}' sits above and branches down to: {child_desc}")
        if len(children) > 1:
            angles = [a for _, a in children]
            spread = max(angles) - min(angles)
            if spread < 15:
                flagged.append(
                    f"Children of '{parent}' ({', '.join(c for c, _ in children)}) branch at nearly the same "
                    f"angle ({spread:.1f} deg apart) -- they visually stack/overlap instead of diverging like a "
                    f"tree's separate branches."
                )
    if flagged:
        lines.append("POTENTIAL LAYOUT ISSUES DETECTED:")
        lines.extend(f"  - {msg}" for msg in flagged)
    lines.append(
        "RULE: For tree-type topics, correctness is not just 'are the edges logically valid' (V = E + 1, "
        "connected, acyclic). The layout must also resemble a hierarchy: each parent should sit above its "
        "children, and when a parent has more than one child, the branches must diverge at visibly different "
        "angles rather than being stacked in a line or bunched together. If the graph is logically a valid tree "
        "but is NOT arranged hierarchically with clear branching as described, mark the answer INCORRECT or "
        "PARTIALLY CORRECT and explain the layout problem using the vertex labels above."
    )
    return "\n".join(lines)

def build_geometry_note(topic: str, current_graph_str: str) -> str:
    """Entry point used by /api/evaluate-answer. Returns '' for topics where
    position/angle is cosmetic (plain Graphs, algorithms, automata, etc.)."""
    mode = classify_geometry_mode(topic)
    if mode == "none":
        return ""
    geo = parse_graph_geometry(current_graph_str)
    if not geo:
        return ""
    if mode == "extrema":
        return build_extrema_geometry_note(geo)
    if mode == "tree":
        return build_tree_geometry_note(geo)
    return ""

SYSTEM_PROMPT = (
    "You are an excellent Professor in Graph Theory, Digital Electronics, Theory of Computation, Data Structures and Algorithms. \n"
    "CRITICAL: You must start all your questions for all topics in the dropdown with a scenario that matches a particular tpoic. \n"
    "TASK: Generate unique questions about the target topic provided. Also give a short explaination to the user about what the particular topic is. \n"
    "EVALUATION CRITERIA FOR VERTEX COLORS: Vertices inside current_graph now include a 'color' attribute mapping to hex codes chosen by the user. "
    "For standard algorithmic problems (e.g., shortest path, trees, network flow), ignore the color attribute entirely during verification. "
    "However, if the topic is 'Graph Colouring' or 'Map Coloring', you must strictly parse and verify the hex colors assigned to each vertex. Check if adjacent vertices share the same color, and evaluate if the solution accurately matches the problem constraints and total chromatic number requirement.\n"
    "CHALLENGE TYPES: \n"
    "1. TEXT CHALLENGE: For basic topics, ask the user to DRAW a graph. \n"
    "2. GRAPH CHALLENGE: For algorithms (Dijkstra, Floyd-Warshall, Bellman-Ford), you MUST provide the solution for the problem asked by the users as a graph. \n"
    "RULES (STRICT):\n"
    "1. When evaluating a user's response: If the user did not draw a graph or the solution is completely missing, you MUST start the response with 'INCORRECT...'. If a graph is drawn but contains errors, start with 'INCORRECT...' or 'PARTIALLY CORRECT...' based on the accuracy. Only start with 'CORRECT...' if the solution is 100% accurate. Never default to 'CORRECT'.\n"
    "2. If the topic is 'Floyd-Warshall-Algorithm', 'Bellman-Ford-Algorithm', or 'Dijkstra-Algorithm', ask the user to draw a weighted graph AND specify the matrix size they should create in the Worksheet (e.g., 'Draw a 4-node weighted graph and solve using a 4x4 distance matrix').\n"
    "3. For 'Map Coloring', ask for a graph and the chromatic number.\n"
    "4. Max 70 words. Be academic and varied.\n"
    "5. If topic is 'Graphs', ask for specific vertex/edge counts.\n"
    "6. For Hamiltonian path, Hamiltonian circuit, Euler path, Euler circuit, ask for the order of vertices or edges to be given in the user justification area based on the actual mathematical concepts.\n"
    "7. If topic is 'Digital', ask for a state machine diagram.\n"
    "8. If topic is 'Lattice or Lattice (Partially Ordered Set)', ask for a lattice structure diagram, where lattice is a partially ordered set (POSET) and algebraic structure where every pair of elements has a unique least upper bound and a unique greatest lower bound, and it should represent hierarchy, divisibility or logic based on the question.\n"
    "9. Generate new, creative and different questions each time.\n"
    "10. Never repeat the same challenge twice.\n"
    "11. Generate your own expressions for the questions on the concepts like Automata Theory - I, Automata Theory - II and Directed Acyclic Graphs which need expressions to solve (e.g., \"Construct a DFA diagram for the given regular expression (a+b)*cd\") and ask the user to draw it.\n"
    "12. For Automata Theory - I, Automata Theory - II and Directed Acyclic Graphs, check the correctness of the number of initial AND final states formed as per the actual concept and inform the users if it is wrong.\n"
    "13. For 'TSP' or 'Travelling Salesman Problem' or 'Travelling Salesperson Problem', ask the user to solve on a weighted graph.\n"
    "14. Always correlate the question with the particular topic in graphs.\n"
    "15. The Weight button is not just to add weights for a edge. Accept if the users express weights of edges, state name in digital principles and system deign, 'inf' value for algorithms like Dijiksta, symbols like <, >, <=, >=, ==, != for problems based on lattice (poset), ε or 'null' for transitions in automata theory, balance factor for tree based problems and relevant expressions for specific problems, all in 'Weight' option itself if the logic of the problem alone is correct.\n"
    "16. The Weight input fields should act as numeric weight inputs for weighted graph and the same input fields as alphabet inputs for paths for problems like directed graphs and automata theory based on problem nature.\n"
    "17. For Prim's and Kruskal's Algorithm, specify the weights by yourself while generating the question itself.\n"
    "18. For indexing structures like B-Tree and B-Plus-Tree, accept if the user groups multiple vertices or keys into a single block structure. Understand that these grouped vertices represent a single disk block node containing multiple keys interleaved with child pointers. Validate the layout rules based on how keys and pointers are dynamically distributed across these grouped blocks.\n"
    "19. POSITION & ANGLE AWARENESS (topic-dependent, STRICT): For 'local maxima/minima' or 'extrema' topics, the vertical position of each plotted vertex is meaningful: a vertex claimed as the ABSOLUTE MAXIMUM must sit higher on the +y axis than every point it is compared to, and one claimed as the ABSOLUTE MINIMUM must sit lower. For 'Trees', 'Spanning Trees', and similar tree-hierarchy topics (but NOT B-Tree/B-Plus-Tree block structures, which follow rule 18 instead), a logically valid tree is not enough -- the layout itself must resemble a hierarchy: parents should sit above their children, and when a vertex has more than one child, those branches must diverge at visibly different angles (like the separate branches of a real tree) rather than being stacked in a line or bunched at nearly the same angle. Connected-correctly does not automatically mean tree-shaped-correctly; judge the drawn hierarchy, not just the adjacency. For every other topic (ordinary Graphs, shortest-path/flow algorithms, automata, colouring, lattices, etc.), vertex position and edge angle are purely cosmetic -- do NOT penalize a logically correct answer for how it happens to be arranged on the canvas. When a 'GEOMETRY NOTE' block appears in the user message, treat its auto-computed rankings/angles as ground truth over any raw coordinate numbers you might otherwise try to infer yourself.\n"
    "MATHEMATICAL SAFETY RULES (STRICT):\n"
    "1. For any graph with V vertices, the number of edges E must be: E <= (V * (V - 1)) / 2.\n"
    "2. NEVER ask for impossible graphs (e.g., 1 vertex and 10 edges).\n"
    "3. For 'Trees' or 'Acyclic' graphs, E must be V - 1.\n"
    "4. Never use the same numbers (nodes/edges) twice in a row.\n"
    "5. Provide a clear, drawing challenge of any number of sentences.\n"
    "SCALE & VARIETY:-\n"
    "Every time you are asked, you MUST pick a DIFFERENT category from this list:\n"
    "1. Basic: (Specific number of vertices/edges)\n"
    "2. Special Types: (Bipartite, Complete, Regular, Cycle, Path)\n"
    "3. Advanced: (Hamiltonian, Eulerian, Planar, Acyclic/Trees)\n"
    "4. Connectivity: (Connected, Disconnected with X components)\n"
    "- MANDATORY NODE COUNT PROCEDURE: Before writing the question, silently compute a node count using this exact procedure: take the current Unix timestamp salt provided in the user message, sum its digits, then take that sum modulo 13, then add 3. This yields a node count between 3 and 15 inclusive. You MUST use this computed number, not a number you find aesthetically pleasing.\n"
    "- It is a STRICT VIOLATION to produce the same node count, or a node count within one unit of the previous response's count, in two consecutive responses.\n"
    "- It is a STRICT VIOLATION to default to a habitual or 'safe' node count out of comfort; the computed number from the timestamp salt overrides any preference.\n"
    "- It is a STRICT VIOLATION to repeatedly favor node counts clustered around the midpoint of the allowed range; actively favor the extremes of the range as often as the middle.\n"
    "CREATIVITY:\n"
    "Use different terminologies: 'Bipartite', 'Wheel Graph', 'Cycle', 'Planar', 'Disconnected with 2 components'.\n"
    "PRONOUN & GENDER COMPLIANCE: When writing a scenario for the primary character provided in the mandate, verify its real-world gender identity. For females (e.g., Cinderella, Snow White, Rapunzel, Medusa, Belle, Laxmi Bhai, Velu Nachiyar, M.S Subbulakshmi, Sophia, Nila, or strings starting with 'A girl'/'A lady'/'A woman'/'A small girl'/'A young girl'), strictly use feminine pronouns (she/her/hers). For males, strictly use masculine pronouns (he/him/his). For apps, objects, or groups, use gender-neutral pronouns (they/them or it/its) correctly.\n"
    "For Spanning trees problem, start the question like, 'In a summer day' or 'In a winter evening' or 'During a spring time' or 'In a rainy day' or 'In a small village' or 'In a large city' or 'In a small town', 'In a busy street', 'In a big town' etc. along with the charecter names.\n"
    "For TSP or Travelling Salesman Problem or Travelling Salesperson Problem, generate delivery related questions like 'A delivery boy wants to deliver food for 5 houses...' using different delivery app names only from the pool_map (e.g, A delivery boy from Swiggy came to deliver food... or A boy wanted to reach Red Hills, so he booked a car in Uber...).\n"
    "For TSP related problems, show your creativity in detemining the name of the location where the charecter (delivery boy) has to go (A delivery boy have to reach the Red Hills, A man wants to go to ECR to deliver...).\n"
    "Name of the source and destination place can be anything and is optional to specify them for TSP, but specifying the company/app name from the list is mandatory.\n"
    "Use Graphs in AI Concepts problem ultimately for depth first search, breadth first search, depth limited search, uniform cost search, greedy search, A* search, graph search, simulated annealing, local beam search, generic concepts and hill climbing algorithm only.\n" 
    "Use Graphs in ML and DL Concepts problem ultimately for expressing single or multilayer or fully connected layer graph in AI and ML, input layer, hidden layer and output layer graph, Computational Graph, Network Architecture Graph, Feed Forward Neutral Network and Graph Neural Networks (GNNs) and drawing perceptrons as graphs.\n"
    "For all graph colouring problems, start like as said in the pool_map and continue like they draw and colour an image.\n"
    "For Bayesian Networks, ask to draw baysian network graphs and give more importance for burglary alarm problem, asking the users to express it in graph form.\n"
    "For Neural Networks, do not mention the word vertices and edges, instead use the term neurons and weights, but the graph connection concept is same.\n"
)

class QuestionRequest(BaseModel):
    topic: str
    nodes: Optional[int] = None
    edges: Optional[int] = None

NO_FIXED_COUNT_FALLBACK_TEMPLATES = [
    "{NAME} wants to design a small communication network. The blueprint calls for {NODES} servers and {EDGES} direct connections. Help {NAME} lay out the network.",
    "{NAME} is mapping out friendships for a new social app. There are {NODES} people to place and {EDGES} friendship links to draw between them. Build the graph for {NAME}.",
    "{NAME} is planning road links between {NODES} towns, with {EDGES} roads connecting them. Sketch the map as a graph.",
    "In a school project, {NAME} needs to represent {NODES} computers on a LAN joined by {EDGES} cables. Draw the network topology.",
    "{NAME} is organizing a tournament bracket with {NODES} players and {EDGES} scheduled matches between them. Represent this as a graph.",
    "{NAME} is designing a delivery route map covering {NODES} locations linked by {EDGES} roads. Help {NAME} draw it out.",
    "For a city-planning assignment, {NAME} must connect {NODES} neighborhoods using exactly {EDGES} bridges. Draw the resulting graph.",
    "{NAME} is building a circuit board layout with {NODES} components and {EDGES} wired connections. Represent the layout as a graph.",
]

QUESTION_BANK_PATH = os.path.join(BASE_DIR, "questions.txt")

def load_question_bank(path: str) -> dict:
    bank = {}
    if not os.path.exists(path):
        return bank
    current_topic = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("###"):
                current_topic = line.lstrip("#").strip().lower()
                bank.setdefault(current_topic, [])
            elif current_topic:
                bank[current_topic].append(line)
    return bank

QUESTION_BANK = load_question_bank(QUESTION_BANK_PATH)

def get_style_example(topic: str) -> str:
    t = (topic or "").lower()
    for key, examples in QUESTION_BANK.items():
        if examples and (key in t or t in key):
            return random.choice(examples)
    return ""

def build_question_prompt(topic: str) -> str:
    example = get_style_example(topic)
    style_clause = (
        f"STYLE REFERENCE from a past question paper (mimic its structure, phrasing style and level of detail "
        f"ONLY -- do NOT copy it verbatim or reuse its exact numbers/names): \"{example}\" "
        if example else ""
    )
    if topic_has_no_fixed_count(topic):
        hint = topic_format_hint(topic)
        return (
            f"Generate ONE short, clear, educational scenario for the topic '{topic}'. "
            f"This topic does not use a vertex/edge count format -- instead, {hint} "
            f"{style_clause}"
            f"Make it different in wording, setting and framing every time you are asked -- never reuse a "
            f"previous phrasing style. The scenario should describe a real-world-feeling situation (networks, "
            f"cities, delivery routes, friendships, circuits, tournaments, etc.) suitable for the reader to "
            f"then draw as a graph. Do NOT include any specific person's name as literal text, and instead you "
            f"MUST use exactly this placeholder, verbatim, wherever it belongs: '{{NAME}}' for the person's "
            f"name, which may appear more than once. Depending on the format, if this topic does not use a "
            f"vertex/edge count format -- instead, {hint} Otherwise, do not include any node count or edge "
            f"count as literal text, and instead you MUST use exactly these two placeholders, verbatim, "
            f"wherever they belong: '{{NODES}}' for the number of vertices, and '{{EDGES}}' for the number of "
            f"edges, where each placeholder should appear at least once. Example shape for the latter (do not "
            f"copy the wording): '{{NAME}} wants to design a communication network. The blueprint requires "
            f"{{NODES}} servers and {{EDGES}} connections. Help {{NAME}} build it.' Keep it to 1-3 sentences. "
            f"Do not solve the problem, do not output a graph structure, and do not add any preamble, "
            f"explanation, or markdown -- output only the raw scenario sentence(s) with the placeholders in place."
        )

@app.post("/api/generate-question")
async def generate_question(req: QuestionRequest):
    question_text = None

    if DRAWALGOSOLVE_AI_API_KEY:
        try:
            prompt = build_question_prompt(req.topic)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a creative assistant that writes short, varied graph-theory word-problem "
                            "scenarios using the literal placeholders {NAME}, {NODES}, and {EDGES}. You never "
                            "repeat the same phrasing twice and you never output anything except the scenario text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,
                max_tokens=200,
            )
            candidate = completion.choices[0].message.content.strip()
            if topic_has_no_fixed_count(req.topic):
                if "{NAME}" in candidate:
                    question_text = candidate
            elif "{NAME}" in candidate and "{NODES}" in candidate and "{EDGES}" in candidate:
                question_text = candidate
        except Exception as e:
            print(f"Groq question generation failed, using fallback template: {e}")

    if not question_text:
        if topic_has_no_fixed_count(req.topic):
            question_text = random.choice(NO_FIXED_COUNT_FALLBACK_TEMPLATES)
        else:
            question_text = random.choice(FALLBACK_TEMPLATES)

    return {"template": question_text, "ai_generated": question_text not in FALLBACK_TEMPLATES and question_text not in NO_FIXED_COUNT_FALLBACK_TEMPLATES}

class EvaluationRequest(BaseModel):
    topic: str
    query: str
    current_graph: str
    justification: str
    processed_matrix: str

@app.post("/api/evaluate-answer")
async def evaluate_answer(payload: EvaluationRequest):
    try:
        geometry_note = build_geometry_note(payload.topic, payload.current_graph)

        evaluation_user_prompt = (
            f"CONTEXT:\n"
            f"1. Topic: {payload.topic}.\n"
            f"2. The user was asked: '{payload.query}'.\n"
            f"3. The user's visual graph: {payload.current_graph}.\n"
            f"4. The user's text justification/answer: '{payload.justification}'.\n"
            f"5. The user has provided the following adjacency matrix: {payload.processed_matrix}.\n"
            f"Please note that '∞' or 'inf' indicates no path exists between vertices.\n"
            + (f"6. {geometry_note}\n" if geometry_note else "")
            + f"TASK:\n"
            f"1. If the user's input contains a doubt or question about the platform or graph theory in the text area.\n"
            f"(User Justification and Doubt Clarification Area), answer it helpfully and empathetically.\n"
            f"2. If the user's input is a justification for their solution, evaluate the graph and matrix.\n"
            f"3. Validate if they solved it correctly. If they provided a numerical answer in the text box, check it against the graph.\n"
            f"4. Start with 'CORRECT:' or 'INCORRECT:' followed by a brief explanation for all graphs.\n"
            f"5. Start with normal sentence it it was a doubt or question and explain in short and accurate way.\n"
            f"6. The user justification area need to be filled by the user compulsarily, but, it is not mandatory.\n"
            f"that the justification must be correct, rather check only the graph drawn more importantly.\n"
            + (
                "7. A GEOMETRY NOTE is included above in CONTEXT item 6 -- for this topic, vertex position and/or "
                "edge branching angle is part of correctness. Weigh it alongside logical connectivity; a "
                "logically-connected-but-wrongly-arranged diagram should not be marked fully CORRECT.\n"
                if geometry_note else
                "7. This topic does not carry positional meaning -- judge only logical connectivity/correctness "
                "and ignore where the user happened to place vertices or the angles of edges on the canvas.\n"
            )
        )
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": evaluation_user_prompt}
            ],
            temperature=0.1
        )
        return {"evaluation": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-graph")
async def process_graph(payload: dict):
    try:
        if payload.get("mode") == "USER_ASK":
            query = payload.get("query", "")
            topic = payload.get("topic", "General")
            prompt = (
                f"Identify the nodes and mathematical edges needed to properly illustrate: {query} for the topic: {topic}. "
                f"Produce a completely valid JSON structure containing exactly two keys: 'vertices' (a list of objects with 'label', 'x', and 'y') and 'edges' (a list of objects with 'source' and 'target' pointing to vertex labels). "
                f"CRITICAL RULES:\n"
                f"1. You MUST include all appropriate connections inside the 'edges' list to fully satisfy the query logic.\n"
                f"2. Each vertex must have a unique alphanumeric 'label'.\n"
                f"3. Each edge object must strictly use the keys 'from_label', 'to_label', and 'weight'.\n"
                f"4. Add weights and directions for the edges of the graphs and self loops for the vertices of the graphs whenever required.\n"
                f"5. Place 'x' coordinates strictly between 150 and 780, and 'y' coordinates strictly between 100 and 340, distributing them uniformly so nothing overlaps.\n"
                f"6. OUTPUT ONLY THE RAW JSON OBJECT. DO NOT inclusion text explanations, markdown blocks, or ```json formatting wrappers."
            )
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a strict automated backend system that only outputs structured graph definitions matching the exact schema request. You never talk or wrap output in markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            clean_response = completion.choices[0].message.content.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.replace("```json", "").replace("```", "").strip()
            return {"response": clean_response}
        else:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a strict automated backend system that only outputs structured graph definitions matching the exact schema request. You never talk, explain, or wrap output in markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            raw_content = completion.choices[0].message.content.strip()
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].split("```")[0].strip()
            return {"response": raw_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/", response_class=HTMLResponse)
def root_screen(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/signin", response_class=HTMLResponse)
def signin_screen(request: Request):
    return templates.TemplateResponse("signin.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
def signup_screen(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/welcome", response_class=HTMLResponse)
def welcome_screen(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})

@app.get("/drawalgosolve", response_class=HTMLResponse)
def main_app(request: Request):
    return templates.TemplateResponse("drawalgosolve.html", {"request": request})

@app.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request):
    try:
        raw_profile = get_or_create_profile()
        
        if raw_profile is None: 
            profile = {"username": "Guest", "email": "No email"}
        elif hasattr(raw_profile, "__dict__"):
            profile = {c.name: getattr(raw_profile, c.name) for c in raw_profile.__table__.columns}
        else:
            profile = dict(raw_profile)
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        profile = {"username": "Error Loading Profile", "email": ""}

    return templates.TemplateResponse("profile.html", {"request": request, "profile": profile})


@app.get("/logout")
def logout():
    return {"redirect": "index.html"}

class TopicStat(BaseModel):
    topic: str
    count: int = 0

@app.get("/random-question")
def get_random_question():
    if not dataset:
        raise HTTPException(status_code=404, detail="No questions loaded.")
    return {"question": random.choice(dataset)}

@app.post("/update-stats")
def update_stats(stat: TopicStat):
    profile = get_or_create_profile()
    current_stats = profile.get("stats", {})
    if stat.topic != "total":
        current_stats[stat.topic] = current_stats.get(stat.topic, 0) + 1
        current_stats["total"]    = current_stats.get("total",    0) + 1
    db.update({"stats": current_stats}, User.type == 'profile_data')
    return {"status": "success", "stats": current_stats}

@app.get("/get-stats")
def get_stats():
    profile = get_or_create_profile()
    return {"stats": profile.get("stats", {})}

@app.post("/save-profile")
async def save_profile(request: Request):
    form_data = await request.form()
    updated_fields = {
        "username": form_data.get("username", ""),
        "middlename": form_data.get("middlename", ""),
        "lastname": form_data.get("lastname", ""),
        "email": form_data.get("email", ""),
        "dept": form_data.get("dept", ""),
        "mobile_no": form_data.get("mobile-no", ""),
        "birthday": form_data.get("birthday", ""),
        "github_url": form_data.get("github-url", ""),
        "coll_name": form_data.get("coll-name", ""),
        "location": form_data.get("location", ""),
        "gender": form_data.get("gender", ""),
        "linkedin": form_data.get("linkedin", "")
    }
    db.update(updated_fields, User.type == 'profile_data')
    return {"status": "success", "message": "Profile synced successfully"}

@app.post("/add-skill")
async def add_skill(request: Request):
    data = await request.json()
    new_skill = data.get("skill", "").strip()
    profile = get_or_create_profile()
    current_skills = profile.get("skills", [])
    if new_skill and new_skill not in current_skills:
        current_skills.append(new_skill)
        db.update({"skills": current_skills}, User.type == 'profile_data')
    return {"status": "success", "skills": current_skills}

@app.post("/delete-skill")
async def delete_skill(request: Request):
    data = await request.json()
    skill_to_remove = data.get("skill", "").strip()
    profile = get_or_create_profile()
    current_skills = profile.get("skills", [])
    if skill_to_remove in current_skills:
        current_skills.remove(skill_to_remove)
        db.update({"skills": current_skills}, User.type == 'profile_data')
    return {"status": "success", "skills": current_skills}

@app.post("/delete-account")
async def delete_account():
    db.update({"stats": {}, "skills": [], "certificates": []}, User.type == 'profile_data')
    db.update({"username": "", "middlename": "", "lastname": "", "email": "", "dept": "",
                "mobile_no": "", "birthday": "", "github_url": "", "coll_name": "",
                "location": "", "gender": "", "linkedin": ""}, User.type == 'profile_data')
    return {"status": "deleted"}


@app.post("/upload-certificate")
async def upload_certificate(request: Request):
    form_data = await request.form()
    file = form_data.get("certificate_file")
    title = form_data.get("title", "Untitled Certificate").strip()
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        import time
        unique_filename = f"cert_{int(time.time())}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)
            
        profile = get_or_create_profile()
        current_certs = profile.get("certificates", [])
        cert_record = {
            "id": f"cert_{len(current_certs) + 1}",
            "title": title if title else filename,
            "filename": unique_filename
        }
        current_certs.append(cert_record)
        db.update({"certificates": current_certs}, User.type == 'profile_data')
        return {"status": "success", "certificate": cert_record}
    return {"status": "error", "message": "Disallowed profile media sequence"}

from fastapi.responses import FileResponse

@app.get("/view-certificate/{filename}")
def view_certificate(filename: str):
    return FileResponse(os.path.join(UPLOAD_FOLDER, secure_filename(filename)))

@app.get("/download-certificate/{filename}")
def download_certificate(filename: str):
    return FileResponse(os.path.join(UPLOAD_FOLDER, secure_filename(filename)), filename=filename)

@app.post("/shutdown-python")
def shutdown_python():
    try:
        subprocess.run(["taskkill", "/F", "/IM", "pythonw.exe", "/TT"], check=True)
        return {"status": "Process terminated"}
    except Exception as e:
        return {"error": str(e), "status_code": 200}

@app.route('/certificate/<cert_id>')
def view_certificate_pdf(cert_id):
    import pdfkit
    html = render_template('certificate.html', cert_id=cert_id)
    pdf = pdfkit.from_string(html, False)
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=certificate.pdf'
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7654)