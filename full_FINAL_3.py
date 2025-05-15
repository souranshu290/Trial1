import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import math
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from io import BytesIO
import base64

# ======================== SHAPE GENERATORS ========================
def circle_shape(r, cx=0, cy=0, segments=20):
    """Generate a circle with given radius and center"""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

def polygon_shape(n_sides, r, cx=0, cy=0):
    """Generate a regular polygon with n sides"""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

def ellipse_shape(r, aspect_ratio, cx=0, cy=0, segments=30):
    """Generate an ellipse with given aspect ratio"""
    angles = np.linspace(0, 2 * np.pi, segments)
    return [(cx + r * np.cos(a), cy + r * aspect_ratio * np.sin(a)) for a in angles]

def star_shape(n_points, r, cx=0, cy=0):
    """Generate a star shape with n points"""
    angles = np.linspace(0, 2 * np.pi, n_points * 2, endpoint=False)
    radius = [r if i % 2 == 0 else r * 0.5 for i in range(len(angles))]
    return [(cx + radius[i] * np.cos(angles[i]), cy + radius[i] * np.sin(angles[i])) for i in range(len(angles))]

def superellipse_shape(r, n=4, cx=0, cy=0, segments=50):
    """Generate a superellipse (squircle) shape"""
    angles = np.linspace(0, 2 * np.pi, segments)
    points = []
    for a in angles:
        x = r * np.sign(np.cos(a)) * abs(np.cos(a))**(2/n)
        y = r * np.sign(np.sin(a)) * abs(np.sin(a))**(2/n)
        points.append((cx + x, cy + y))
    return points

def lemniscate_shape(r, cx=0, cy=0, segments=50):
    """Generate a lemniscate (infinity symbol) shape"""
    t = np.linspace(0, 2 * np.pi, segments)
    a = r / np.sqrt(2)
    x = a * np.sqrt(2) * np.cos(t) / (np.sin(t)**2 + 1)
    y = a * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)
    return [(cx + x[i], cy + y[i]) for i in range(len(t))]

def gear_shape(r, teeth=12, pressure_angle=20, cx=0, cy=0):
    """Generate a gear shape with specified teeth"""
    pa = math.radians(pressure_angle)
    base_r = r * math.cos(pa)
    points = []
    for i in range(teeth * 4):
        angle = 2 * math.pi * i / (teeth * 4)
        inv_angle = math.sqrt((r / base_r)**2 - 1)
        x = base_r * (math.cos(angle) + (angle - inv_angle) * math.sin(angle))
        y = base_r * (math.sin(angle) - (angle - inv_angle) * math.cos(angle))
        points.append((cx + x, cy + y))
    return points

def hypotrochoid_shape(r, cx=0, cy=0, segments=100):
    """Generate a hypotrochoid shape"""
    t = np.linspace(0, 2 * np.pi, segments)
    a, b, h = r, r/3, r/2
    x = (a - b) * np.cos(t) + h * np.cos((a - b)/b * t)
    y = (a - b) * np.sin(t) - h * np.sin((a - b)/b * t)
    return [(cx + x[i], cy + y[i]) for i in range(len(t))]

def flower_shape(r, petals=5, cx=0, cy=0, segments=50):
    """Generate a flower shape with specified petals"""
    angles = np.linspace(0, 2 * np.pi, segments)
    points = []
    for a in angles:
        radius = r * (0.5 + 0.5 * np.sin(petals * a))
        x = cx + radius * np.cos(a)
        y = cy + radius * np.sin(a)
        points.append((x, y))
    return points

def random_shape(r, cx=0, cy=0, segments=30):
    """Generate a random organic shape"""
    base_circle = circle_shape(r, cx, cy, segments)
    base_circle = np.array(base_circle)
    
    # Add random perturbations
    perturbations = np.random.normal(0, 0.2*r, (segments, 2))
    smoothed_perturbations = np.zeros_like(perturbations)
    
    # Smooth the perturbations
    for i in range(2):
        tck, _ = splprep([perturbations[:, i]], s=segments)
        smoothed_perturbations[:, i] = splev(np.linspace(0, 1, segments), tck)[0]
    
    random_shape = base_circle + smoothed_perturbations
    return [(p[0], p[1]) for p in random_shape]

# ======================== THREAD GENERATION ========================
def generate_threads(Lx, Ly, min_r, max_r, num_threads, materials, packing='grid'):
    """
    Generate thread positions with different packing algorithms
    packing options: 'grid', 'random', 'hexagonal', 'poisson'
    """
    threads = []
    placed = 0
    max_attempts = num_threads * 1000
    
    if packing == 'grid':
        # Grid-based placement with jitter
        grid_size = int(np.sqrt(num_threads) * 1.2)
        x_grid = np.linspace(max_r, Lx - max_r, grid_size)
        y_grid = np.linspace(max_r, Ly - max_r, grid_size)
        
        for x in x_grid:
            for y in y_grid:
                if placed >= num_threads:
                    break
                
                # Add some jitter to the grid positions
                jitter_x = np.random.uniform(-max_r/2, max_r/2)
                jitter_y = np.random.uniform(-max_r/2, max_r/2)
                x_jittered = x + jitter_x
                y_jittered = y + jitter_y
                
                if x_jittered < max_r or x_jittered > Lx - max_r:
                    continue
                if y_jittered < max_r or y_jittered > Ly - max_r:
                    continue
                
                r = np.random.uniform(min_r, max_r)
                material = random.choice(materials)
                waviness = np.random.uniform(0.05 * r, 0.2 * r)
                roughness = np.random.uniform(0.03, 0.1)
                max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
                
                if not check_collision(x_jittered, y_jittered, r, threads):
                    threads.append((x_jittered, y_jittered, r, material, waviness, roughness, max_length))
                    placed += 1
    
    elif packing == 'hexagonal':
        # Hexagonal close packing
        spacing = max_r * 2 * 1.05  # 5% spacing between fibers
        rows = int((Ly - 2*max_r) / (spacing * np.sqrt(3)/2)) + 1
        cols = int((Lx - 2*max_r) / spacing) + 1
        
        for row in range(rows):
            for col in range(cols):
                if placed >= num_threads:
                    break
                
                x = max_r + col * spacing
                if row % 2 == 1:
                    x += spacing / 2
                
                y = max_r + row * spacing * np.sqrt(3)/2
                
                if x > Lx - max_r or y > Ly - max_r:
                    continue
                
                r = np.random.uniform(min_r, max_r)
                material = random.choice(materials)
                waviness = np.random.uniform(0.05 * r, 0.2 * r)
                roughness = np.random.uniform(0.03, 0.1)
                max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
                
                if not check_collision(x, y, r, threads):
                    threads.append((x, y, r, material, waviness, roughness, max_length))
                    placed += 1
    
    elif packing == 'poisson':
        # Poisson disk sampling for more organic distribution
        points = []
        active = []
        
        # First point
        x, y = np.random.uniform(max_r, Lx - max_r), np.random.uniform(max_r, Ly - max_r)
        points.append((x, y))
        active.append((x, y))
        
        k = 30  # Number of attempts before rejection
        
        while active and placed < num_threads:
            idx = np.random.randint(0, len(active))
            x, y = active[idx]
            found = False
            
            for _ in range(k):
                angle = np.random.uniform(0, 2*np.pi)
                distance = np.random.uniform(2*min_r, 2*max_r)
                new_x = x + distance * np.cos(angle)
                new_y = y + distance * np.sin(angle)
                
                if (new_x < max_r or new_x > Lx - max_r or 
                    new_y < max_r or new_y > Ly - max_r):
                    continue
                
                valid = True
                for (px, py) in points:
                    if np.sqrt((px - new_x)**2 + (py - new_y)**2) < 2*min_r:
                        valid = False
                        break
                
                if valid:
                    points.append((new_x, new_y))
                    active.append((new_x, new_y))
                    found = True
                    break
            
            if not found:
                active.pop(idx)
        
        for x, y in points[:num_threads]:
            r = np.random.uniform(min_r, max_r)
            material = random.choice(materials)
            waviness = np.random.uniform(0.05 * r, 0.2 * r)
            roughness = np.random.uniform(0.03, 0.1)
            max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
            threads.append((x, y, r, material, waviness, roughness, max_length))
            placed += 1
    
    else:  # random
        # Pure random placement with collision checking
        attempts = 0
        while placed < num_threads and attempts < max_attempts:
            x = np.random.uniform(max_r, Lx - max_r)
            y = np.random.uniform(max_r, Ly - max_r)
            r = np.random.uniform(min_r, max_r)
            material = random.choice(materials)
            waviness = np.random.uniform(0.05 * r, 0.2 * r)
            roughness = np.random.uniform(0.03, 0.1)
            max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
            
            if not check_collision(x, y, r, threads):
                threads.append((x, y, r, material, waviness, roughness, max_length))
                placed += 1
            attempts += 1
    
    placement_ratio = placed / num_threads * 100
    st.info(f"Successfully placed {placed}/{num_threads} threads ({placement_ratio:.1f}%) using {packing} packing")
    
    if placement_ratio < 90:
        st.warning("Low placement percentage. Consider increasing domain size or decreasing thread count.")
    
    return threads

def check_collision(x, y, r, existing_threads, safety_margin=1.05):
    """Check if a new thread would collide with existing ones"""
    if not existing_threads:
        return False
    
    existing_pos = np.array([[t[0], t[1]] for t in existing_threads])
    existing_rad = np.array([t[2] * safety_margin for t in existing_threads])
    
    distances = np.sqrt((existing_pos[:, 0] - x)**2 + (existing_pos[:, 1] - y)**2)
    return np.any(distances < (r * safety_margin + existing_rad))

# ======================== 3D THREAD RENDERING ========================
def generate_thread_layers(x, y, r, waviness, roughness, height, segments, shape_type, shape_params):
    """Generate 3D layers for a single thread"""
    layers = []
    z_values = np.linspace(0, height, segments + 1)
    
    # Generate base shape
    if shape_type == "circle":
        base_shape = circle_shape(r, 0, 0)
    elif shape_type == "polygon":
        base_shape = polygon_shape(shape_params['sides'], r, 0, 0)
    elif shape_type == "ellipse":
        base_shape = ellipse_shape(r, shape_params['aspect_ratio'], 0, 0)
    elif shape_type == "star":
        base_shape = star_shape(shape_params['points'], r, 0, 0)
    elif shape_type == "superellipse":
        base_shape = superellipse_shape(r, shape_params['n'], 0, 0)
    elif shape_type == "lemniscate":
        base_shape = lemniscate_shape(r, 0, 0)
    elif shape_type == "gear":
        base_shape = gear_shape(r, shape_params['teeth'], shape_params.get('pressure_angle', 20), 0, 0)
    elif shape_type == "hypotrochoid":
        base_shape = hypotrochoid_shape(r, 0, 0)
    elif shape_type == "flower":
        base_shape = flower_shape(r, shape_params['petals'], 0, 0)
    elif shape_type == "random":
        base_shape = random_shape(r, 0, 0)
    else:
        base_shape = circle_shape(r, 0, 0)
    
    base_shape = np.array(base_shape)
    
    # Add some twist to the shape
    twist_factor = shape_params.get('twist', 0)  # rotations per unit height
    
    for i, z in enumerate(z_values):
        # Current radius modulation
        r_mod = r * (1 + roughness * np.sin(2 * np.pi * z / height))
        
        # Position modulation (waviness)
        cx = x + waviness * np.sin(2 * np.pi * z / height)
        cy = y + waviness * np.cos(2 * np.pi * z / height)
        
        # Scale and rotate the base shape
        scaled_shape = base_shape * (r_mod / r)
        
        # Apply twist if specified
        if twist_factor != 0:
            angle = 2 * np.pi * z * twist_factor
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
            scaled_shape = np.dot(scaled_shape, rot_matrix)
        
        # Translate to final position
        translated_shape = scaled_shape + np.array([cx, cy])
        layers.append([(p[0], p[1], z) for p in translated_shape])
    
    return layers

def render_3d_view(threads, Lx, Ly, Lz, shape_type, shape_params, view='front', color_by='material'):
    """Render a 3D view of threads with specified viewpoint"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colormaps for different coloring schemes
    if color_by == 'material':
        materials = list(set([t[3] for t in threads]))
        material_colors = plt.cm.tab20(np.linspace(0, 1, len(materials)))
        color_map = {mat: color for mat, color in zip(materials, material_colors)}
    elif color_by == 'radius':
        radii = [t[2] for t in threads]
        norm = plt.Normalize(min(radii), max(radii))
        cmap = plt.cm.viridis
    elif color_by == 'height':
        heights = [t[6] for t in threads]
        norm = plt.Normalize(min(heights), max(heights))
        cmap = plt.cm.plasma
    
    for idx, (x, y, r, material, waviness, roughness, max_length) in enumerate(threads):
        height = Lz * max_length
        layers = generate_thread_layers(x, y, r, waviness, roughness, height, 10, shape_type, shape_params)
        
        # Determine color based on coloring scheme
        if color_by == 'material':
            color = color_map[material]
        elif color_by == 'radius':
            color = cmap(norm(r))
        elif color_by == 'height':
            color = cmap(norm(max_length))
        
        for i in range(len(layers) - 1):
            verts = []
            n = len(layers[i])
            for j in range(n):
                k = (j + 1) % n
                quad = [layers[i][j], layers[i][k], layers[i+1][k], layers[i+1][j]]
                verts.append(quad)
            
            poly = Poly3DCollection(verts, alpha=0.8)
            poly.set_facecolor(color)
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_box_aspect([Lx, Ly, Lz])
    
    # Set view angle
    if view == 'front':
        ax.view_init(elev=20, azim=30)
    elif view == 'top':
        ax.view_init(elev=90, azim=0)
    elif view == 'side':
        ax.view_init(elev=0, azim=0)
    elif view == 'isometric':
        ax.view_init(elev=30, azim=45)
    
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    return buf

# ======================== STREAMLIT APP ========================
def main():
    st.set_page_config(layout="wide", page_title="Advanced Composite Thread Simulator")
    st.title("Advanced Composite Thread Simulator")
    
    with st.sidebar:
        st.header("Simulation Parameters")
        shape_type = st.selectbox(
            "Thread cross-section shape",
            ["circle", "polygon", "ellipse", "star", "superellipse", "lemniscate", "gear", "hypotrochoid", "flower", "random"],
            index=0
        )
        
        shape_params = {}
        if shape_type == "polygon":
            shape_params['sides'] = st.slider("Number of sides", 3, 12, 6)
        elif shape_type == "ellipse":
            shape_params['aspect_ratio'] = st.slider("Aspect ratio", 0.1, 5.0, 1.5)
        elif shape_type == "star":
            shape_params['points'] = st.slider("Number of points", 5, 20, 5)
        elif shape_type == "superellipse":
            shape_params['n'] = st.slider("Exponent", 2.0, 10.0, 4.0)
        elif shape_type == "gear":
            shape_params['teeth'] = st.slider("Number of teeth", 8, 40, 12)
            shape_params['pressure_angle'] = st.slider("Pressure angle", 10.0, 30.0, 20.0)
        elif shape_type == "flower":
            shape_params['petals'] = st.slider("Number of petals", 3, 12, 5)
        elif shape_type == "random":
            shape_params['twist'] = st.slider("Twist factor", 0.0, 2.0, 0.5)
        
        Lx = st.number_input("Domain length (X)", min_value=1.0, max_value=100.0, value=10.0)
        Ly = st.number_input("Domain width (Y)", min_value=1.0, max_value=100.0, value=10.0)
        Lz = st.number_input("Domain height (Z)", min_value=1.0, max_value=50.0, value=5.0)
        
        num_threads = st.slider("Number of threads", 10, 5000, 200)
        min_d = st.number_input("Minimum diameter", min_value=0.01, max_value=5.0, value=0.1)
        max_d = st.number_input("Maximum diameter", min_value=0.01, max_value=5.0, value=0.3)
        
        packing = st.selectbox(
            "Packing algorithm",
            ["grid", "hexagonal", "poisson", "random"],
            index=2
        )
        
        materials = st.multiselect(
            "Materials",
            ["carbon", "glass", "kevlar", "steel", "nylon", "copper"],
            default=["carbon", "glass", "kevlar"]
        )
        
        color_by = st.selectbox(
            "Color by",
            ["material", "radius", "height"],
            index=0
        )
        
        if st.button("Run Simulation"):
            with st.spinner("Generating threads..."):
                threads = generate_threads(
                    Lx, Ly, min_d/2, max_d/2, num_threads, 
                    materials, packing
                )
            
            st.session_state.threads = threads
            st.session_state.Lx = Lx
            st.session_state.Ly = Ly
            st.session_state.Lz = Lz
            st.session_state.shape_type = shape_type
            st.session_state.shape_params = shape_params
            st.session_state.color_by = color_by
            st.success("Simulation complete!")
    
    if 'threads' in st.session_state:
        threads = st.session_state.threads
        Lx = st.session_state.Lx
        Ly = st.session_state.Ly
        Lz = st.session_state.Lz
        shape_type = st.session_state.shape_type
        shape_params = st.session_state.shape_params
        color_by = st.session_state.color_by
        
        st.header("3D Views")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Front View")
            front_view = render_3d_view(threads, Lx, Ly, Lz, shape_type, shape_params, 'front', color_by)
            st.image(front_view)
            
            st.subheader("Top View")
            top_view = render_3d_view(threads, Lx, Ly, Lz, shape_type, shape_params, 'top', color_by)
            st.image(top_view)
        
        with col2:
            st.subheader("Side View")
            side_view = render_3d_view(threads, Lx, Ly, Lz, shape_type, shape_params, 'side', color_by)
            st.image(side_view)
            
            st.subheader("Isometric View")
            iso_view = render_3d_view(threads, Lx, Ly, Lz, shape_type, shape_params, 'isometric', color_by)
            st.image(iso_view)

if __name__ == "__main__":
    main()
