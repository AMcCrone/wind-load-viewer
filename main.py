import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple

def rotate_points(points: List[Tuple[float, float]], angle_deg: float) -> List[Tuple[float, float]]:
    """Rotate points around origin by angle in degrees"""
    angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    rotated_points = []
    for x, y in points:
        rotated = rot_matrix @ np.array([x, y])
        rotated_points.append((rotated[0], rotated[1]))
    
    return rotated_points

def create_building_mesh(length, width, height, inset=False, inset_height=0, inset_distance=0):
    """Create building vertices and faces for 3D visualization"""
    # Base building coordinates (centered at origin)
    half_length = length / 2
    half_width = width / 2
    
    # Base building vertices (bottom face)
    vertices = [
        [-half_length, -half_width, 0],  # 0: bottom SW
        [half_length, -half_width, 0],   # 1: bottom SE
        [half_length, half_width, 0],    # 2: bottom NE
        [-half_length, half_width, 0],   # 3: bottom NW
    ]
    
    # Add top vertices
    if inset and inset_height > 0 and inset_distance > 0:
        # First add the middle level vertices (at height-inset_height)
        mid_height = height - inset_height
        vertices.extend([
            [-half_length, -half_width, mid_height],  # 4: middle SW
            [half_length, -half_width, mid_height],   # 5: middle SE
            [half_length, half_width, mid_height],    # 6: middle NE
            [-half_length, half_width, mid_height],   # 7: middle NW
        ])
        
        # Then add the top level vertices (inset)
        inset_half_length = half_length - inset_distance
        inset_half_width = half_width - inset_distance
        vertices.extend([
            [-inset_half_length, -inset_half_width, mid_height],  # 8: inset bottom SW
            [inset_half_length, -inset_half_width, mid_height],   # 9: inset bottom SE
            [inset_half_length, inset_half_width, mid_height],    # 10: inset bottom NE
            [-inset_half_length, inset_half_width, mid_height],   # 11: inset bottom NW
            [-inset_half_length, -inset_half_width, height],      # 12: inset top SW
            [inset_half_length, -inset_half_width, height],       # 13: inset top SE
            [inset_half_length, inset_half_width, height],        # 14: inset top NE
            [-inset_half_length, inset_half_width, height],       # 15: inset top NW
        ])
        
        # Define faces (triangles)
        i = np.array([
            # Base building - bottom face
            0, 0, 0,
            # Base building - side faces (Elevation 1-4)
            0, 1, 2, 0, 2, 3,  # Elevation 1 (bottom)
            4, 5, 1, 4, 1, 0,  # Elevation 2 (right)
            7, 6, 5, 7, 5, 4,  # Elevation 3 (top)
            3, 2, 6, 3, 6, 7,  # Elevation 4 (left)
            # Middle level top face (with hole for inset)
            4, 0, 3, 4, 3, 7,
            4, 5, 1, 4, 1, 0,
            # Inset - bottom face is part of the middle level top face
            # Inset - side faces
            8, 9, 13, 8, 13, 12,  # Inset Elevation 1
            9, 10, 14, 9, 14, 13, # Inset Elevation 2
            10, 11, 15, 10, 15, 14, # Inset Elevation 3
            11, 8, 12, 11, 12, 15, # Inset Elevation 4
            # Inset - top face
            12, 13, 14, 12, 14, 15
        ])
        
        j = np.array([
            # Base building - bottom face
            1, 2, 3,
            # Base building - side faces (Elevation 1-4)
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            3, 2, 6, 3, 6, 7,
            7, 6, 2, 7, 2, 3,
            # Middle level top face (with hole for inset)
            5, 1, 2, 5, 2, 6,
            7, 4, 0, 7, 0, 3,
            # Inset - side faces
            9, 13, 12, 8, 12, 8,
            10, 14, 13, 9, 13, 9,
            11, 15, 14, 10, 14, 10,
            8, 12, 15, 11, 15, 11,
            # Inset - top face
            13, 14, 15, 12, 15, 12
        ])
        
        k = np.array([
            # Base building - bottom face
            2, 3, 0,
            # Base building - side faces (Elevation 1-4)
            5, 1, 2, 5, 2, 6,
            1, 0, 4, 1, 4, 5,
            2, 6, 7, 2, 7, 3,
            6, 7, 3, 6, 3, 2,
            # Middle level top face (with hole for inset)
            1, 0, 3, 1, 3, 2,
            5, 4, 0, 5, 0, 1,
            # Inset - side faces
            13, 9, 8, 13, 8, 12,
            14, 10, 9, 14, 9, 13,
            15, 11, 10, 15, 10, 14,
            12, 8, 11, 12, 11, 15,
            # Inset - top face
            14, 13, 12, 14, 12, 15
        ])
        
    else:
        # Regular building without inset
        vertices.extend([
            [-half_length, -half_width, height],  # 4: top SW
            [half_length, -half_width, height],   # 5: top SE
            [half_length, half_width, height],    # 6: top NE
            [-half_length, half_width, height],   # 7: top NW
        ])
        
        # Define faces (triangles for mesh3d)
        i = np.array([
            # Bottom face
            0, 0, 0,
            # Side faces (Elevation 1-4)
            0, 1, 5, 0, 5, 4,  # Elevation 1 (South)
            1, 2, 6, 1, 6, 5,  # Elevation 2 (East)
            2, 3, 7, 2, 7, 6,  # Elevation 3 (North)
            3, 0, 4, 3, 4, 7,  # Elevation 4 (West)
            # Top face
            4, 5, 6, 4, 6, 7
        ])
        
        j = np.array([
            # Bottom face
            1, 2, 3,
            # Side faces (Elevation 1-4)
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
            0, 1, 5, 0, 5, 4,
            # Top face
            5, 6, 7, 4, 7, 4
        ])
        
        k = np.array([
            # Bottom face
            2, 3, 0,
            # Side faces (Elevation 1-4)
            5, 1, 0, 5, 0, 4,
            6, 2, 1, 6, 1, 5,
            7, 3, 2, 7, 2, 6,
            4, 0, 3, 4, 3, 7,
            # Top face
            6, 5, 4, 6, 4, 7
        ])
    
    x, y, z = zip(*vertices)
    return x, y, z, i, j, k

def plot_building_3d(length, width, height, orientation, inset=False, inset_height=0, inset_distance=0):
    """Create 3D plot of building with orientation"""
    fig = go.Figure()
    
    # Create the building mesh
    x, y, z, i, j, k = create_building_mesh(length, width, height, inset, inset_height, inset_distance)
    
    # Apply rotation to x,y coordinates based on orientation
    rotated_points = rotate_points(list(zip(x, y)), orientation)
    x_rot, y_rot = zip(*rotated_points)
    
    # Add building mesh
    fig.add_trace(go.Mesh3d(
        x=x_rot, y=y_rot, z=z,
        i=i, j=j, k=k,
        opacity=0.8,
        colorscale=[[0, 'lightblue'], [1, 'darkblue']],
        intensity=np.ones(len(x)),
        name='Building'
    ))
    
    # Add coordinate axes
    axis_length = max(length, width, height) * 1.5
    
    # North arrow and compass
    arrow_length = axis_length * 0.3
    north_x = [0, 0, 0, 0.05*arrow_length, 0, -0.05*arrow_length]
    north_y = [0, arrow_length, 0.8*arrow_length, arrow_length, 0.8*arrow_length, arrow_length]
    north_z = [0, 0, 0, 0, 0, 0]
    
    # Apply rotation to North arrow
    north_points_rot = rotate_points(list(zip(north_x, north_y)), orientation)
    north_x_rot, north_y_rot = zip(*north_points_rot)
    
    fig.add_trace(go.Scatter3d(
        x=north_x_rot, y=north_y_rot, z=north_z,
        mode='lines',
        line=dict(color='red', width=5),
        name='North'
    ))
    
    # Add "N" label for North
    n_label_x, n_label_y = rotate_points([(0, 1.1*arrow_length)], orientation)[0]
    fig.add_trace(go.Scatter3d(
        x=[n_label_x], y=[n_label_y], z=[0],
        mode='text',
        text=['N'],
        textposition='middle center',
        textfont=dict(size=14, color='red'),
        name='North Label'
    ))
    
    # Add elevation labels
    half_length = length / 2
    half_width = width / 2
    mid_height = height / 2
    
    # Calculate positions for elevation labels
    elevation_points = [
        [0, -half_width - axis_length*0.1, mid_height],  # Elevation 1 (South)
        [half_length + axis_length*0.1, 0, mid_height],  # Elevation 2 (East)
        [0, half_width + axis_length*0.1, mid_height],   # Elevation 3 (North)
        [-half_length - axis_length*0.1, 0, mid_height]  # Elevation 4 (West)
    ]
    
    # Apply rotation to elevation label positions
    for i, point in enumerate(elevation_points):
        rotated = rotate_points([(point[0], point[1])], orientation)[0]
        elevation_points[i][0] = rotated[0]
        elevation_points[i][1] = rotated[1]
    
    # Add elevation labels
    for i, point in enumerate(elevation_points):
        fig.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]],
            mode='text',
            text=[f'Elevation {i+1}'],
            textposition='middle center',
            textfont=dict(size=12, color='black'),
            name=f'Elevation {i+1}'
        ))
    
    # Update layout
    fig.update_layout(
        title='Building 3D Model',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0, y=1),
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        )
    )
    
    return fig

def main():
    st.title("Wind Load Calculator - BS EN 1991-1-4")
    
    st.header("Building Geometry Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        length = st.number_input("Building Length (d) [m]", min_value=1.0, value=20.0, step=0.5)
        width = st.number_input("Building Width (b) [m]", min_value=1.0, value=10.0, step=0.5)
        height = st.number_input("Building Height (h) [m]", min_value=1.0, value=15.0, step=0.5)
        orientation = st.slider("Building Orientation (° from North)", min_value=0, max_value=359, value=0, step=1)
    
    with col2:
        inset = st.checkbox("Add Inset Zone")
        
        inset_height = 0.0
        inset_distance = 0.0
        
        if inset:
            inset_height = st.number_input("Inset Height [m]", min_value=0.1, max_value=height-0.1, value=min(5.0, height/3), step=0.5)
            inset_distance = st.number_input("Inset Distance from Edges [m]", min_value=0.1, max_value=min(length/2-0.1, width/2-0.1), value=2.0, step=0.5)
    
    # Calculate e parameter
    e = min(width, 2*height)
    st.sidebar.header("Wind Load Parameters")
    st.sidebar.write(f"Parameter e = min(b, 2h) = {e:.2f} m")
    
    # Display building 3D model
    fig = plot_building_3d(length, width, height, orientation, inset, inset_height, inset_distance)
    st.plotly_chart(fig, use_container_width=True)
    
    # Future implementation notes
    st.header("Next Steps")
    st.write("""
    In future updates, this app will:
    1. Show wind zones (A-E) on each elevation with different shades of blue
    2. Calculate wind pressures according to BS EN 1991-1-4
    3. Allow input of terrain category and other wind parameters
    4. Generate summary reports of wind loads
    """)

if __name__ == "__main__":
    main()
