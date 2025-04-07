import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Dict

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

def calculate_wind_zones(length, width, height, wind_direction):
    """
    Calculate wind zones for each elevation based on BS EN 1991-1-4
    Wind direction: 0=North, 90=East, 180=South, 270=West
    """
    e = min(width, 2 * height)  # Calculate e parameter
    
    # For simplicity, we'll assume wind coming from one of the four cardinal directions
    # We need to determine which dimension is along the wind and which is across
    if wind_direction in [0, 180]:  # Wind from North or South
        d = length  # length is along wind
        b = width   # width is crosswind
    else:  # Wind from East or West
        d = width   # width is along wind
        b = length  # length is crosswind
    
    # Zones dictionary to store the zones for each face
    zones = {}
    
    # Windward face (face the wind hits directly)
    if wind_direction == 0:  # North
        windward_face = 3
    elif wind_direction == 90:  # East
        windward_face = 2
    elif wind_direction == 180:  # South
        windward_face = 1
    else:  # West
        windward_face = 4
    
    # Determine zones for windward face
    if e < d:
        # Case where e < d (divide into A, B, C)
        zones[windward_face] = {
            'A': {'x1': 0, 'x2': e/5, 'y1': 0, 'y2': height},
            'B': {'x1': e/5, 'x2': e, 'y1': 0, 'y2': height},
            'C': {'x1': e, 'x2': d, 'y1': 0, 'y2': height}
        }
    elif e >= d and e < 5*d:
        # Case where e ≥ d (divide into A, B)
        zones[windward_face] = {
            'A': {'x1': 0, 'x2': e/5, 'y1': 0, 'y2': height},
            'B': {'x1': e/5, 'x2': d, 'y1': 0, 'y2': height}
        }
    else:
        # Case where e ≥ 5d (only zone A)
        zones[windward_face] = {
            'A': {'x1': 0, 'x2': d, 'y1': 0, 'y2': height}
        }
    
    # Leeward face (opposite to windward)
    leeward_face = (windward_face + 2) % 4
    if leeward_face == 0:
        leeward_face = 4
        
    # Leeward face is always zone D
    zones[leeward_face] = {
        'D': {'x1': 0, 'x2': d, 'y1': 0, 'y2': height}
    }
    
    # Side faces (perpendicular to wind direction)
    side_faces = []
    for i in range(1, 5):
        if i != windward_face and i != leeward_face:
            side_faces.append(i)
    
    # Side faces are zone E
    for face in side_faces:
        zones[face] = {
            'E': {'x1': 0, 'x2': b, 'y1': 0, 'y2': height}
        }
    
    return zones, e, windward_face

def create_building_with_zones(length, width, height, orientation, wind_direction, inset=False, inset_height=0, inset_distance=0):
    """Create 3D plot of building with wind zones"""
    fig = go.Figure()
    
    # Calculate wind zones
    zones, e_param, windward_face = calculate_wind_zones(length, width, height, wind_direction)
    
    # Define zone colors (different shades of blue)
    zone_colors = {
        'A': 'rgb(0, 0, 139)',      # Dark blue
        'B': 'rgb(0, 91, 187)',      # Medium blue
        'C': 'rgb(30, 144, 255)',    # Dodger blue
        'D': 'rgb(135, 206, 235)',   # Sky blue
        'E': 'rgb(176, 224, 230)'    # Powder blue
    }
    
    # Basic building dimensions
    half_length = length / 2
    half_width = width / 2
    
    # Create vertices for each face separately to apply different colors to each zone
    
    # Ground face - just for reference
    x_ground = [-half_length, half_length, half_length, -half_length]
    y_ground = [-half_width, -half_width, half_width, half_width]
    z_ground = [0, 0, 0, 0]
    
    fig.add_trace(go.Mesh3d(
        x=x_ground, y=y_ground, z=z_ground,
        i=[0], j=[1], k=[2],
        color='lightgray',
        opacity=0.7,
        name='Ground'
    ))
    
    # Determine vertices and faces for each elevation based on wind direction
    # This gets complex because we need to create separate mesh for each zone
    
    # For simplicity, let's focus on implementing the basic rectangular building first
    # without inset, and with zones on each face
    
    # Elevation 1 (South)
    if 1 in zones:
        for zone, boundaries in zones[1].items():
            # Calculate vertices for this zone
            x_zone = [
                -half_length + boundaries['x1'],
                -half_length + boundaries['x2'],
                -half_length + boundaries['x2'],
                -half_length + boundaries['x1']
            ]
            y_zone = [-half_width, -half_width, -half_width, -half_width]
            z_zone = [
                boundaries['y1'],
                boundaries['y1'],
                boundaries['y2'],
                boundaries['y2']
            ]
            
            # Apply rotation
            rotated_points = rotate_points(list(zip(x_zone, y_zone)), orientation)
            x_rot, y_rot = zip(*rotated_points)
            
            # Add zone mesh
            fig.add_trace(go.Mesh3d(
                x=x_rot, y=y_rot, z=z_zone,
                i=[0], j=[1], k=[2],
                color=zone_colors[zone],
                opacity=0.8,
                name=f'Elevation 1 - Zone {zone}'
            ))
            fig.add_trace(go.Mesh3d(
                x=x_rot, y=y_rot, z=z_zone,
                i=[0], j=[2], k=[3],
                color=zone_colors[zone],
                opacity=0.8,
                showlegend=False
            ))
    
    # Elevation 2 (East)
    if 2 in zones:
        for zone, boundaries in zones[2].items():
            # Calculate vertices for this zone
            x_zone = [
                half_length, 
                half_length, 
                half_length, 
                half_length
            ]
            y_zone = [
                -half_width + boundaries['x1'],
                -half_width + boundaries['x2'],
                -half_width + boundaries['x2'],
                -half_width + boundaries['x1']
            ]
            z_zone = [
                boundaries['y1'],
                boundaries['y1'],
                boundaries['y2'],
                boundaries['y2']
            ]
            
            # Apply rotation
            rotated_points = rotate_points(list(zip(x_zone, y_zone)), orientation)
            x_rot, y_rot = zip(*rotated_points)
            
            # Add zone mesh
            fig.add_trace(go.Mesh3d(
                x=x_rot, y=y_rot, z=z_zone,
                i=[0], j=[1], k=[2],
                color=zone_colors[zone],
                opacity=0.8,
                name=f'Elevation 2 - Zone {zone}'
            ))
            fig.add_trace(go.Mesh3d(
                x=x_rot, y=y_rot, z=z_zone,
                i=[0], j=[2], k=[3],
                color=zone_colors[zone],
                opacity=0.8,
                showlegend=False
            ))
    
    # Elevation 3 (North)
    if 3 in zones:
        for zone, boundaries in zones[3].items():
            # Calculate vertices for this zone
            x_zone = [
                -half_length + boundaries['x1'],
                -half_length + boundaries['x2'],
                -half_length + boundaries['x2'],
                -half_length + boundaries['x1']
            ]
            y_zone = [half_width, half_width, half_width, half_width]
            z_zone = [
                boundaries['y1'],
                boundaries['y1'],
                boundaries['y2'],
                boundaries['y2']
            ]
            
            # Apply rotation
            rotated_points = rotate_points(list(zip(x_zone, y_zone)), orientation)
            x_rot, y_rot = zip(*rotated_points)
            
            # Add zone mesh
            fig.add_trace(go.Mesh3d(
                x=x_rot, y=y_rot, z=z_zone,
                i=[0], j=[1], k=[2],
                color=zone_colors[zone],
                opacity=0.8,
                name=f'Elevation 3 - Zone {zone}'
            ))
            fig.add_trace(go.Mesh3d(
                x=x_rot, y=y_rot, z=z_zone,
                i=[0], j=[2], k=[3],
                color=zone_colors[zone],
                opacity=0.8,
                showlegend=False
            ))
    
    # Elevation 4 (West)
    if 4 in zones:
        for zone, boundaries in zones[4].items():
            # Calculate vertices for this zone
            x_zone = [
                -half_length, 
                -half_length, 
                -half_length, 
                -half_length
            ]
            y_zone = [
                -half_width + boundaries['x1'],
                -half_width + boundaries['x2'],
                -half_width + boundaries['x2'],
                -half_width + boundaries['x1']
            ]
            z_zone = [
                boundaries['y1'],
                boundaries['y1'],
                boundaries['y2'],
                boundaries['y2']
            ]
            
            # Apply rotation
            rotated_points = rotate_points(list(zip(x_zone, y_zone)), orientation)
            x_rot, y_rot = zip(*rotated_points)
            
            # Add zone mesh
            fig.add_trace(go.Mesh3d(
                x=x_rot, y=y_rot, z=z_zone,
                i=[0], j=[1], k=[2],
                color=zone_colors[zone],
                opacity=0.8,
                name=f'Elevation 4 - Zone {zone}'
            ))
            fig.add_trace(go.Mesh3d(
                x=x_rot, y=y_rot, z=z_zone,
                i=[0], j=[2], k=[3],
                color=zone_colors[zone],
                opacity=0.8,
                showlegend=False
            ))
    
    # Roof - assume flat roof for now
    x_roof = [-half_length, half_length, half_length, -half_length]
    y_roof = [-half_width, -half_width, half_width, half_width]
    z_roof = [height, height, height, height]
    
    # Apply rotation
    rotated_points = rotate_points(list(zip(x_roof, y_roof)), orientation)
    x_rot, y_rot = zip(*rotated_points)
    
    fig.add_trace(go.Mesh3d(
        x=x_rot, y=y_rot, z=z_roof,
        i=[0], j=[1], k=[2],
        color='lightblue',
        opacity=0.7,
        name='Roof'
    ))
    fig.add_trace(go.Mesh3d(
        x=x_rot, y=y_rot, z=z_roof,
        i=[0], j=[2], k=[3],
        color='lightblue',
        opacity=0.7,
        showlegend=False
    ))
    
    # Add wind direction arrow
    arrow_length = max(length, width) * 0.8
    
    # Convert wind_direction to radians and calculate arrow coordinates
    wind_rad = np.radians(wind_direction)
    end_x = -np.sin(wind_rad) * arrow_length
    end_y = -np.cos(wind_rad) * arrow_length
    
    # Apply building orientation
    rotated_arrow = rotate_points([(0, 0), (end_x, end_y)], orientation)
    
    fig.add_trace(go.Scatter3d(
        x=[rotated_arrow[0][0], rotated_arrow[1][0]],
        y=[rotated_arrow[0][1], rotated_arrow[1][1]],
        z=[height/2, height/2],  # Place at mid-height
        mode='lines+markers',
        line=dict(color='red', width=5),
        marker=dict(size=[0, 8]),
        name='Wind Direction'
    ))
    
    # Add North arrow
    north_arrow_length = max(length, width) * 0.5
    north_x = [0, 0, 0, 0.05*north_arrow_length, 0, -0.05*north_arrow_length]
    north_y = [0, north_arrow_length, 0.8*north_arrow_length, north_arrow_length, 0.8*north_arrow_length, north_arrow_length]
    north_z = [0, 0, 0, 0, 0, 0]
    
    # Apply rotation to North arrow
    north_points_rot = rotate_points(list(zip(north_x, north_y)), orientation)
    north_x_rot, north_y_rot = zip(*north_points_rot)
    
    fig.add_trace(go.Scatter3d(
        x=north_x_rot, y=north_y_rot, z=north_z,
        mode='lines',
        line=dict(color='blue', width=5),
        name='North'
    ))
    
    # Add "N" label for North
    n_label_x, n_label_y = rotate_points([(0, 1.1*north_arrow_length)], orientation)[0]
    fig.add_trace(go.Scatter3d(
        x=[n_label_x], y=[n_label_y], z=[0],
        mode='text',
        text=['N'],
        textposition='middle center',
        textfont=dict(size=14, color='blue'),
        name='North Label'
    ))
    
    # Add elevation labels
    half_length = length / 2
    half_width = width / 2
    mid_height = height / 2
    
    # Calculate positions for elevation labels
    elevation_points = [
        [0, -half_width - arrow_length*0.1, mid_height],  # Elevation 1 (South)
        [half_length + arrow_length*0.1, 0, mid_height],  # Elevation 2 (East)
        [0, half_width + arrow_length*0.1, mid_height],   # Elevation 3 (North)
        [-half_length - arrow_length*0.1, 0, mid_height]  # Elevation 4 (West)
    ]
    
    # Apply rotation to elevation label positions
    for i, point in enumerate(elevation_points):
        rotated = rotate_points([(point[0], point[1])], orientation)[0]
        elevation_points[i][0] = rotated[0]
        elevation_points[i][1] = rotated[1]
    
    # Add elevation labels
    for i, point in enumerate(elevation_points):
        # Add zone information to the label if available
        if i+1 in zones:
            zone_str = ", ".join([f"Zone {z}" for z in zones[i+1].keys()])
            label = f'Elevation {i+1} ({zone_str})'
        else:
            label = f'Elevation {i+1}'
            
        fig.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]],
            mode='text',
            text=[label],
            textposition='middle center',
            textfont=dict(size=12, color='black'),
            name=f'Elevation {i+1}'
        ))
    
    # Update layout
    fig.update_layout(
        title='Building 3D Model with Wind Zones',
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
    
    return fig, e_param, windward_face

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
        wind_direction = st.selectbox(
            "Wind Direction",
            options=[
                ("North (0°)", 0),
                ("East (90°)", 90),
                ("South (180°)", 180),
                ("West (270°)", 270)
            ],
            format_func=lambda x: x[0],
            index=0
        )[1]
        
        inset = st.checkbox("Add Inset Zone")
        
        inset_height = 0.0
        inset_distance = 0.0
        
        if inset:
            inset_height = st.number_input("Inset Height [m]", min_value=0.1, max_value=height-0.1, value=min(5.0, height/3), step=0.5)
            inset_distance = st.number_input("Inset Distance from Edges [m]", min_value=0.1, max_value=min(length/2-0.1, width/2-0.1), value=2.0, step=0.5)
    
    # Display building 3D model with wind zones
    fig, e_param, windward_face = create_building_with_zones(
        length, width, height, orientation, wind_direction, inset, inset_height, inset_distance
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Wind parameter information
    st.sidebar.header("Wind Load Parameters")
    st.sidebar.write(f"Parameter e = min(b, 2h) = {e_param:.2f} m")
    st.sidebar.write(f"Windward face: Elevation {windward_face}")
    st.sidebar.write(f"Leeward face: Elevation {(windward_face + 2) % 4 or 4}")
    
    # Zone color legend
    st.sidebar.header("Zone Color Legend")
    zone_colors = {
        'A': 'rgb(0, 0, 139)',      # Dark blue
        'B': 'rgb(0, 91, 187)',      # Medium blue
        'C': 'rgb(30, 144, 255)',    # Dodger blue
        'D': 'rgb(135, 206, 235)',   # Sky blue
        'E': 'rgb(176, 224, 230)'    # Powder blue
    }
    
    for zone, color in zone_colors.items():
        st.sidebar.markdown(
            f"<div style='display: flex; align-items: center;'>"
            f"<div style='width: 20px; height: 20px; background-color: {color}; margin-right: 10px;'></div>"
            f"<div>Zone {zone}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # Explanation of zones
    st.header("Wind Zone Explanation")
    st.write("""
    According to BS EN 1991-1-4, building faces are divided into different zones for wind load calculation:
    
    - **Zone A**: Areas with highest wind pressure (typically corners and edges)
    - **Zone B**: Areas with moderate wind pressure
    - **Zone C**: Areas with lower wind pressure (typically center areas)
    - **Zone D**: Leeward face (opposite to windward face) - negative pressure (suction)
    - **Zone E**: Side faces perpendicular to wind direction - negative pressure (suction)
    
    The parameter **e = min(b, 2h)** determines how these zones are distributed on the building faces.
    """)
    
    # Future implementation notes
    st.header("Next Steps")
    st.write("""
    In future updates, this app will:
    1. Handle more complex building geometries including inset zones
    2. Calculate actual wind pressures according to BS EN 1991-1-4
    3. Allow input of terrain category and other wind parameters
    4. Generate summary reports of wind loads
    """)

if __name__ == "__main__":
    main()
