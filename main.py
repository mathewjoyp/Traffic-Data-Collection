import cv2
import numpy as np
import osmnx as ox
import psycopg2 # CHANGED: Switched to Postgres for Neon
import time
import geopy.distance
import os
import zipfile
from datetime import datetime
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path

# --- CONFIGURATION ---

# 1. HARDCODE YOUR COORDINATES HERE (GitHub cannot accept input())
# Replace these numbers with the specific area you want to monitor
USER_BOUNDING_BOX = (10.8122, 10.8000, 76.6500, 76.6400) # Example: (North, South, East, West)

# 2. FILE SETTINGS
SCREENSHOT_FILENAME = "gmaps_screenshot.png"
OUTPUT_IMAGE_FILENAME = "traffic_only.png"
LOCAL_MAP_ARCHIVE = "map.osm.zip"  # The file you uploaded
EXTRACTED_MAP_NAME = "map.osm"     # The file inside the zip

# --- HELPER FUNCTIONS ---

def get_db_connection():
    """Establishes and returns a Postgres database connection."""
    try:
        # CHANGED: Get connection string from Environment Variable
        db_url = os.getenv('NEON_DB_URL')
        if not db_url:
            print("Error: NEON_DB_URL environment variable is missing.")
            return None
        
        conn = psycopg2.connect(db_url)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to Postgres: {e}")
        return None

def clear_traffic_data():
    """Clears all data from the traffic_data table."""
    # OPTIONAL: You might not want to clear data every 15 mins if you are building a history.
    # If you want to keep history, comment out the call to this function in the main block.
    print("Clearing old data from 'traffic_data' table...")
    conn = None
    try:
        conn = get_db_connection()
        if conn is None: return
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE traffic_data")
        conn.commit()
        print("Table 'traffic_data' cleared successfully.")
    except psycopg2.Error as e:
        print(f"Error while clearing table: {e}")
    finally:
        if conn: conn.close()

def is_bbox_contained(inner_bbox, outer_bbox):
    i_north, i_south, i_east, i_west = inner_bbox
    o_north, o_south, o_east, o_west = outer_bbox
    return (i_north <= o_north and 
            i_south >= o_south and 
            i_east <= o_east and 
            i_west >= o_west)

def map_pixels_to_geo(pixel_coords, img_shape, bbox):
    img_height, img_width, _ = img_shape
    north, south, east, west = bbox
    
    geo_coords = []
    for x, y in pixel_coords:
        lon = west + (x / img_width) * (east - west)
        lat = north - (y / img_height) * (north - south)
        geo_coords.append((lat, lon))
        
    return geo_coords

# --- MAP LOADING FUNCTION ---
def get_road_network_from_osm_file(osm_filepath, user_bbox):
    print(f"Loading road network graph from file: {osm_filepath}...")
    try:
        graph = ox.graph_from_xml(osm_filepath)
        
        print("Map file loaded. Checking boundaries...")
        nodes = graph.nodes()
        map_north = max(data['y'] for n, data in nodes.items())
        map_south = min(data['y'] for n, data in nodes.items())
        map_east = max(data['x'] for n, data in nodes.items())
        map_west = min(data['x'] for n, data in nodes.items())
        map_bounds = (map_north, map_south, map_east, map_west)
        
        if is_bbox_contained(user_bbox, map_bounds):
            print("User's bounding box is valid. Cropping graph...")
            graph = ox.truncate.truncate_graph_bbox(graph, bbox=user_bbox, truncate_by_edge=True)
            print("Projecting graph...")
            graph_proj = ox.project_graph(graph)
            return graph, graph_proj
        else:
            print(f"ERROR: Bounding Box is outside the map file.")
            print(f"User: {user_bbox}")
            print(f"Map: {map_bounds}")
            return None, None

    except Exception as e:
        print(f"Error loading map file: {e}")
        return None, None

# --- CORE LOGIC ---

def take_google_maps_screenshot(bbox, filename):
    print("Taking screenshot from Google Maps...")
    north, south, east, west = bbox
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    zoom = np.interp(max(north - south, east - west), [0.01, 0.1, 1], [15, 12, 8])

    url = f"https://www.google.com/maps/@{center_lat},{center_lon},{zoom:.2f}z/data=!5m1!1e1"

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new") # UPDATED for modern Chrome
    options.add_argument("--window-size=1920,1080")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        print("Waiting 20 seconds for map and traffic layer to load...")
        time.sleep(20) 
        driver.save_screenshot(filename)
        driver.quit()
        print(f"Screenshot saved as {filename}")
        return True
    except Exception as e:
        print(f"An error occurred while taking the screenshot: {e}")
        return False

def process_image_for_traffic(filename, output_filename):
    print("Processing image...")
    img = cv2.imread(filename)
    if img is None:
        print("Error: Could not read the screenshot file.")
        return None, None
    img = img[100:-50, 50:-50]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.add(cv2.inRange(hsv_img, lower_red1, upper_red1),
                       cv2.inRange(hsv_img, lower_red2, upper_red2))
    
    # Remove icons
    circles = cv2.HoughCircles(red_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=15, minRadius=5, maxRadius=30)
    noise_mask = np.zeros_like(red_mask)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(noise_mask, (i[0], i[1]), i[2] + 2, 255, -1)

    final_mask = cv2.subtract(red_mask, noise_mask)
    result = np.zeros_like(img)
    result[final_mask > 0] = (0, 0, 255)
    cv2.imwrite(output_filename, result)
    return result, img.shape

def analyze_and_store_traffic_data(processed_img, img_shape, road_graph, road_graph_proj, bbox, timestamp):
    print("Analyzing traffic contours...")
    gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No heavy traffic detected.")
        return

    conn = get_db_connection()
    if not conn: return
        
    cursor = conn.cursor()
    processed_edges = set()
    segments_stored = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        geo_center = map_pixels_to_geo([(cX, cY)], img_shape, bbox)[0]
        center_lat, center_lon = geo_center

        nearest_edge_u, nearest_edge_v, _ = ox.distance.nearest_edges(road_graph, X=[center_lon], Y=[center_lat], return_dist=False)[0]
        edge_id = (nearest_edge_u, nearest_edge_v)
        if edge_id in processed_edges: continue
        processed_edges.add(edge_id)

        try:
            edge_data = road_graph_proj.get_edge_data(nearest_edge_u, nearest_edge_v)[0]
            start_node = road_graph.nodes[nearest_edge_u]
            end_node = road_graph.nodes[nearest_edge_v]

            street_name = edge_data.get('name', 'Unnamed')
            if isinstance(street_name, list): street_name = street_name[0]
            
            segment_length = edge_data.get('length', 0)
            
            # CHANGED: Use %s for Postgres placeholders
            sql = """INSERT INTO traffic_data 
                     (street_name, segment_length_meters, start_lat, start_lon, end_lat, end_lon, capture_timestamp) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (street_name, segment_length, start_node['y'], start_node['x'], end_node['y'], end_node['x'], timestamp))
            segments_stored += 1
            
        except Exception as e:
            print(f"Could not process edge {edge_id}: {e}")

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Stored {segments_stored} new records.")

# --- SCRIPT EXECUTION ---

if __name__ == "__main__":
    print("--- 🚦 Starting Traffic Monitor Job ---")
    
    SCRIPT_DIR = Path(__file__).parent
    ZIP_FILE_PATH = SCRIPT_DIR / LOCAL_MAP_ARCHIVE
    OSM_FILE_PATH = SCRIPT_DIR / EXTRACTED_MAP_NAME
    
    # 1. Check if we need to unzip
    if not OSM_FILE_PATH.exists():
        if ZIP_FILE_PATH.is_file():
            print(f"Unzipping {LOCAL_MAP_ARCHIVE}...")
            with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(SCRIPT_DIR)
            print("Unzip complete.")
        else:
            print(f"Error: Neither {EXTRACTED_MAP_NAME} nor {LOCAL_MAP_ARCHIVE} found.")
            exit(1)

    # 2. Load Map (Now point to the extracted OSM_FILE_PATH)
    road_graph, road_graph_proj = get_road_network_from_osm_file(OSM_FILE_PATH, USER_BOUNDING_BOX)
    
    if road_graph and road_graph_proj:
        # Run ONCE (GitHub Actions handles the schedule)
        print("Road network loaded. Starting scan...")
        
        # NOTE: If you want to keep history, COMMENT OUT this line:
        #clear_traffic_data() 
        
        if take_google_maps_screenshot(USER_BOUNDING_BOX, SCREENSHOT_FILENAME):
            processed_img, img_shape = process_image_for_traffic(SCREENSHOT_FILENAME, OUTPUT_IMAGE_FILENAME)
            if processed_img is not None:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                analyze_and_store_traffic_data(processed_img, img_shape, road_graph, road_graph_proj, USER_BOUNDING_BOX, timestamp)
        
        print("--- Job Complete ---")
    else:
        print("Job Failed: Map could not be loaded.")
        exit(1)
