#!/usr/bin/env python3
"""
patrol_and_detect.py

A Python script using MAVSDK and YOLO to patrol a geofence, detect a
soccer ball (sports ball class in COCO dataset), calculate its GPS
coordinates from the image frame, navigate to it, and land nearby.
Supports both real ArduPilot SITL flight and a simulated mock mode.
"""

import asyncio
import math
import sys
import argparse
import time
from mavsdk import System
from mavsdk.action import ActionError

# Optional imports with graceful degradation
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def get_distance_meters(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two GPS points using the Haversine formula.
    """
    R = 6378137.0  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def is_inside_polygon(lat, lon, polygon):
    """
    Checks if a point (lat, lon) is inside a polygon using the Ray-Casting Algorithm.
    polygon: List of (latitude, longitude) tuples.
    """
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if lon > min(p1y, p2y):
            if lon <= max(p1y, p2y):
                if lat <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (lon - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or lat <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
        
    return inside


class BallPatroller:
    def __init__(self, args):
        self.address = args.address
        self.video_source = args.video_source
        self.patrol_altitude = args.altitude
        self.mock_mode = args.mock
        self.show_video = not args.no_video
        self.video_start_frame = getattr(args, "video_start_frame", 0)
        
        self.drone = None
        self.running = True
        self.state = "INIT"  # States: INIT, TAKEOFF, PATROLLING, BALL_DETECTED, NAVIGATING_TO_BALL, LANDING, DONE
        
        # Telemetry Cache
        self.current_lat = None
        self.current_lon = None
        self.current_alt = None  # Relative altitude
        self.current_heading = None
        self.home_abs_alt = None
        self.home_lat = None
        self.home_lon = None
        
        # Geofence & Waypoints (Geocentric Coordinates, generated relative to Home)
        self.geofence = []
        self.patrol_waypoints = []
        
    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        mode_prefix = "[MOCK]" if self.mock_mode else "[REAL]"
        print(f"{timestamp} {mode_prefix} [{self.state}] {message}")

    def generate_relative_points(self, home_lat, home_lon):
        """
        Generates a bounding geofence and patrol path relative to the home position.
        Geofence is a square with ~110m sides.
        Patrol path is an inner square with ~44m sides.
        """
        # Coordinate offsets: 0.0005 degrees is approx 55 meters North/South and East/West
        self.geofence = [
            (home_lat - 0.0005, home_lon - 0.0005),
            (home_lat + 0.0005, home_lon - 0.0005),
            (home_lat + 0.0005, home_lon + 0.0005),
            (home_lat - 0.0005, home_lon + 0.0005)
        ]
        
        # 0.0002 degrees is approx 22 meters North/South and East/West
        self.patrol_waypoints = [
            (home_lat - 0.0002, home_lon - 0.0002),
            (home_lat + 0.0002, home_lon - 0.0002),
            (home_lat + 0.0002, home_lon + 0.0002),
            (home_lat - 0.0002, home_lon + 0.0002)
        ]
        self.log(f"Generated relative Geofence boundaries centered around home ({home_lat:.6f}, {home_lon:.6f})")
        self.log(f"Geofence vertices: {self.geofence}")
        self.log(f"Patrol waypoints: {self.patrol_waypoints}")

    def calculate_ball_gps(self, drone_lat, drone_lon, drone_alt, drone_heading, x_center, y_center, img_w, img_h):
        """
        Calculates the GPS coordinates of the soccer ball assuming a downward-pointing camera (nadir).
        """
        # Standard camera specs: 80° HFOV and 60° VFOV
        hfov = math.radians(80.0)
        vfov = math.radians(60.0)
        
        # Calculate ground footprint dimensions at the drone's altitude
        w_ground = 2.0 * drone_alt * math.tan(hfov / 2.0)
        h_ground = 2.0 * drone_alt * math.tan(vfov / 2.0)
        
        # Scale: meters per pixel
        scale_x = w_ground / img_w
        scale_y = h_ground / img_h
        
        # Pixel offsets from the image center
        # Center is (img_w/2, img_h/2). Image X increases right, Y increases down.
        # Body frame: forward (up in image) is positive body X, right (right in image) is positive body Y.
        dx_pixel = x_center - (img_w / 2.0)
        dy_pixel = (img_h / 2.0) - y_center
        
        # Physical offset in body frame
        d_right = dx_pixel * scale_x
        d_forward = dy_pixel * scale_y
        
        # Rotate by drone heading (yaw) to align with North-East frame
        # yaw (psi) is clockwise from North
        psi = math.radians(drone_heading)
        d_north = d_forward * math.cos(psi) - d_right * math.sin(psi)
        d_east = d_forward * math.sin(psi) + d_right * math.cos(psi)
        
        # Convert offsets in meters to changes in Latitude and Longitude
        R = 6378137.0  # Earth's radius in meters
        d_lat = (d_north / R) * (180.0 / math.pi)
        d_lon = (d_east / (R * math.cos(math.radians(drone_lat)))) * (180.0 / math.pi)
        
        ball_lat = drone_lat + d_lat
        ball_lon = drone_lon + d_lon
        
        self.log(f"Offset calculation: d_forward={d_forward:.2f}m, d_right={d_right:.2f}m -> d_north={d_north:.2f}m, d_east={d_east:.2f}m")
        return ball_lat, ball_lon

    # ==========================================
    # MOCK MODE IMPLEMENTATION (No hardware/SITL)
    # ==========================================
    async def run_mock(self):
        self.log("Initializing MOCK simulation...")
        self.state = "INIT"
        
        # Set mock home coordinates (San Francisco Presidio)
        self.home_lat = 37.8016
        self.home_lon = -122.4648
        self.home_abs_alt = 10.0  # AMSL
        
        self.current_lat = self.home_lat
        self.current_lon = self.home_lon
        self.current_alt = 0.0
        self.current_heading = 0.0
        
        self.generate_relative_points(self.home_lat, self.home_lon)
        
        self.log("Mock Drone status: Ready. Arming...")
        self.state = "TAKEOFF"
        await asyncio.sleep(1.0)
        
        self.log(f"Mock Drone: Taking off to {self.patrol_altitude}m...")
        # Simulate gradual climb
        while self.current_alt < self.patrol_altitude:
            self.current_alt += 2.0
            self.log(f"Mock Altitude: {self.current_alt:.1f}m")
            await asyncio.sleep(0.5)
            
        self.log("Mock Drone: Takeoff complete. Starting patrol.")
        self.state = "PATROLLING"
        
        # Start a concurrent task to mock soccer ball detection
        detection_task = asyncio.create_task(self.mock_detection_timer())
        
        wp_index = 0
        while self.state == "PATROLLING" and self.running:
            target_wp = self.patrol_waypoints[wp_index]
            self.log(f"Mock Drone flying towards Waypoint {wp_index + 1}: {target_wp}")
            
            # Simulate travel to waypoint
            steps = 5
            for step in range(steps):
                if self.state != "PATROLLING":
                    break
                # Interpolate coordinate
                fraction = (step + 1) / steps
                self.current_lat = self.current_lat + (target_wp[0] - self.current_lat) * fraction
                self.current_lon = self.current_lon + (target_wp[1] - self.current_lon) * fraction
                # Simulate a rotating heading
                self.current_heading = (self.current_heading + 15) % 360
                
                dist = get_distance_meters(self.current_lat, self.current_lon, target_wp[0], target_wp[1])
                self.log(f"Mock Drone position: {self.current_lat:.6f}, {self.current_lon:.6f} | Distance: {dist:.1f}m")
                await asyncio.sleep(1.0)
                
            if self.state == "PATROLLING":
                self.log(f"Mock Drone reached Waypoint {wp_index + 1}")
                wp_index = (wp_index + 1) % len(self.patrol_waypoints)
                
        # Wait for the mock process to conclude
        await detection_task
        self.state = "DONE"
        self.log("Mock mission completed successfully.")

    async def mock_detection_timer(self):
        # Wait 8 seconds before triggering a detection event
        await asyncio.sleep(8.0)
        if self.state == "PATROLLING":
            self.log("Simulating soccer ball detection frame from camera...")
            # Imagine we detect a soccer ball in a 640x480 frame at x=380, y=200 (offset from center)
            # This simulates a real YOLO box detection callback
            await self.on_ball_detected(x_center=380.0, y_center=200.0, img_w=640, img_h=480, conf=0.92)

    # ==========================================
    # REAL MODE IMPLEMENTATION (MAVSDK & YOLO)
    # ==========================================
    async def run_real(self):
        if not OPENCV_AVAILABLE:
            self.log("[ERROR] OpenCV library is missing. Please run pip install opencv-python")
            return
            
        self.drone = System()
        self.log(f"Connecting to MAVSDK drone system at: {self.address}")
        await self.drone.connect(system_address=self.address)
        
        # Start telemetry listeners
        asyncio.create_task(self._listen_position())
        asyncio.create_task(self._listen_heading())
        asyncio.create_task(self._listen_home())
        
        # Wait for telemetry connection
        self.log("Waiting for GPS lock and home position...")
        while self.current_lat is None or self.current_lon is None or self.home_abs_alt is None:
            await asyncio.sleep(0.5)
            
        self.home_lat = self.current_lat
        self.home_lon = self.current_lon
        self.generate_relative_points(self.home_lat, self.home_lon)
        
        # Start processing camera feed via YOLO
        video_task = asyncio.create_task(self.process_video_feed())
        
        # Flight Plan Start
        try:
            self.log("Arming vehicle...")
            await self.drone.action.arm()
            
            self.log(f"Taking off to {self.patrol_altitude}m...")
            await self.drone.action.takeoff()
            
            # Wait for takeoff completion (climb above 1.5 meters)
            self.state = "TAKEOFF"
            while self.current_alt < 1.5:
                self.log(f"Altitude climbing: {self.current_alt:.1f}m / 1.5m")
                await asyncio.sleep(1.0)
                
            self.log("Takeoff complete! Starting patrol loop...")
            self.state = "PATROLLING"
            
            wp_index = 0
            while self.state == "PATROLLING" and self.running:
                target_wp = self.patrol_waypoints[wp_index]
                self.log(f"Flying to Waypoint {wp_index + 1}/{len(self.patrol_waypoints)}: {target_wp}")
                
                # Navigate to waypoint
                # goto_location parameters: latitude, longitude, absolute altitude AMSL, yaw
                target_abs_alt = self.home_abs_alt + self.patrol_altitude
                await self.drone.action.goto_location(target_wp[0], target_wp[1], target_abs_alt, 0.0)
                
                # Check for waypoint completion or state changes
                loop_cnt = 0
                while self.state == "PATROLLING" and self.running:
                    dist = get_distance_meters(self.current_lat, self.current_lon, target_wp[0], target_wp[1])
                    loop_cnt += 1
                    if loop_cnt % 4 == 0:
                        self.log(f"Drone at ({self.current_lat:.6f}, {self.current_lon:.6f}), Alt: {self.current_alt:.1f}m. Dist to WP {wp_index + 1}: {dist:.1f}m")
                    if dist < 1.5:  # Arrived within 1.5m radius
                        self.log(f"Arrived at Waypoint {wp_index + 1}")
                        wp_index = (wp_index + 1) % len(self.patrol_waypoints)
                        break
                    await asyncio.sleep(0.5)
            # If patrol loop exited due to ball detection, wait for navigation and landing to complete
            while self.state in ["BALL_DETECTED", "NAVIGATING_TO_BALL", "LANDING"] and self.running:
                await asyncio.sleep(1.0)
                
        except ActionError as e:
            self.log(f"[FLIGHT ACTION ERROR] Command rejected: {e}")
        finally:
            self.running = False
            await video_task
            self.state = "DONE"
            self.log("Mission terminated.")

    async def _listen_position(self):
        async for position in self.drone.telemetry.position():
            self.current_lat = position.latitude_deg
            self.current_lon = position.longitude_deg
            self.current_alt = position.relative_altitude_m

    async def _listen_heading(self):
        async for heading in self.drone.telemetry.heading():
            self.current_heading = heading.heading_deg

    async def _listen_home(self):
        async for home in self.drone.telemetry.home():
            self.home_abs_alt = home.absolute_altitude_m
            break

    async def process_video_feed(self):
        """
        Runs the YOLO inference loop on frames captured from the video source.
        """
        self.log(f"Starting video processing. Source: {self.video_source}")
        
        # Load YOLO model
        if YOLO_AVAILABLE:
            self.log("Loading YOLOv8 model (yolov8s.pt)...")
            model = YOLO("yolov8s.pt")
            self.log("YOLO model loaded.")
        else:
            self.log("[WARNING] YOLO library (ultralytics) is not available. Camera stream will run without inference.")
            model = None
            
        # Wait until the drone transitions to PATROLLING state before opening the video file.
        # This prevents the video from playing through keyframes while the drone is in TAKEOFF state.
        while self.state != "PATROLLING" and self.running:
            await asyncio.sleep(0.1)
            
        self.log("Video processing thread woke up! Opening video capture...")
        try:
            # Parse video source (string for stream/file, or int for local USB camera)
            try:
                src = int(self.video_source)
            except ValueError:
                src = self.video_source
                
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                self.log(f"[WARNING] Could not open video source '{src}'. Visual navigation disabled.")
                return
            self.log(f"Video capture opened successfully. Source: {src}")
                
            import os
            out_writer = None
            if self.video_start_frame > 0 and isinstance(src, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_start_frame)
                self.log(f"Seeking video file to start frame: {self.video_start_frame}")
                
            # Read first frame to initialize VideoWriter with matching dimensions
            ret, first_frame = cap.read()
            if ret:
                img_h, img_w, _ = first_frame.shape
                os.makedirs("test-flight", exist_ok=True)
                output_video_path = "test-flight/detection_run.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (img_w, img_h))
                self.log(f"Recording processed video run to: {output_video_path}")
                # Seek back to start frame so we don't miss the first frame in recording
                if self.video_start_frame > 0 and isinstance(src, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_start_frame)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.log("[WARNING] Could not read first frame of video stream. Video recording disabled.")
                
            frame_cnt = 0
            while self.running:
                ret, frame = cap.read()
                if ret:
                    frame_cnt += 1
                    if frame_cnt % 50 == 0:
                        self.log(f"Video loop: processed {frame_cnt} frames...")
                if not ret:
                    # Loop video if it is a file
                    if isinstance(src, str) and src.endswith(('.mp4', '.avi', '.mkv')):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_start_frame)
                        continue
                    await asyncio.sleep(0.1)
                    continue
                    
                img_h, img_w, _ = frame.shape
                
                # Perform YOLO detection if model is loaded and in active flight states (except INIT or DONE)
                if model and self.state not in ["INIT", "DONE"]:
                    # COCO class index 32: sports ball, class 0: person
                    # Lower confidence threshold to 0.15 to catch small/moving humans with YOLOv8s
                    results = model.predict(frame, classes=[0, 32], conf=0.15, verbose=False)
                    for result in results:
                        boxes = result.boxes
                        
                        # Handle person detections (visual overlays only)
                        person_boxes = [b for b in boxes if int(b.cls[0].item()) == 0]
                        for person_box in person_boxes:
                            xyxy = person_box.xyxy[0].tolist()
                            conf = person_box.conf[0].item()
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                            cv2.putText(frame, f"Person {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                        
                        # Handle ball detections (visual overlays + flight trigger)
                        ball_boxes = [b for b in boxes if int(b.cls[0].item()) == 32]
                        if len(ball_boxes) > 0:
                            # Take the detection with highest confidence
                            best_box = sorted(ball_boxes, key=lambda b: b.conf[0].item(), reverse=True)[0]
                            xyxy = best_box.xyxy[0].tolist()
                            conf = best_box.conf[0].item()
                            
                            x_center = (xyxy[0] + xyxy[2]) / 2.0
                            y_center = (xyxy[1] + xyxy[3]) / 2.0
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                            cv2.putText(frame, f"Soccer Ball {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Trigger callback ONLY for high-confidence sports ball detections in PATROLLING state
                            if conf >= 0.5 and self.state == "PATROLLING":
                                asyncio.create_task(self.on_ball_detected(x_center, y_center, img_w, img_h, conf))
                            
                # Write frame with overlays to output video file
                if out_writer and ret:
                    out_writer.write(frame)
                    
                # Display output if window GUI is enabled
                if self.show_video:
                    cv2.imshow("Drone Downward YOLO View", frame)
                    # Press 'q' inside OpenCV window to exit manually
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                        
                await asyncio.sleep(0.03)  # Loop at ~30 FPS
                
            cap.release()
            if out_writer:
                out_writer.release()
                self.log("Saved processed output video to: test-flight/detection_run.mp4")
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.log(f"[VIDEO PROCESSING ERROR] {e}")

    # ==========================================
    # INTERACTION & STATE CONTROL
    # ==========================================
    async def on_ball_detected(self, x_center, y_center, img_w, img_h, conf):
        """
        Callback triggered when a soccer ball is detected. 
        """
        # Guard state to prevent double execution
        if self.state != "PATROLLING":
            return
            
        self.state = "BALL_DETECTED"
        self.log(f"YOLO MATCH: Soccer ball detected! Confidence: {conf:.2f}")
        
        # Take telemetry snapshot at the exact moment of detection
        drone_lat = self.current_lat
        drone_lon = self.current_lon
        drone_alt = self.current_alt
        drone_heading = self.current_heading if self.current_heading is not None else 0.0
        
        self.log(f"Telemetry Snapshot -> Lat: {drone_lat:.6f}, Lon: {drone_lon:.6f}, Alt: {drone_alt:.2f}m, Yaw: {drone_heading:.1f}°")
        
        # Compute GPS coordinates of target ball
        ball_lat, ball_lon = self.calculate_ball_gps(
            drone_lat, drone_lon, drone_alt, drone_heading,
            x_center, y_center, img_w, img_h
        )
        self.log(f"Ball target calculated at: Latitude {ball_lat:.7f}, Longitude {ball_lon:.7f}")
        
        # Validate that the soccer ball target is inside the geofence
        if is_inside_polygon(ball_lat, ball_lon, self.geofence):
            self.log("Geofence Check: TARGET INSIDE GEOFENCE. Initiating fly-to and land sequence.")
            await self.navigate_and_land(ball_lat, ball_lon)
        else:
            self.log("[GEOFENCE VIOLATION WARNING] Soccer ball target is OUTSIDE the geofence bounds!")
            self.log("For safety, the drone will hold position and land immediately at its current coordinate.")
            await self.hold_and_land()

    async def navigate_and_land(self, target_lat, target_lon):
        self.state = "NAVIGATING_TO_BALL"
        self.log(f"Navigating to ball coordinates: {target_lat:.6f}, {target_lon:.6f}")
        
        if self.mock_mode:
            # Simulate flight to target
            self.current_lat = target_lat
            self.current_lon = target_lon
            await asyncio.sleep(2.0)
            self.log("Mock Drone: Arrived directly above target ball.")
            
            # Simulate Landing
            self.state = "LANDING"
            self.log("Mock Drone: Landing...")
            while self.current_alt > 0.0:
                self.current_alt -= 2.0
                self.log(f"Mock Altitude: {max(0.0, self.current_alt):.1f}m")
                await asyncio.sleep(0.5)
            self.log("Mock Drone: Landed successfully near soccer ball. Disarming.")
            self.state = "DONE"
            self.running = False
        else:
            try:
                target_abs_alt = self.home_abs_alt + self.patrol_altitude
                await self.drone.action.goto_location(target_lat, target_lon, target_abs_alt, 0.0)
                
                # Poll distance to ball
                while self.state == "NAVIGATING_TO_BALL" and self.running:
                    dist = get_distance_meters(self.current_lat, self.current_lon, target_lat, target_lon)
                    self.log(f"Approaching ball. Distance remaining: {dist:.1f} meters")
                    if dist < 1.0:  # Threshold of 1 meter
                        self.log("Target reached. Initiating precision landing.")
                        break
                    await asyncio.sleep(0.5)
                    
                await self.drone.action.land()
                self.state = "LANDING"
                self.log("Landing command transmitted. Waiting for ground contact...")
                
                # Check for landed status
                async for landed_state in self.drone.telemetry.landed_state():
                    if landed_state == landed_state.ON_GROUND:
                        self.log("Drone landed on ground. Disarming.")
                        await self.drone.action.disarm()
                        break
                        
            except ActionError as e:
                self.log(f"[FLIGHT ACTION ERROR during Landing] {e}")
            finally:
                self.state = "DONE"
                self.running = False

    async def hold_and_land(self):
        self.state = "LANDING"
        if self.mock_mode:
            self.log("Mock Drone landing at current position.")
            self.running = False
        else:
            try:
                # Land right where we are
                await self.drone.action.land()
                self.log("Landing command sent at current location due to safety override.")
            except ActionError as e:
                self.log(f"[FLIGHT ACTION ERROR during safety land] {e}")
            finally:
                self.running = False


def main():
    parser = argparse.ArgumentParser(
        description="MAVSDK script for ArduPilot SITL to patrol a geofence and land near a YOLO-detected soccer ball."
    )
    parser.add_argument(
        "--address",
        default="udpin://127.0.0.1:14550",
        help="MAVSDK server address (e.g. udpin://127.0.0.1:14550 or tcp://127.0.0.1:5760)"
    )
    parser.add_argument(
        "--video-source",
        default="0",
        help="OpenCV capture source. Number (e.g., '0' for webcam) or RTSP/file URI"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock simulation mode without connection to a physical/SITL drone or camera"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable GUI output window (useful for headless execution)"
    )
    parser.add_argument(
        "--video-start-frame",
        type=int,
        default=0,
        help="Frame index to start video processing from (default: 0)"
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=10.0,
        help="Patrol altitude in meters (default: 10.0)"
    )
    
    args = parser.parse_args()
    
    patroller = BallPatroller(args)
    
    try:
        if args.mock:
            asyncio.run(patroller.run_mock())
        else:
            asyncio.run(patroller.run_real())
    except KeyboardInterrupt:
        print("\nAborted by user.")
        patroller.running = False


if __name__ == "__main__":
    main()
