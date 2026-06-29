#!/usr/bin/env python3
"""
hawthorne_flight.py

An autonomous and telemetry monitoring flight script using MAVSDK and PyYAML.
Supports:
1. --args=qgroundcontrol: Monitoring mode that prints drone status and telemetry.
2. --args=programmatic --file=<file.yaml>: Autonomous flight mode that parses
   and executes commands from a YAML file.
"""

import asyncio
import math
import argparse
import sys
import time
import yaml
from mavsdk import System
from mavsdk.action import ActionError

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

class HawthorneFlightController:
    def __init__(self, address, mode, file_path):
        self.address = address
        self.mode = mode
        self.file_path = file_path
        self.drone = None
        self.running = True
        
        # Telemetry Cache
        self.current_lat = None
        self.current_lon = None
        self.current_alt = None  # Relative altitude in meters
        self.home_abs_alt = None
        self.home_lat = None
        self.home_lon = None
        self.is_armed = False
        self.flight_mode = None
        self.battery_remaining = None

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} [HawthorneFlight] {message}")

    async def run(self):
        if self.mode == "qgroundcontrol":
            await self.run_qgroundcontrol_mode()
        elif self.mode == "programmatic":
            await self.run_programmatic_mode()

    async def run_qgroundcontrol_mode(self):
        self.drone = System()
        self.log(f"Connecting to MAVSDK drone system in MONITOR mode at: {self.address}")
        await self.drone.connect(system_address=self.address)
        
        # Start telemetry listeners
        asyncio.create_task(self._listen_position())
        asyncio.create_task(self._listen_armed())
        asyncio.create_task(self._listen_flight_mode())
        asyncio.create_task(self._listen_battery())
        
        self.log("Telemetry monitor started. Waiting for telemetry data...")
        
        try:
            while self.running:
                if self.current_lat is not None:
                    armed_str = "ARMED" if self.is_armed else "DISARMED"
                    mode_str = self.flight_mode if self.flight_mode else "UNKNOWN"
                    alt_str = f"{self.current_alt:.1f}m" if self.current_alt is not None else "N/A"
                    gps_str = f"({self.current_lat:.6f}, {self.current_lon:.6f})"
                    bat_str = f"{int(self.battery_remaining * 100)}%" if self.battery_remaining is not None else "N/A"
                    
                    print(f"[Telemetry] Status: {armed_str} | Mode: {mode_str} | Alt: {alt_str} | GPS: {gps_str} | Bat: {bat_str}")
                else:
                    self.log("Waiting for GPS telemetry lock...")
                await asyncio.sleep(2.0)
        except KeyboardInterrupt:
            self.log("Monitor stopped by user.")
        finally:
            self.running = False
            self.log("Telemetry monitor stopped.")

    async def run_programmatic_mode(self):
        # 1. Parse instructions
        self.log(f"Loading YAML flight instructions from: {self.file_path}")
        try:
            with open(self.file_path, 'r') as f:
                instructions = yaml.safe_load(f)
        except Exception as e:
            self.log(f"[ERROR] Failed to read/parse YAML file: {e}")
            return

        # Validate start location
        if "start_location" not in instructions:
            self.log("[ERROR] Missing 'start_location' in YAML instructions.")
            return
        
        start_lat = instructions["start_location"].get("latitude")
        start_lon = instructions["start_location"].get("longitude")
        if start_lat is None or start_lon is None:
            self.log("[ERROR] Missing latitude or longitude in 'start_location'.")
            return

        self.drone = System()
        self.log(f"Connecting to MAVSDK drone system in AUTONOMOUS mode at: {self.address}")
        await self.drone.connect(system_address=self.address)
        
        # Start telemetry listeners
        asyncio.create_task(self._listen_position())
        asyncio.create_task(self._listen_home())
        asyncio.create_task(self._listen_armed())
        asyncio.create_task(self._listen_flight_mode())
        
        # Wait for telemetry connection and GPS lock
        self.log("Waiting for GPS lock and home position...")
        while self.current_lat is None or self.current_lon is None or self.home_abs_alt is None:
            await asyncio.sleep(0.5)
            
        self.home_lat = self.current_lat
        self.home_lon = self.current_lon
        self.log(f"GPS Lock acquired. Home Position: ({self.home_lat:.6f}, {self.home_lon:.6f}), Abs Alt: {self.home_abs_alt:.1f}m")
        
        # Verify drone is close to start location
        dist_from_start = get_distance_meters(self.home_lat, self.home_lon, start_lat, start_lon)
        self.log(f"Drone is {dist_from_start:.1f} meters away from the configured start_location.")
        if dist_from_start > 100.0:
            self.log("[WARNING] Drone is too far from start_location (> 100 meters). Aborting flight for safety.")
            return

        # Initialize flight target position tracking
        target_lat = self.home_lat
        target_lon = self.home_lon
        target_alt_m = 0.0
        
        # Earth radius for distance to coordinate conversion
        R_EARTH = 6378137.0

        # Execute commands
        commands = instructions.get("commands", [])
        self.log(f"Parsed {len(commands)} commands. Starting execution...")
        
        try:
            for idx, cmd in enumerate(commands, 1):
                action = cmd.get("action")
                self.log(f"[{idx}/{len(commands)}] Executing Action: {action.upper()}")

                if action == "arm":
                    await self.drone.action.arm()
                    self.log("Vehicle armed.")

                elif action == "takeoff":
                    alt = float(cmd.get("altitude", 10.0))
                    unit = cmd.get("unit", "meters").lower()
                    
                    # Convert feet to meters
                    if "foot" in unit or "feet" in unit:
                        alt_m = alt * 0.3048
                    else:
                        alt_m = alt
                    
                    target_alt_m = alt_m
                    self.log(f"Taking off to target altitude: {alt:.1f} {unit} ({alt_m:.1f} meters)...")
                    await self.drone.action.takeoff()
                    
                    # Wait for initial takeoff launch (climb above 2.0 meters)
                    while self.current_alt < 2.0:
                        self.log(f"Launching... Current Altitude: {self.current_alt:.1f}m / 2.0m")
                        await asyncio.sleep(1.0)
                    
                    # Command the drone to climb to target altitude at the home position
                    self.log(f"Climbing to target altitude of {target_alt_m:.1f}m...")
                    target_abs_alt = self.home_abs_alt + target_alt_m
                    await self.drone.action.goto_location(self.home_lat, self.home_lon, target_abs_alt, 0.0)
                    
                    # Wait for takeoff completion (climb above target - 0.5m)
                    while self.current_alt < (target_alt_m - 0.5):
                        self.log(f"Climbing... Current Altitude: {self.current_alt:.1f}m / {target_alt_m:.1f}m")
                        await asyncio.sleep(1.0)
                    self.log("Takeoff and climb complete.")

                elif action == "fly":
                    dist = float(cmd.get("distance"))
                    unit = cmd.get("unit", "meters").lower()
                    direction = cmd.get("direction").lower()
                    
                    # Convert feet to meters
                    if "foot" in unit or "feet" in unit:
                        dist_m = dist * 0.3048
                    else:
                        dist_m = dist
                        
                    # Calculate new GPS target coordinates
                    if direction == "north":
                        target_lat += (dist_m / R_EARTH) * (180.0 / math.pi)
                    elif direction == "south":
                        target_lat -= (dist_m / R_EARTH) * (180.0 / math.pi)
                    elif direction == "east":
                        target_lon += (dist_m / (R_EARTH * math.cos(math.radians(target_lat)))) * (180.0 / math.pi)
                    elif direction == "west":
                        target_lon -= (dist_m / (R_EARTH * math.cos(math.radians(target_lat)))) * (180.0 / math.pi)
                    else:
                        self.log(f"[WARNING] Unknown direction '{direction}'. Skipping command.")
                        continue
                    
                    self.log(f"Flying {dist:.1f} {unit} {direction.upper()} to GPS: ({target_lat:.6f}, {target_lon:.6f})")
                    target_abs_alt = self.home_abs_alt + target_alt_m
                    await self.drone.action.goto_location(target_lat, target_lon, target_abs_alt, 0.0)
                    
                    # Wait until arrival within 1.5 meters
                    loop_cnt = 0
                    while self.running:
                        dist_rem = get_distance_meters(self.current_lat, self.current_lon, target_lat, target_lon)
                        loop_cnt += 1
                        if loop_cnt % 6 == 0:
                            self.log(f"Distance remaining: {dist_rem:.1f}m (Altitude: {self.current_alt:.1f}m)")
                        if dist_rem < 1.5:
                            self.log(f"Arrived at waypoint target.")
                            break
                        await asyncio.sleep(0.5)

                elif action == "land":
                    self.log("Landing drone...")
                    await self.drone.action.land()
                    
                    # Wait for landing
                    while True:
                        if self.current_alt < 0.3:
                            self.log("Drone landed.")
                            break
                        self.log(f"Landing... Altitude: {self.current_alt:.1f}m")
                        await asyncio.sleep(1.0)
                        
                    self.log("Disarming drone...")
                    try:
                        await self.drone.action.disarm()
                    except ActionError:
                        self.log("Drone disarmed (automatically or explicitly).")

                else:
                    self.log(f"[WARNING] Unknown action type '{action}'. Skipping command.")

            self.log("All programmatic flight commands executed successfully!")

        except ActionError as e:
            self.log(f"[FLIGHT ACTION ERROR] Action failed or rejected: {e}")
        except Exception as e:
            self.log(f"[ERROR] Unexpected error during autonomous route: {e}")
        finally:
            self.running = False
            self.log("Autonomous flight controller shut down.")

    # Telemetry listener handlers
    async def _listen_position(self):
        async for position in self.drone.telemetry.position():
            self.current_lat = position.latitude_deg
            self.current_lon = position.longitude_deg
            self.current_alt = position.relative_altitude_m

    async def _listen_home(self):
        async for home in self.drone.telemetry.home():
            self.home_abs_alt = home.absolute_altitude_m
            break

    async def _listen_armed(self):
        async for is_armed in self.drone.telemetry.armed():
            self.is_armed = is_armed

    async def _listen_flight_mode(self):
        async for flight_mode in self.drone.telemetry.flight_mode():
            self.flight_mode = str(flight_mode)

    async def _listen_battery(self):
        async for battery in self.drone.telemetry.battery():
            self.battery_remaining = battery.remaining_percent

def main():
    parser = argparse.ArgumentParser(description="Multi-Mode Drone Flight Script at Hawthorne-Feather Airpark")
    parser.add_argument("--address", type=str, default="udpin://127.0.0.1:14540",
                        help="MAVLink port address for connection (default: udpin://127.0.0.1:14540)")
    parser.add_argument("--args", type=str, choices=["qgroundcontrol", "programmatic"], default="programmatic",
                        help="Flight execution mode: 'qgroundcontrol' (passive monitor) or 'programmatic' (execute flight instructions)")
    parser.add_argument("--file", type=str, default="flight_instructions.yaml",
                        help="Path to YAML instructions file (required for programmatic mode)")
    args = parser.parse_args()

    controller = HawthorneFlightController(args.address, args.args, args.file)
    try:
        asyncio.run(controller.run())
    except KeyboardInterrupt:
        print("\nFlight script interrupted by user.")

if __name__ == "__main__":
    main()
