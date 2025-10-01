#!/usr/bin/env python3

"""
CARLA Image Zenoh Subscriber (OpenCV version)

This script subscribes to camera images published by the CARLA manual control
via Zenoh and displays them in real-time using OpenCV.

Usage:
    python image_subscriber_cv.py [--router IP_ADDRESS]

Controls:
    - Press 'q' or ESC to exit
    - Press 's' to save current frame
    - Press 'r' to reset frame counter
"""

import argparse
import base64
import json
import numpy as np
import zenoh
import time
import os
from collections import deque

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed. Please install it with:")
    print("pip install opencv-python")
    exit(1)


class ImageSubscriberCV:
    def __init__(self, router_ip="127.0.0.1"):
        """
        Initialize the image subscriber with OpenCV display.
        
        Args:
            router_ip: IP address of the Zenoh router
        """
        self.router_ip = router_ip
        self.latest_image = None
        self.image_queue = deque(maxlen=10)
        self.frame_count = 0
        self.last_timestamp = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.save_counter = 0
        
        # Create output directory for saved images
        self.output_dir = "saved_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Zenoh session
        self._setup_zenoh()
        
    def _setup_zenoh(self):
        """Setup Zenoh session and subscriber."""
        try:
            # Configure Zenoh
            zenoh_config = zenoh.Config()
            zenoh_config.insert_json5("mode", json.dumps("peer"))
            zenoh_config.insert_json5("connect/endpoints", json.dumps([f"tcp/{self.router_ip}:7447"]))
            
            # Open session
            self.session = zenoh.open(zenoh_config)
            print(f"Connected to Zenoh router at {self.router_ip}:7447")
            
            # Create subscriber
            self.subscriber = self.session.declare_subscriber(
                "vehicle/camera/image", 
                self._image_callback
            )
            print("Subscribed to topic: vehicle/camera/image")
            
        except Exception as e:
            print(f"[ERROR] Failed to setup Zenoh: {e}")
            raise
    
    def _image_callback(self, sample):
        """
        Callback function for received images.
        
        Args:
            sample: Zenoh sample containing the image data
        """
        try:
            # Parse the JSON message
            if hasattr(sample.payload, 'to_string'):
                message_str = sample.payload.to_string()
            else:
                # Handle bytes payload directly
                message_str = sample.payload.decode('utf-8') if isinstance(sample.payload, bytes) else str(sample.payload)
            message = json.loads(message_str)
            
            # Extract metadata
            metadata = message.get("metadata", {})
            image_format = metadata.get("format", "unknown")
            timestamp = metadata.get("timestamp", 0)
            encoding = metadata.get("encoding", "base64")
            width = metadata.get("width", 0)
            height = metadata.get("height", 0)
            channels = metadata.get("channels", 3)
            
            # Decode image data
            if encoding == "base64":
                image_data = base64.b64decode(message["data"])
            else:
                print(f"[WARNING] Unsupported encoding: {encoding}")
                return
            
            # Convert to numpy array based on format
            if image_format == "rgb":
                if width > 0 and height > 0:
                    # Use metadata dimensions if available
                    image_array = self._decode_rgb_image_with_dims(image_data, width, height, channels)
                else:
                    # Fallback to dimension guessing
                    image_array = self._decode_rgb_image(image_data)
            else:
                print(f"[INFO] Received {image_format} image (timestamp: {timestamp})")
                if width > 0 and height > 0:
                    image_array = self._decode_generic_image_with_dims(image_data, width, height, channels)
                else:
                    image_array = self._decode_generic_image(image_data)
            
            if image_array is not None:
                # Update statistics
                self.frame_count += 1
                self.last_timestamp = timestamp
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                else:
                    self.fps_counter += 1
                
                # Store the latest image
                self.latest_image = image_array
                self.image_queue.append({
                    'image': image_array,
                    'timestamp': timestamp,
                    'format': image_format
                })
                
                if self.frame_count % 30 == 0:  # Print every 30 frames to reduce spam
                    print(f"[INFO] Frame {self.frame_count}: {image_format} image "
                          f"({image_array.shape}) - FPS: {self.current_fps}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process image: {e}")
    
    def _decode_rgb_image_with_dims(self, image_data, width, height, channels):
        """
        Decode RGB image data with known dimensions.
        
        Args:
            image_data: Raw image bytes
            width: Image width in pixels
            height: Image height in pixels  
            channels: Number of channels (3 for RGB, 4 for RGBA)
            
        Returns:
            numpy array representing the image or None if failed
        """
        try:
            expected_size = width * height * channels
            if len(image_data) == expected_size:
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                if channels == 4:
                    # RGBA to BGR
                    image_array = image_array.reshape((height, width, channels))
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                else:
                    # RGB to BGR
                    image_array = image_array.reshape((height, width, channels))
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                return image_array
            else:
                print(f"[WARNING] Size mismatch: expected {expected_size}, got {len(image_data)}")
                return self._decode_rgb_image(image_data)
        except Exception as e:
            print(f"[ERROR] Failed to decode RGB image with dims: {e}")
            return self._decode_rgb_image(image_data)

    def _decode_generic_image_with_dims(self, image_data, width, height, channels):
        """
        Decode generic image data with known dimensions.
        
        Args:
            image_data: Raw image bytes
            width: Image width in pixels
            height: Image height in pixels
            channels: Number of channels
            
        Returns:
            numpy array or None if failed
        """
        try:
            expected_size = width * height * channels
            if len(image_data) == expected_size:
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                if channels == 1:
                    return image_array.reshape((height, width))
                elif channels == 3:
                    reshaped = image_array.reshape((height, width, channels))
                    return cv2.cvtColor(reshaped, cv2.COLOR_RGB2BGR)
                elif channels == 4:
                    rgba = image_array.reshape((height, width, channels))
                    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                else:
                    return image_array.reshape((height, width, channels))
            else:
                print(f"[WARNING] Size mismatch: expected {expected_size}, got {len(image_data)}")
                return self._decode_generic_image(image_data)
        except Exception as e:
            print(f"[ERROR] Failed to decode generic image with dims: {e}")
            return self._decode_generic_image(image_data)

    def _decode_rgb_image(self, image_data):
        """
        Decode RGB image data.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            numpy array representing the image or None if failed
        """
        try:
            # Try common CARLA camera resolutions
            common_sizes = [
                (640, 480, 3),   # 640x480 RGB
                (800, 600, 3),   # 800x600 RGB
                (1024, 768, 3),  # 1024x768 RGB
                (1280, 720, 3),  # 720p RGB
                (1920, 1080, 3), # 1080p RGB
            ]
            
            data_length = len(image_data)
            
            for width, height, channels in common_sizes:
                expected_size = width * height * channels
                if data_length == expected_size:
                    image_array = np.frombuffer(image_data, dtype=np.uint8)
                    image_array = image_array.reshape((height, width, channels))
                    # Convert RGB to BGR for OpenCV
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    return image_array
            
            # If no common size matches, try to calculate dimensions
            total_pixels = data_length // 3
            
            # Try square-ish aspect ratios
            for ratio in [4/3, 16/9, 3/2, 1.0]:
                width = int(np.sqrt(total_pixels * ratio))
                height = int(total_pixels / width)
                
                if width * height * 3 == data_length:
                    image_array = np.frombuffer(image_data, dtype=np.uint8)
                    image_array = image_array.reshape((height, width, 3))
                    # Convert RGB to BGR for OpenCV
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    return image_array
            
            print(f"[WARNING] Could not determine image dimensions for {data_length} bytes")
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to decode RGB image: {e}")
            return None
    
    def _decode_generic_image(self, image_data):
        """
        Generic image decoder for non-RGB formats.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            numpy array or None if failed
        """
        try:
            # Try to decode as raw bytes and reshape based on size
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            
            data_length = len(image_data)
            
            # Try common single-channel sizes first
            common_sizes = [
                (640, 480),   # 640x480
                (800, 600),   # 800x600
                (1024, 768),  # 1024x768
            ]
            
            for width, height in common_sizes:
                if data_length == width * height:
                    return image_array.reshape((height, width))
                elif data_length == width * height * 3:
                    reshaped = image_array.reshape((height, width, 3))
                    return cv2.cvtColor(reshaped, cv2.COLOR_RGB2BGR)
                elif data_length == width * height * 4:
                    # RGBA, convert to BGR
                    rgba = image_array.reshape((height, width, 4))
                    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            
            # If we can't determine the shape, try to make a square image
            sqrt_len = int(np.sqrt(data_length))
            if sqrt_len * sqrt_len == data_length:
                return image_array.reshape((sqrt_len, sqrt_len))
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to decode generic image: {e}")
            return None
    
    def _save_current_frame(self):
        """Save the current frame to disk."""
        if self.latest_image is not None:
            filename = f"frame_{self.save_counter:05d}_ts_{self.last_timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, self.latest_image)
            print(f"[INFO] Saved frame to: {filepath}")
            self.save_counter += 1
        else:
            print("[WARNING] No image to save")
    
    def start_display(self):
        """Start the OpenCV display loop."""
        try:
            print("Starting image display with OpenCV...")
            print("Controls:")
            print("  - Press 'q' or ESC to exit")
            print("  - Press 's' to save current frame")
            print("  - Press 'r' to reset frame counter")
            print("  - Press 'h' to show this help")
            
            window_name = "CARLA Camera Feed"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            while True:
                if self.latest_image is not None:
                    # Create a copy for display with info overlay
                    display_image = self.latest_image.copy()
                    
                    # Add text overlay with information
                    info_text = f"Frame: {self.frame_count} | FPS: {self.current_fps} | TS: {self.last_timestamp}"
                    cv2.putText(display_image, info_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show the image
                    cv2.imshow(window_name, display_image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Save frame
                    self._save_current_frame()
                elif key == ord('r'):  # Reset counter
                    self.frame_count = 0
                    print("[INFO] Frame counter reset")
                elif key == ord('h'):  # Help
                    print("Controls:")
                    print("  - Press 'q' or ESC to exit")
                    print("  - Press 's' to save current frame")
                    print("  - Press 'r' to reset frame counter")
                    print("  - Press 'h' to show this help")
                
                # Small delay to prevent high CPU usage
                time.sleep(0.01)
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"[ERROR] Display error: {e}")
        finally:
            cv2.destroyAllWindows()
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'subscriber'):
                self.subscriber.undeclare()
            if hasattr(self, 'session'):
                self.session.close()
            print("Cleanup completed")
        except Exception as e:
            print(f"[ERROR] Cleanup failed: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='CARLA Image Zenoh Subscriber (OpenCV)')
    parser.add_argument(
        '--router',
        default='127.0.0.1',
        help='IP address of the Zenoh router (default: 127.0.0.1)'
    )
    
    args = parser.parse_args()
    
    print("CARLA Image Subscriber (OpenCV)")
    print("=" * 50)
    print(f"Router: {args.router}:7447")
    print("Topic: vehicle/camera/image")
    print("=" * 50)
    
    try:
        subscriber = ImageSubscriberCV(args.router)
        subscriber.start_display()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"[ERROR] Failed to start subscriber: {e}")


if __name__ == '__main__':
    main()