#!/usr/bin/env python3

"""
CARLA Image Zenoh Subscriber

This script subscribes to camera images published by the CARLA manual control
via Zenoh and displays them in real-time using matplotlib.

Usage:
    python image_subscriber.py [--router IP_ADDRESS]

Controls:
    - Close the matplotlib window to exit
    - The script will automatically update the display with new frames
"""

import argparse
import base64
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import zenoh
import time
from collections import deque


class ImageSubscriber:
    def __init__(self, router_ip="127.0.0.1"):
        """
        Initialize the image subscriber.
        
        Args:
            router_ip: IP address of the Zenoh router
        """
        self.router_ip = router_ip
        self.latest_image = None
        self.image_queue = deque(maxlen=10)  # Keep last 10 images
        self.frame_count = 0
        self.last_timestamp = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("CARLA Camera Feed")
        self.ax.axis('off')
        self.im = None
        
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
                    # RGBA, convert to RGB
                    image_array = image_array.reshape((height, width, channels))
                    return image_array[:, :, :3]
                else:
                    # RGB
                    return image_array.reshape((height, width, channels))
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
                elif channels in [3, 4]:
                    reshaped = image_array.reshape((height, width, channels))
                    return reshaped[:, :, :3] if channels == 4 else reshaped
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
                    return image_array
            
            # If no common size matches, try to calculate dimensions
            # Assume 3 channels (RGB)
            total_pixels = data_length // 3
            
            # Try square-ish aspect ratios
            for ratio in [4/3, 16/9, 3/2, 1.0]:
                width = int(np.sqrt(total_pixels * ratio))
                height = int(total_pixels / width)
                
                if width * height * 3 == data_length:
                    image_array = np.frombuffer(image_data, dtype=np.uint8)
                    image_array = image_array.reshape((height, width, 3))
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
            
            # For visualization, reshape to a reasonable size if possible
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
                    return image_array.reshape((height, width, 3))
                elif data_length == width * height * 4:
                    # RGBA, convert to RGB
                    rgba = image_array.reshape((height, width, 4))
                    return rgba[:, :, :3]
            
            # If we can't determine the shape, return a 1D representation
            return image_array
            
        except Exception as e:
            print(f"[ERROR] Failed to decode generic image: {e}")
            return None
    
    def _update_plot(self, frame):
        """
        Update function for matplotlib animation.
        
        Args:
            frame: Frame number (unused)
        """
        if self.latest_image is not None:
            if self.im is None:
                # First time setup
                if len(self.latest_image.shape) == 3:
                    self.im = self.ax.imshow(self.latest_image)
                else:
                    self.im = self.ax.imshow(self.latest_image, cmap='gray')
            else:
                # Update existing image
                self.im.set_array(self.latest_image)
                
            # Update title with info
            self.ax.set_title(f"CARLA Camera Feed - Frame: {self.frame_count} | "
                            f"FPS: {self.current_fps} | Timestamp: {self.last_timestamp}")
            
        return [self.im] if self.im else []
    
    def start_display(self):
        """Start the matplotlib display with animation."""
        try:
            print("Starting image display...")
            print("Close the matplotlib window to exit.")
            
            # Create animation
            _ani = animation.FuncAnimation(
                self.fig, self._update_plot, interval=50, blit=False, cache_frame_data=False
            )
            
            # Show the plot
            plt.show()
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"[ERROR] Display error: {e}")
        finally:
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
    parser = argparse.ArgumentParser(description='CARLA Image Zenoh Subscriber')
    parser.add_argument(
        '--router',
        default='127.0.0.1',
        help='IP address of the Zenoh router (default: 127.0.0.1)'
    )
    
    args = parser.parse_args()
    
    print("CARLA Image Subscriber")
    print("=" * 50)
    print(f"Router: {args.router}:7447")
    print("Topic: vehicle/camera/image")
    print("=" * 50)
    
    try:
        subscriber = ImageSubscriber(args.router)
        subscriber.start_display()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"[ERROR] Failed to start subscriber: {e}")


if __name__ == '__main__':
    main()