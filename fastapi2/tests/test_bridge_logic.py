import unittest
from unittest.mock import MagicMock, patch
import queue
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from bridge import HardwareBridge, BridgeConfig, RadarReader, ESP32Reader

class TestBridgeAggregation(unittest.TestCase):
    def setUp(self):
        self.config = BridgeConfig(
            face_capture_seconds=1, # Short duration for test
            body_capture_seconds=1,
            api_url="http://mock-api"
        )
        self.bridge = HardwareBridge(self.config)
        
        # Mock readers
        self.bridge.radar_reader = MagicMock()
        self.bridge.radar_reader.data_queue = queue.Queue()
        
        self.bridge.esp32_reader = MagicMock()
        self.bridge.esp32_reader.data_queue = queue.Queue()
        
        # Mock camera
        self.bridge.camera = MagicMock()
        self.bridge.camera.capture_and_process_frames.return_value = ([], 0)
        
        # Mock DataFusion
        self.bridge.data_fusion = MagicMock()
        self.bridge.data_fusion.build_screening_request.return_value = {"systems": []}
        self.bridge.data_fusion.send_screening.return_value = {"screening_id": "test"}

    def test_radar_aggregation(self):
        # Populate queue with known values
        # Avg HR should be (60+80)/2 = 70
        # Avg Resp should be (10+20)/2 = 15
        data1 = {"radar": {"heart_rate": 60, "respiration_rate": 10.0}}
        data2 = {"radar": {"heart_rate": 80, "respiration_rate": 20.0}}
        
        self.bridge.radar_reader.data_queue.put(data1)
        self.bridge.radar_reader.data_queue.put(data2)
        
        # Run screening (this clears queues at start, so we need to mock the check to NOT clear or 
        # populating AFTER the clear.
        # Ah, run_single_screening clears queues at the START.
        # So we can't easily test it unless we populate the queue *during* the camera capture.
        
        # Let's mock camera.capture_and_process_frames to populate the queue as a side effect
        def side_effect(*args, **kwargs):
            self.bridge.radar_reader.data_queue.put(data1)
            self.bridge.radar_reader.data_queue.put(data2)
            return [], 0
            
        self.bridge.camera.capture_and_process_frames.side_effect = side_effect
        
        self.bridge.run_single_screening("test_patient")
        
        # Check what was passed to build_screening_request
        call_args = self.bridge.data_fusion.build_screening_request.call_args
        radar_arg = call_args.kwargs.get('radar_data')
        
        print(f"Aggregated Radar Data: {radar_arg}")
        
        self.assertEqual(radar_arg['radar']['heart_rate'], 70)
        self.assertEqual(radar_arg['radar']['respiration_rate'], 15.0)

    def test_esp32_aggregation(self):
        # Real firmware format: core_regions/stability_metrics/symmetry/gradients
        # Canthus temps: 32.0, 33.0 -> Avg 32.5
        # Neck temps: 30.0, 31.0 -> Avg 30.5 
        data1 = {"thermal": {"core_regions": {"canthus_mean": 32.0, "neck_mean": 30.0}, "stability_metrics": {"canthus_range": 0.4}, "symmetry": {"cheek_asymmetry": 0.2}, "gradients": {"forehead_nose_gradient": -2.0}}}
        data2 = {"thermal": {"core_regions": {"canthus_mean": 33.0, "neck_mean": 31.0}, "stability_metrics": {"canthus_range": 0.6}, "symmetry": {"cheek_asymmetry": 0.3}, "gradients": {"forehead_nose_gradient": -1.0}}}
        
        def side_effect(*args, **kwargs):
            self.bridge.esp32_reader.data_queue.put(data1)
            self.bridge.esp32_reader.data_queue.put(data2)
            return [], 0
            
        self.bridge.camera.capture_and_process_frames.side_effect = side_effect
        
        self.bridge.run_single_screening("test_patient")
        
        call_args = self.bridge.data_fusion.build_screening_request.call_args
        esp32_arg = call_args.kwargs.get('esp32_data')
        
        print(f"Aggregated Thermal Data: {esp32_arg}")
        
        self.assertEqual(esp32_arg['thermal']['core_regions']['canthus_mean'], 32.5)
        self.assertEqual(esp32_arg['thermal']['core_regions']['neck_mean'], 30.5)
        self.assertEqual(esp32_arg['thermal']['stability_metrics']['canthus_range'], 0.5)
        self.assertEqual(esp32_arg['thermal']['symmetry']['cheek_asymmetry'], 0.25)
        self.assertEqual(esp32_arg['thermal']['gradients']['forehead_nose_gradient'], -1.5)

if __name__ == '__main__':
    unittest.main()
