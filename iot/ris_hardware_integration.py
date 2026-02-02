# iot/ris_hardware_integration.py

"""
RIS Hardware Integration for Raspberry Pi Pico W
Combines existing sensors with RIS passive resonator platform
"""

import machine
import network
import urequests
import ujson
import time
import math
from machine import Pin, SPI, I2C, ADC

class RISHardwareController:
    """
    Unified hardware controller for RIS + existing sensors
    """
    
    def __init__(self):
        # Existing sensor setup
        self.setup_existing_sensors()
        
        # RIS-specific hardware
        self.setup_ris_hardware()
        
        # WiFi setup
        self.setup_wifi()
        
        # Data buffers
        self.rf_buffer = []
        self.sensor_buffer = []
        
    def setup_existing_sensors(self):
        """Setup existing MAX30102, AD8232, etc."""
        # I2C for MAX30102 and MLX90614
        self.i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=400000)
        
        # SPI for potential future sensors
        self.spi = SPI(0, baudrate=1000000, polarity=0, phase=0,
                      sck=Pin(2), mosi=Pin(3), miso=Pin(4))
        
        # ADC for AD8232 ECG
        self.ecg_adc = ADC(Pin(26))
        
        # DHT11 temperature/humidity
        self.dht_pin = Pin(22)
        
        print("Existing sensors initialized")
    
    def setup_ris_hardware(self):
        """Setup RIS-specific hardware components"""
        
        # RIS Control Pins (example configuration)
        self.ris_control_pins = []
        for i in range(8, 16):  # 8 RIS elements example
            pin = Pin(i, Pin.OUT)
            self.ris_control_pins.append(pin)
        
        # RF Signal Generator Control (via SPI or I2C)
        # This would typically be an external RF synthesizer
        self.rf_gen_cs = Pin(17, Pin.OUT, value=1)
        
        # RF Receiver ADC (high-speed sampling)
        self.rf_adc = ADC(Pin(27))
        
        # Phase control for RIS elements
        self.current_phase_config = [0] * len(self.ris_control_pins)
        
        print(f"RIS hardware initialized: {len(self.ris_control_pins)} elements")
    
    def setup_wifi(self):
        """Setup WiFi connection"""
        self.wlan = network.WLAN(network.STA_IF)
        self.wlan.active(True)
        
        # Replace with your WiFi credentials
        ssid = "YOUR_WIFI_SSID"
        password = "YOUR_WIFI_PASSWORD"
        
        if not self.wlan.isconnected():
            print("Connecting to WiFi...")
            self.wlan.connect(ssid, password)
            
            timeout = 10
            while not self.wlan.isconnected() and timeout > 0:
                time.sleep(1)
                timeout -= 1
            
            if self.wlan.isconnected():
                print(f"WiFi connected: {self.wlan.ifconfig()}")
            else:
                print("WiFi connection failed")
    
    def configure_ris_phase(self, phase_config):
        """
        Configure RIS element phases for beamforming
        phase_config: list of phase values in radians
        """
        for i, phase in enumerate(phase_config):
            if i < len(self.ris_control_pins):
                # Convert phase to digital control signal
                # This is simplified - actual implementation depends on RIS hardware
                digital_value = int((phase / (2 * math.pi)) * 255) % 256
                
                # For demonstration, using PWM to control phase
                # Real implementation would use phase shifters
                pwm = machine.PWM(self.ris_control_pins[i])
                pwm.freq(1000)
                pwm.duty_u16(digital_value * 256)
                
        self.current_phase_config = phase_config.copy()
        print(f"RIS phase configured: {len(phase_config)} elements")
    
    def collect_rf_backscatter(self, duration_ms=5000, sample_rate=1000):
        """
        Collect RF backscatter data from RIS
        """
        samples_needed = int(duration_ms * sample_rate / 1000)
        rf_data = []
        
        print(f"Collecting {samples_needed} RF samples...")
        
        start_time = time.ticks_ms()
        for i in range(samples_needed):
            # Read RF ADC value
            rf_value = self.rf_adc.read_u16()
            rf_data.append(rf_value)
            
            # Maintain sample rate
            target_time = start_time + int(i * 1000 / sample_rate)
            while time.ticks_ms() < target_time:
                pass
        
        # Convert to 2D array (frequency bins x time samples)
        # This is simplified - real implementation would do FFT processing
        rf_matrix = []
        chunk_size = 64  # Frequency bins
        
        for i in range(0, len(rf_data), chunk_size):
            chunk = rf_data[i:i+chunk_size]
            if len(chunk) == chunk_size:
                rf_matrix.append(chunk)
        
        return rf_matrix
    
    def collect_existing_sensors(self):
        """Collect data from existing sensors"""
        sensor_data = {}
        
        try:
            # MAX30102 (simplified - would need actual library)
            sensor_data["heart_rate"] = 75.0  # Placeholder
            sensor_data["spo2"] = 98.0
            
            # AD8232 ECG
            ecg_samples = []
            for _ in range(100):  # 100 samples
                ecg_samples.append(self.ecg_adc.read_u16())
                time.sleep_ms(4)  # 250Hz sampling
            sensor_data["ecg_samples"] = ecg_samples
            
            # MLX90614 temperature (simplified)
            sensor_data["body_temp"] = 36.5
            
            # DHT11 (simplified)
            sensor_data["ambient_temp"] = 25.0
            sensor_data["humidity"] = 50.0
            
        except Exception as e:
            print(f"Sensor collection error: {e}")
            
        return sensor_data
    
    def adaptive_ris_beamforming(self, target_distance):
        """
        Adaptive beamforming based on target distance
        """
        # Calculate optimal phase configuration for target distance
        wavelength = 3e8 / 2.45e9  # ~12.2 cm at 2.45 GHz
        
        phase_config = []
        for i in range(len(self.ris_control_pins)):
            # Simple linear array beamforming
            element_spacing = wavelength / 2
            phase_shift = (2 * math.pi * element_spacing * i * 
                          math.sin(math.atan(target_distance / (i * element_spacing))))
            phase_config.append(phase_shift % (2 * math.pi))
        
        self.configure_ris_phase(phase_config)
        return phase_config
    
    def send_to_backend(self, rf_data, sensor_data, phase_config):
        """Send collected data to FastAPI backend"""
        
        payload = {
            "rf_backscatter_data": rf_data,
            "ris_phase_config": phase_config,
            "target_distance": 1.0,  # meters
            "frequency_range": [2.4e9, 2.5e9],
            "signal_quality": 0.85,
            "sensor_data": sensor_data,
            "timestamp": time.time()
        }
        
        try:
            # Send to RIS endpoint
            response = urequests.post(
                "http://YOUR_BACKEND_IP:8000/assess/ris-resonator",
                headers={"Content-Type": "application/json"},
                data=ujson.dumps(payload)
            )
            
            result = response.json()
            response.close()
            
            print(f"Backend response: {result.get('risk_level', 'Unknown')}")
            return result
            
        except Exception as e:
            print(f"Backend communication error: {e}")
            return None
    
    def run_continuous_monitoring(self):
        """Main monitoring loop"""
        print("Starting RIS continuous monitoring...")
        
        while True:
            try:
                # 1. Adaptive beamforming
                target_distance = 1.0  # meters, could be dynamic
                phase_config = self.adaptive_ris_beamforming(target_distance)
                
                # 2. Collect RF backscatter data
                rf_data = self.collect_rf_backscatter(duration_ms=5000)
                
                # 3. Collect existing sensor data
                sensor_data = self.collect_existing_sensors()
                
                # 4. Send to backend for analysis
                result = self.send_to_backend(rf_data, sensor_data, phase_config)
                
                if result:
                    # Process result (LED indicators, alerts, etc.)
                    risk_level = result.get("risk_level", "UNKNOWN")
                    vital_signs = result.get("vital_signs", {})
                    
                    print(f"Risk: {risk_level}")
                    print(f"HR: {vital_signs.get('heart_rate', 0):.1f} BPM")
                    print(f"RR: {vital_signs.get('respiratory_rate', 0):.1f} BPM")
                
                # Wait before next cycle
                time.sleep(5)
                
            except Exception as e:
                print(f"Monitoring loop error: {e}")
                time.sleep(10)  # Longer wait on error

# Usage example
def main():
    controller = RISHardwareController()
    controller.run_continuous_monitoring()

if __name__ == "__main__":
    main()