#!/usr/bin/env python3
import rclpy
import numpy as np

from rclpy.node import Node
from wamv_msgs.msg import GuidanceType, NavigationType, ControlType, WAMVInfo
from std_msgs.msg import Header

class WAMVInfoNode(Node):
    def __init__(self):
        super().__init__('wamv_info_node')

        # Initial subscriber data
        self.guidance_data   = None
        self.navigation_data = None
        self.control_data    = None

        # subscriber
        self.guidance_subscriber   = self.create_subscription(GuidanceType,   '/guidance',   self.guidance_callback,   10)
        self.navigation_subscriber = self.create_subscription(NavigationType, '/navigation', self.navigation_callback, 20)
        self.control_subscriber    = self.create_subscription(ControlType,    '/control',    self.control_callback,    10)

        # publisher
        self.wamv_info_publisher = self.create_publisher(WAMVInfo, '/wamv_info', 10)

        self.dt = 0.1

        self.timer = self.create_timer(self.dt, self.publish_wamv_info_msg)

    def guidance_callback(self, msg):
        self.guidance_data = msg
        self.desired_psi   = self.guidance_data.desired_psi
        self.desired_u     = self.guidance_data.desired_u
        self.error_psi     = self.guidance_data.error_psi
        self.error_u       = self.guidance_data.error_u

    def navigation_callback(self, msg):
        self.navigation_data = msg 
        self.x               = self.navigation_data.x
        self.y               = self.navigation_data.y
        self.psi             = self.navigation_data.psi
        self.u               = self.navigation_data.u

    def control_callback(self, msg):
        self.control_data       = msg 
        self.delta_psi          = self.control_data.delta_psi
        self.delta_u            = self.control_data.delta_u
        self.pwm_standard       = self.control_data.pwm_standard
        self.thruster_pwm_port  = self.control_data.thruster_pwm_port
        self.thruster_pwm_stbd  = self.control_data.thruster_pwm_stbd
        self.azimuth_angle_port = self.control_data.azimuth_angle_port
        self.azimuth_angle_stbd = self.control_data.azimuth_angle_stbd

    def publish_wamv_info_msg(self):
        if self.guidance_data is None or self.navigation_data is None or self.control_data is None:
            return

        header          = Header()
        header.stamp    = self.get_clock().now().to_msg()
        header.frame_id = 'WAM-V'

        wamv_info                 = WAMVInfo()
        wamv_info.header          = header
        wamv_info.guidance        = self.guidance_data
        wamv_info.navigation      = self.navigation_data
        wamv_info.control         = self.control_data
        self.wamv_info_publisher.publish(wamv_info)

def main(args=None):
    rclpy.init(args=args)
    try:
        wamv_info_node = WAMVInfoNode()
        try:
            rclpy.spin(wamv_info_node)
        except KeyboardInterrupt:
            wamv_info_node.get_logger().info('Keyboard Interrupt (SIGINT)')
        finally:
            wamv_info_node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
