import rospy

from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                 accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        
        # TODO: Implement
        min_speed = 0.1
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        
        ############################
        #    Set PID controller    #
        ############################
        
        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0  # minimum throttle value
        mx = 0.2  # maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        
        ############################
        # Velocity Low Pass filter #
        ############################
        
        # Reject high frequency noises in velocity
        tau = 0.5 # cut off frequency: 1/(2*pi*tau)
        ts  = 0.02 # sampling time
        self.vel_lpf = LowPassFilter(tau, ts)
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        
        # Comfort parameters
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        
        self.wheel_radius = wheel_radius
        self.last_time    = rospy.get_time()

    # the control method is intended to run at 50Hz from the caller
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        # Reset the PID controller
        # when DBW is not enabled
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
        
        current_vel = self.vel_lpf.filt(current_vel)
        
        # rospy.logwarn("Angular velocity: {0}\n".format(angular_vel))
        # rospy.logwarn("Target velocity : {0}\n".format(linear_val))
        # rospy.logwarn("Target angular velocity: {0}\n".format(angular_vel))
        # rospy.logwarn("Current velocity : {0}\n".format(current_vel))
        # rospy.logwarn("Filtered velocity: {0}\n".format(self.vel_lpf.get()))

        #############################
        #  Steering/Yaw controller  #
        #############################

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        # TODO: Consider to add dampening term to change steering
        # when there is significant diff between target
        # and curreng angular velocity
        
        # Check velocity error
        vel_error     = linear_vel - current_vel
        self.last_vel = current_vel
        
        ############################
        #       PID time step      #
        ############################
        
        current_time   = rospy.get_time()
        sample_time    = current_time - self.last_time
        self.last_time = current_time
        
        ############################
        # Throttle step controller #
        ############################
        
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake    = 0
        
        ###########################
        # Simple brake controller #
        ###########################
        
        # Hold the car in place if we are stopped at a light
        if linear_vel == 0.0 and current_vel < 0.1:
            throttle = 0   # [0...1]
            brake    = 400 # Braking torque [Nm]
        # Slow down the car when little to no throttle 
        # and current velocity is faster than target velocity
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0 # [0...1]
            decel    = max(vel_error, self.decel_limit) # Deceleration [m/s^2]
            brake    = self.vehicle_mass * abs(decel) * self.wheel_radius # Braking torque [Nm]
        
        return throttle, brake, steering