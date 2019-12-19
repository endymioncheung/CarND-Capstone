#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree

import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        
        # Initalize ROS node
        rospy.init_node('tl_detector')

        self.pose         = None
        self.waypoints    = None
        self.waypoints_2d = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        `/vehicle/traffic_lights` provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. 
        You'll need to rely on the position of the light and the camera image to predict it.
        '''
        
        ############################################################################
        #                     FOR OFFLINE TESTING PURPOSE ONLY                     #
        # The ROS topic `/vehicle/traffic_lights` is available from the vehicle    #
        # simulator and contains state of light (i.e. red, green or yellow)        #
        ############################################################################
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        
        # Subscribe ROS topic /image_color (front mounted camera data)
        # `/image_color` contains raw image data, image color for the classifier
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
    
        # Load the traffic light config (i.e. image color or preferrably raw image)
        config_string = rospy.get_param("/traffic_light_config")
        self.config   = yaml.safe_load(config_string)
        
        # Publish upcoming red light traffic waypoint
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # Initalize traffic light classifer
        self.bridge   = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        # Initalize traffic light state
        self.state       = TrafficLight.UNKNOWN
        self.last_state  = TrafficLight.UNKNOWN
        self.last_wp     = -1
        self.state_count = 0

        # Continue running
        rospy.spin()

    # Vehicle position callback
    def pose_cb(self, msg):
        # Get vehicle position
        self.pose = msg

    # Waypoints callback
    def waypoints_cb(self, waypoints):
        # Get waypoints
        self.waypoints = waypoints
        
        # Create KDTree for fast 2D waypoint lookup. Log(N) efficient
        # Take first LOOKAHEAD_WPS (i.e. 200) points of way points ahead of car
        # Prevent race condition: make sure the self.waypoints_2d is initalized before subscriber
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    # Traffic lights callback
    def traffic_cb(self, msg):
        # Get traffic light state
        self.lights = msg.lights

    # Image callback
    def image_cb(self, msg):
        """Identify red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        
        # Process the image each time when an image is captured
        self.has_image    = True
        self.camera_image = msg
        
        # Get the closest traffic light waypoint and its state
        light_wp, state   = self.process_traffic_lights()
        # rospy.logwarn("Closest light wp: {0} \n And light state: {1}",format(light_wp,state))

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        
        if self.state != state:
            '''
            If the traffic light state has changed (i.e. green->yellow, yellow->red)
            then start counting the traffic light state from zero
            '''
            self.state_count = 0
            self.state = state
                
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            '''
            Make sure the traffic lights stay before taking any action.
            (i.e. saving the last state and publish upcoming red traffic light).
            
            Check if the traffic light staying consistent before taking any actions
            to prevent cases like the classifier is noisy, or lights changing 
            consecutively too quickly (i.e. green->yellow->red)
            '''
            
            self.last_state = self.state
            
            '''
            Light waypoint updater:
            Only red traffic lights is considered for stopping;
            For all other lights, the car can continue driving 
            '''
            light_wp     = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            
            # Publish red light way point
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            
            '''
            [optional] Consider update the code if the light is yellow
            to prepare to stop, or look at its pose and environment,
            '''
        else:
            '''
            If the traffic light doesn't change or light change
            counts below the STATE_COUNT_THRESHOLD,
            then publish the last red light waypoint
            '''
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            
        # Keep counting the traffic light until the light changes
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # TODO: Implement (hint: use KDTree to search for the closest waypoint)
        # KDTree.query(<point>,<num_items>)[<item_index>]
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        #################################################
        # !!!ATTN!!! FOR OFFLINE TESTING PURPOSE ONLY   #
        ################################################# 
        '''
        Intent is to use your classifier initially with
        off-line testing by returning the traffic light 
        state from the simulator
        '''
        rospy.logwarn("Light State: {0}".format(light.state))
        return light.state
        
        #################################################
        # !!!ATTN!!!       FOR VEHICLE TESTING ONLY     #
        #################################################
        
        # if(not self.has_image):
        #     self.prev_light_loc = None
        #     return False
        #
        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #
        # # Return traffic light classification
        # return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """ Finds closest visible traffic light.
            If one exists, determines its location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        '''
        Closest traffic light
        Each traffic light comes with a traffic line, 
        which is a stop_line for that traffic light
        '''
        closest_light = None
        light         = None

        # Get list of positions that correspond to the line 
        # to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            '''
            Get index of the closest car waypoint using KDTree
            by iterate through the list of traffic lights to find the closest one
            '''
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            # TODO: Find the closest visible traffic light (if one exists)
            '''
            For each traffic light state, start off with the largest distance 
            difference (i.e. 8 intersections). Use list to iterate over 
            rather than using KDTree because small size of the list.
            '''
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get line of traffic (x,y)
                line        = stop_line_positions[i]
                
                # Get index of the stop line waypoint
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                
                # Find closest stop line waypoint index
                # Check the difference between the closest waypoint and car's waypoint
                d = temp_wp_idx - car_wp_idx
                
                '''
                Linear check to see which is the closest light by..
                updating the stop line waypoint if the light waypoint is front of the car
                and the light waypoint does not go past the 8 different light states,
                then set the closest light to light and the line waypoint index
                to the tempoary index
                '''
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    
                    # Closest traffic line in front of the car
                    line_wp_idx   = temp_wp_idx
        
        # Set the closest light state when found
        if closest_light:
            '''
            Return the classified light state (for vehicle testing)
            or the simulated light state (for offline testing)
            '''
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
        
        '''
        Otherwise return -1 and unknown when no closest traffic light is found,
        or if traffic light is detected by the state is UNKNOWN
        '''
        return -1, TrafficLight.UNKNOWN

    '''
    Recommendation for this project if traffic light is not detected or the
    traffic state is UNKNOWN, just keep the car moving.
    '''

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
